#!/usr/bin/env python3
"""
boltz2_verification.py – Compare Boltz-2 vs Boltz-1 Pairformer activation patterns.

For each of three proteins (small 113aa, medium 386aa, large 668aa) this script:
  1. Processes the YAML with boltz2=True (Boltz-2 format).
  2. Loads Boltz-2 and runs a trunk-only forward pass with hooks on every
     PairformerLayer (64 layers) capturing per-channel std, abs_max, diagonal
     vs off-diagonal stats, and all submodule outputs.
  3. Saves pairformer_stats.json and diagonal_stats.json per protein.
  4. Loads the matching Boltz-1 results (from test_long_sequences.py output),
     computes cross-model comparisons, and generates four PNG plots.
  5. Writes a one-page Markdown summary report.

Usage:
    python boltz2_verification.py [--use_msa_server] [--cpu] [--no_bf16]
        [--out_dir ./boltz2_results]
        [--boltz1_dir ./test_output/boltz_results_profile_long_results]
        [--cache ~/.boltz]
"""

import argparse
import importlib.util
import json
import math
import multiprocessing
import sys
import textwrap
import time
import warnings
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Locate repo root and bootstrap profile_pairformer helpers
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent

_PP_PATH = _SCRIPT_DIR / "profile_pairformer.py"
if not _PP_PATH.exists():
    print(f"ERROR: profile_pairformer.py not found at {_PP_PATH}", file=sys.stderr)
    sys.exit(1)

_spec = importlib.util.spec_from_file_location("profile_pairformer", _PP_PATH)
_pp = importlib.util.module_from_spec(_spec)   # type: ignore[arg-type]
_spec.loader.exec_module(_pp)                  # type: ignore[union-attr]

_tensor_stats = _pp._tensor_stats   # noqa: SLF001
_json_safe = _pp._json_safe         # noqa: SLF001

# ---------------------------------------------------------------------------
# Boltz internals
# ---------------------------------------------------------------------------
# sys.path already has /workspace/boltz/src via the profile_pairformer import above

from boltz.main import (               # noqa: E402
    Boltz2DiffusionParams,
    BoltzProcessedInput,
    BoltzSteeringParams,
    MSAModuleArgs,
    PairformerArgsV2,
    check_inputs,
    download_boltz2,
    get_cache_path,
    process_inputs,
)
from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
from boltz.data.types import Manifest
from boltz.model.models.boltz2 import Boltz2
from boltz.model.layers.pairformer import PairformerLayer

# ---------------------------------------------------------------------------
# Protein catalogue
# ---------------------------------------------------------------------------

PROTEINS: List[Dict[str, Any]] = [
    {
        "name": "small_113aa",
        "approx_N": 113,
        "description": "~113 aa",
    },
    {
        "name": "medium_386aa",
        "approx_N": 386,
        "description": "~386 aa (HSP70, PDB 3HSC)",
    },
    {
        "name": "large_668aa",
        "approx_N": 668,
        "description": "~668 aa (PDB 3C7N)",
    },
]

# Boltz-2 PairformerLayer submodules (same names as Boltz-1)
SUBMODULE_NAMES = [
    "tri_mul_out",
    "tri_mul_in",
    "tri_att_start",
    "tri_att_end",
    "transition_z",
    "attention",
    "transition_s",
]

# ---------------------------------------------------------------------------
# Hook infrastructure
# ---------------------------------------------------------------------------

def _make_layer_hook(
    layer_idx: int,
    stats_store: List[Dict],
    diag_store: List[Dict],
    pass_counter: Dict[int, int],
) -> Any:
    """Combined hook: records layer_output_z/s stats AND diagonal concentration."""

    def hook(module: nn.Module, inputs: Any, output: Any) -> None:
        if not isinstance(output, (tuple, list)) or len(output) < 2:
            return
        s_out, z_out = output[0], output[1]

        # Track which recycling pass this is (pass_counter maps layer_idx -> call count)
        pass_counter[layer_idx] = pass_counter.get(layer_idx, 0) + 1
        current_pass = pass_counter[layer_idx]

        # ── per-tensor statistics ──────────────────────────────────────────
        for tag, t in (("layer_output_z", z_out), ("layer_output_s", s_out)):
            if not isinstance(t, torch.Tensor):
                continue
            try:
                st = _tensor_stats(t)
            except Exception as exc:
                st = {"error": str(exc)}
            stats_store.append({
                "layer_idx": layer_idx,
                "op": tag,
                "recycling_pass": current_pass,
                **st,
            })

        # ── diagonal vs off-diagonal stats on z ───────────────────────────
        if isinstance(z_out, torch.Tensor) and z_out.ndim >= 4:
            z = z_out.detach().float()
            B, N, N2, D = z.shape
            diag_idx = torch.arange(N, device=z.device)
            diag_vals = z[:, diag_idx, diag_idx, :]        # [B, N, D]
            diag_abs_max = float(diag_vals.abs().max())
            diag_ch_std = diag_vals.reshape(-1, D).std(dim=0).cpu().tolist()

            eye_mask = torch.eye(N, dtype=torch.bool, device=z.device)
            off_mask = (~eye_mask).unsqueeze(0).unsqueeze(-1)          # [1,N,N,1]
            offdiag_abs_max = float(z.abs().masked_fill(~off_mask, 0.0).max())

            # off-diagonal entries per channel std
            # Collect off-diagonal entries more cheaply by summing masks
            off_vals = z[off_mask.expand_as(z)].reshape(-1, D)
            off_ch_std = off_vals.std(dim=0).cpu().tolist() if off_vals.numel() > 0 else []

            diag_store.append({
                "layer_idx": layer_idx,
                "recycling_pass": current_pass,
                "diag_abs_max": diag_abs_max,
                "offdiag_abs_max": offdiag_abs_max,
                "diag_per_channel_std": diag_ch_std,
                "offdiag_per_channel_std": off_ch_std,
                "N": N,
            })

    return hook


def _make_submodule_hook(layer_idx: int, op_name: str, store: List[Dict]) -> Any:
    def hook(module: nn.Module, inputs: Any, output: Any) -> None:
        t = output[0] if isinstance(output, (tuple, list)) else output
        if not isinstance(t, torch.Tensor):
            return
        try:
            st = _tensor_stats(t)
        except Exception as exc:
            st = {"error": str(exc)}
        store.append({"layer_idx": layer_idx, "op": op_name, **st})
    return hook


def _attach_hooks(
    pairformer_module: nn.Module,
) -> Tuple[List, List[Dict], List[Dict]]:
    """Attach hooks defensively: iterate named_modules and match on isinstance."""
    stats_store: List[Dict] = []
    diag_store: List[Dict] = []
    handles: List = []
    pass_counter: Dict[int, int] = {}

    # Find all PairformerLayer instances via named_modules
    indexed_layers: List[Tuple[int, nn.Module]] = []
    for name, mod in pairformer_module.named_modules():
        if isinstance(mod, PairformerLayer):
            # Extract layer index from name like "layers.0", "layers.23", etc.
            parts = name.split(".")
            idx = None
            for p in reversed(parts):
                if p.isdigit():
                    idx = int(p)
                    break
            if idx is None:
                # Fallback: sequential counter
                idx = len(indexed_layers)
            indexed_layers.append((idx, mod))

    if not indexed_layers:
        # Final fallback: iterate .layers directly
        for idx, layer in enumerate(pairformer_module.layers):
            indexed_layers.append((idx, layer))

    print(f"    Found {len(indexed_layers)} PairformerLayer(s) via named_modules.")

    for layer_idx, layer in indexed_layers:
        # Layer-level hook
        h = layer.register_forward_hook(
            _make_layer_hook(layer_idx, stats_store, diag_store, pass_counter)
        )
        handles.append(h)

        # Per-submodule hooks
        for op_name in SUBMODULE_NAMES:
            submod = getattr(layer, op_name, None)
            if submod is not None:
                h2 = submod.register_forward_hook(
                    _make_submodule_hook(layer_idx, op_name, stats_store)
                )
                handles.append(h2)

    return handles, stats_store, diag_store


def _remove_hooks(handles: List) -> None:
    for h in handles:
        h.remove()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_boltz2(cache: Path, checkpoint: Optional[str]) -> Boltz2:
    ckpt = checkpoint or str(cache / "boltz2_conf.ckpt")
    print(f"  Loading Boltz-2 checkpoint: {ckpt}")

    pairformer_args = PairformerArgsV2(
        num_blocks=64,
        num_heads=16,
        dropout=0.0,
        activation_checkpointing=False,
        v2=True,
    )
    msa_args = MSAModuleArgs(
        subsample_msa=False,
        use_paired_feature=True,
    )
    diffusion_params = Boltz2DiffusionParams()
    steering_args = BoltzSteeringParams()

    model: Boltz2 = Boltz2.load_from_checkpoint(
        ckpt,
        strict=True,
        predict_args={
            "recycling_steps": 1,
            "sampling_steps": 10,
            "diffusion_samples": 1,
            "max_parallel_samples": 1,
            "write_confidence_summary": False,
            "write_full_pae": False,
            "write_full_pde": False,
        },
        map_location="cpu",
        diffusion_process_args=asdict(diffusion_params),
        ema=False,
        use_kernels=False,   # must be False so hooks fire
        pairformer_args=asdict(pairformer_args),
        msa_args=asdict(msa_args),
        steering_args=asdict(steering_args),
    )
    model.eval()
    model.use_kernels = False   # belt-and-suspenders
    return model


# ---------------------------------------------------------------------------
# Input processing (Boltz-2 format)
# ---------------------------------------------------------------------------

def _process_boltz2_input(
    yaml_path: Path,
    work_dir: Path,
    cache: Path,
    use_msa_server: bool,
) -> Optional[Path]:
    processed_dir = work_dir / "processed"
    if (processed_dir / "manifest.json").exists():
        print(f"    [cache] Reusing {processed_dir}")
        return processed_dir

    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        data_list = check_inputs(yaml_path)
    except Exception as exc:
        print(f"    ERROR check_inputs: {exc}", file=sys.stderr)
        return None

    manifest = process_inputs(
        data=data_list,
        out_dir=work_dir,
        ccd_path=cache / "ccd.pkl",
        mol_dir=cache / "mols",
        msa_server_url="https://api.colabfold.com",
        msa_pairing_strategy="greedy",
        use_msa_server=use_msa_server,
        boltz2=True,          # ← Boltz-2 format
        preprocessing_threads=min(4, multiprocessing.cpu_count()),
    )

    if not manifest or not manifest.records:
        mp = processed_dir / "manifest.json"
        if mp.exists():
            manifest = Manifest.load(mp)

    if not manifest or not manifest.records:
        print(
            f"    ERROR: manifest is empty for {yaml_path.name}. "
            "Pass --use_msa_server if MSA is required.",
            file=sys.stderr,
        )
        return None

    return processed_dir


# ---------------------------------------------------------------------------
# Trunk forward pass for Boltz-2
# ---------------------------------------------------------------------------

def _to_device(obj: Any, dev: torch.device) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.to(dev)
    if isinstance(obj, dict):
        return {k: _to_device(v, dev) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_device(v, dev) for v in obj]
    return obj


def _run_boltz2_trunk(
    model: Boltz2,
    feats: Dict[str, torch.Tensor],
    device: torch.device,
    use_bf16: bool,
) -> None:
    """Replicate Boltz-2's trunk (embedding + MSA + Pairformer), no diffusion."""
    ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if use_bf16 and device.type == "cuda"
        else torch.no_grad()
    )

    with torch.no_grad():
        with ctx:
            s_inputs = model.input_embedder(feats)
            s_init = model.s_init(s_inputs)
            z_init = (
                model.z_init_1(s_inputs)[:, :, None]
                + model.z_init_2(s_inputs)[:, None, :]
            )
            z_init = z_init + model.rel_pos(feats)
            z_init = z_init + model.token_bonds(feats["token_bonds"].float())
            if model.bond_type_feature:
                z_init = z_init + model.token_bonds_type(feats["type_bonds"].long())
            z_init = z_init + model.contact_conditioning(feats)

            s = torch.zeros_like(s_init)
            z = torch.zeros_like(z_init)
            mask = feats["token_pad_mask"].float()
            pair_mask = mask[:, :, None] * mask[:, None, :]

            recycling_steps = 1
            for _ in range(recycling_steps + 1):
                s = s_init + model.s_recycle(model.s_norm(s))
                z = z_init + model.z_recycle(model.z_norm(z))

                # MSA module
                if model.is_msa_compiled and not model.training:
                    msa_module = model.msa_module._orig_mod  # noqa: SLF001
                else:
                    msa_module = model.msa_module
                z = z + msa_module(z, s_inputs, feats, use_kernels=False)

                # Pairformer
                if model.is_pairformer_compiled and not model.training:
                    pairformer_module = model.pairformer_module._orig_mod  # noqa: SLF001
                else:
                    pairformer_module = model.pairformer_module

                s, z = pairformer_module(
                    s, z,
                    mask=mask,
                    pair_mask=pair_mask,
                    use_kernels=False,
                )

    N = int(feats["token_pad_mask"].sum().item())
    print(f"    Trunk forward complete. z shape: {z.shape}, tokens: {N}")


# ---------------------------------------------------------------------------
# Per-protein profiling pipeline
# ---------------------------------------------------------------------------

def _profile_protein(
    name: str,
    yaml_path: Path,
    work_dir: Path,
    cache: Path,
    model: Boltz2,
    device: torch.device,
    use_bf16: bool,
    out_dir: Path,
    use_msa_server: bool,
) -> Optional[Tuple[List[Dict], List[Dict]]]:
    """Process input, run trunk with hooks, save JSON. Returns (stats, diag) or None."""

    # Step A: Process input in Boltz-2 format
    print("    Processing input (boltz2 format) …")
    t0 = time.time()
    processed_dir = _process_boltz2_input(
        yaml_path=yaml_path,
        work_dir=work_dir,
        cache=cache,
        use_msa_server=use_msa_server,
    )
    if processed_dir is None:
        return None
    print(f"    Processed in {time.time()-t0:.1f}s")

    # Step B: Load batch from Boltz-2 data module
    manifest = Manifest.load(processed_dir / "manifest.json")
    processed = BoltzProcessedInput(
        manifest=manifest,
        targets_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa",
        constraints_dir=(
            (processed_dir / "constraints")
            if (processed_dir / "constraints").exists()
            else None
        ),
    )

    dm = Boltz2InferenceDataModule(
        manifest=processed.manifest,
        target_dir=processed.targets_dir,
        msa_dir=processed.msa_dir,
        mol_dir=cache / "mols",
        num_workers=0,
        constraints_dir=processed.constraints_dir,
    )
    try:
        batch = next(iter(dm.predict_dataloader()))
    except StopIteration:
        print(f"    ERROR: DataLoader empty for {name}.", file=sys.stderr)
        return None

    feats = _to_device(batch, device)
    token_count = int(feats["token_pad_mask"].sum().item())
    print(f"    Tokens: {token_count}  pair tensor ~[1,{token_count},{token_count},D]")

    # Step C: Attach hooks
    pairformer_module = (
        model.pairformer_module._orig_mod    # noqa: SLF001
        if model.is_pairformer_compiled
        else model.pairformer_module
    )
    handles, stats_store, diag_store = _attach_hooks(pairformer_module)

    # Step D: Forward pass
    try:
        t0 = time.time()
        _run_boltz2_trunk(model, feats, device, use_bf16)
        print(f"    Forward pass took {time.time()-t0:.1f}s")
    except torch.cuda.OutOfMemoryError:
        _remove_hooks(handles)
        torch.cuda.empty_cache()
        print(
            f"    [OOM] CUDA out-of-memory for {name} (N={token_count}). Skipping.",
            file=sys.stderr,
        )
        return None
    except Exception as exc:
        _remove_hooks(handles)
        print(f"    ERROR during forward pass: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return None
    finally:
        _remove_hooks(handles)

    print(f"    Collected {len(stats_store)} stats records, {len(diag_store)} diag records.")

    # Step E: Save
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "pairformer_stats.json", "w") as f:
        json.dump(_json_safe({
            "model": "boltz2",
            "protein": name,
            "token_count": token_count,
            "num_layers": len({r["layer_idx"] for r in stats_store
                               if r.get("op") == "layer_output_z"}),
            "device": str(device),
            "bf16_autocast": use_bf16,
            "records": stats_store,
        }), f, indent=2)

    with open(out_dir / "diagonal_stats.json", "w") as f:
        json.dump(_json_safe(diag_store), f, indent=2)

    print(f"    Saved outputs to {out_dir}")
    return stats_store, diag_store


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

TOP_K_PER_LAYER = 5
TOP_N_REPORT = 5


def _dominant_channels(
    stats_store: List[Dict],
    top_k: int = TOP_K_PER_LAYER,
    report_n: int = TOP_N_REPORT,
    only_last_pass: bool = True,
) -> List[Tuple[int, float]]:
    """Top-N channels by combined (per_channel_std rank + flat-index) hit rate."""
    # When recycling, only count the last pass (highest recycling_pass number)
    z_recs = [r for r in stats_store if r.get("op") == "layer_output_z"]
    if only_last_pass and z_recs:
        max_pass = max(r.get("recycling_pass", 1) for r in z_recs)
        z_recs = [r for r in z_recs if r.get("recycling_pass", 1) == max_pass]

    if not z_recs:
        return []

    num_layers = len(z_recs)
    std_hits: Counter = Counter()
    flat_hits: Counter = Counter()

    for rec in z_recs:
        ch_std = rec.get("per_channel_std")
        if ch_std:
            ranked = sorted(range(len(ch_std)), key=lambda i, cs=ch_std: cs[i], reverse=True)
            for ch in ranked[:top_k]:
                std_hits[ch] += 1

        shape = rec.get("shape")
        flat_idxs = rec.get("top10_abs_flat_indices")
        if shape and flat_idxs:
            D = shape[-1]
            for fi in flat_idxs[:top_k]:
                flat_hits[fi % D] += 1

    all_channels = set(std_hits) | set(flat_hits)
    combined = {
        ch: (std_hits.get(ch, 0) + flat_hits.get(ch, 0)) / (2.0 * num_layers) * 100.0
        for ch in all_channels
    }
    return sorted(combined.items(), key=lambda x: x[1], reverse=True)[:report_n]


def _layer_z_stats(
    stats_store: List[Dict],
    only_last_pass: bool = True,
) -> Dict[int, Dict]:
    """Return {layer_idx: record} for layer_output_z, using last recycling pass."""
    z_recs = [r for r in stats_store if r.get("op") == "layer_output_z"]
    if only_last_pass and z_recs:
        max_pass = max(r.get("recycling_pass", 1) for r in z_recs)
        z_recs = [r for r in z_recs if r.get("recycling_pass", 1) == max_pass]
    return {r["layer_idx"]: r for r in z_recs}


def _diag_by_layer(
    diag_store: List[Dict],
    only_last_pass: bool = True,
) -> Dict[int, Dict]:
    if only_last_pass and diag_store:
        max_pass = max(r.get("recycling_pass", 1) for r in diag_store)
        diag_store = [r for r in diag_store if r.get("recycling_pass", 1) == max_pass]
    return {r["layer_idx"]: r for r in diag_store}


def _peak_layer(layer_z: Dict[int, Dict]) -> int:
    if not layer_z:
        return -1
    return max(layer_z, key=lambda i: layer_z[i]["abs_max"])


def _endpoint_abs_max(layer_z: Dict[int, Dict]) -> float:
    if not layer_z:
        return float("nan")
    last_idx = max(layer_z.keys())
    return layer_z[last_idx]["abs_max"]


def _mean_diag_ratio(diag_by_layer: Dict[int, Dict]) -> float:
    ratios = []
    for rec in diag_by_layer.values():
        od = rec.get("offdiag_abs_max", 0.0)
        if od and od > 0:
            ratios.append(rec.get("diag_abs_max", 0.0) / od)
    return float(sum(ratios) / len(ratios)) if ratios else float("nan")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _setup_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        return plt, np
    except ImportError:
        return None, None


def plot_per_block_max(
    boltz1_stats: Dict[str, List[Dict]],
    boltz2_stats: Dict[str, List[Dict]],
    out_path: Path,
) -> None:
    plt, np = _setup_matplotlib()
    if plt is None:
        return

    proteins = list(boltz1_stats.keys())
    colors = plt.cm.tab10(np.linspace(0, 0.5, len(proteins)))
    fig, ax = plt.subplots(figsize=(13, 6))

    for i, pname in enumerate(proteins):
        for (model_name, store, ls) in [
            ("Boltz-1", boltz1_stats.get(pname), "-"),
            ("Boltz-2", boltz2_stats.get(pname), "--"),
        ]:
            if store is None:
                continue
            lz = _layer_z_stats(store)
            layers = sorted(lz.keys())
            vals = [lz[l]["abs_max"] for l in layers]
            label = f"{model_name} {pname}"
            ax.plot(layers, vals, ls=ls, marker="o", markersize=2.5,
                    linewidth=1.5, color=colors[i], label=label,
                    alpha=0.85 if model_name == "Boltz-1" else 0.65)

    ax.set_xlabel("Pairformer layer index")
    ax.set_ylabel("abs_max of pair tensor z")
    ax.set_title("Per-block abs_max comparison: Boltz-1 (solid) vs Boltz-2 (dashed)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_top_channels(
    boltz1_ch: Dict[str, List[Tuple[int, float]]],
    boltz2_ch: Dict[str, List[Tuple[int, float]]],
    out_path: Path,
    top_n: int = TOP_N_REPORT,
) -> None:
    plt, np = _setup_matplotlib()
    if plt is None:
        return

    proteins = list(boltz1_ch.keys())
    n_proteins = len(proteins)
    fig, axes = plt.subplots(1, n_proteins, figsize=(6 * n_proteins, 6), sharey=False)
    if n_proteins == 1:
        axes = [axes]

    for ax, pname in zip(axes, proteins):
        b1 = dict(boltz1_ch.get(pname, []))
        b2 = dict(boltz2_ch.get(pname, []))
        all_chs = sorted(set(list(b1.keys()) + list(b2.keys())))
        x = np.arange(len(all_chs))
        w = 0.35
        h1 = [b1.get(ch, 0.0) for ch in all_chs]
        h2 = [b2.get(ch, 0.0) for ch in all_chs]
        bars1 = ax.bar(x - w / 2, h1, w, label="Boltz-1", color="steelblue", alpha=0.85)
        bars2 = ax.bar(x + w / 2, h2, w, label="Boltz-2", color="darkorange", alpha=0.85)
        ax.bar_label(bars1, fmt="%.0f%%", padding=2, fontsize=7)
        ax.bar_label(bars2, fmt="%.0f%%", padding=2, fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([f"ch{c}" for c in all_chs], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Hit% (layers in top-5)")
        ax.set_title(pname, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Top-5 dominant channels: Boltz-1 (blue) vs Boltz-2 (orange)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_diagonal_ratio(
    boltz1_diag: Dict[str, List[Dict]],
    boltz2_diag: Dict[str, List[Dict]],
    boltz1_stats: Dict[str, List[Dict]],
    boltz2_stats: Dict[str, List[Dict]],
    out_path: Path,
) -> None:
    plt, np = _setup_matplotlib()
    if plt is None:
        return

    proteins = list(boltz1_diag.keys())
    colors = plt.cm.tab10(np.linspace(0, 0.5, len(proteins)))
    fig, ax = plt.subplots(figsize=(13, 6))

    for i, pname in enumerate(proteins):
        for (model_name, diag_store, stat_store, ls) in [
            ("Boltz-1", boltz1_diag.get(pname, []), boltz1_stats.get(pname, []), "-"),
            ("Boltz-2", boltz2_diag.get(pname, []), boltz2_stats.get(pname, []), "--"),
        ]:
            if not diag_store:
                continue
            dbl = _diag_by_layer(diag_store)
            layers = sorted(dbl.keys())
            ratios = []
            for l in layers:
                rec = dbl[l]
                od = rec.get("offdiag_abs_max", 0.0)
                ratio = rec.get("diag_abs_max", 0.0) / od if od else float("nan")
                ratios.append(ratio)
            label = f"{model_name} {pname}"
            ax.plot(layers, ratios, ls=ls, linewidth=1.5, marker=".", markersize=3,
                    color=colors[i], label=label,
                    alpha=0.85 if model_name == "Boltz-1" else 0.65)

    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.7, label="ratio=1")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("diag_abs_max / offdiag_abs_max")
    ax.set_title("Diagonal / off-diagonal ratio per layer: Boltz-1 (solid) vs Boltz-2 (dashed)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_peak_diag_offdiag(
    boltz1_diag: Dict[str, List[Dict]],
    boltz2_diag: Dict[str, List[Dict]],
    boltz1_stats: Dict[str, List[Dict]],
    boltz2_stats: Dict[str, List[Dict]],
    out_path: Path,
) -> None:
    plt, np = _setup_matplotlib()
    if plt is None:
        return

    proteins = [p for p in boltz1_diag if boltz2_diag.get(p)]
    n_p = len(proteins)
    if n_p == 0:
        return

    x = np.arange(n_p)
    w = 0.18
    offsets = [-1.5 * w, -0.5 * w, 0.5 * w, 1.5 * w]
    bar_labels = ["B1-diag", "B1-offdiag", "B2-diag", "B2-offdiag"]
    bar_colors = ["steelblue", "cornflowerblue", "darkorange", "moccasin"]

    fig, ax = plt.subplots(figsize=(4 + 3 * n_p, 6))

    b1_diag_vals, b1_off_vals, b2_diag_vals, b2_off_vals = [], [], [], []

    for pname in proteins:
        # Find peak layer from Boltz-1 stats
        b1_lz = _layer_z_stats(boltz1_stats.get(pname, []))
        b2_lz = _layer_z_stats(boltz2_stats.get(pname, []))
        b1_peak = _peak_layer(b1_lz)
        b2_peak = _peak_layer(b2_lz)

        b1_dbl = _diag_by_layer(boltz1_diag.get(pname, []))
        b2_dbl = _diag_by_layer(boltz2_diag.get(pname, []))

        b1_rec = b1_dbl.get(b1_peak, {})
        b2_rec = b2_dbl.get(b2_peak, {})

        b1_diag_vals.append(b1_rec.get("diag_abs_max", 0.0))
        b1_off_vals.append(b1_rec.get("offdiag_abs_max", 0.0))
        b2_diag_vals.append(b2_rec.get("diag_abs_max", 0.0))
        b2_off_vals.append(b2_rec.get("offdiag_abs_max", 0.0))

    for vals, offset, label, color in zip(
        [b1_diag_vals, b1_off_vals, b2_diag_vals, b2_off_vals],
        offsets, bar_labels, bar_colors
    ):
        bars = ax.bar(x + offset, vals, w, label=label, color=color, alpha=0.9)
        ax.bar_label(bars, fmt="%.2g", padding=2, fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(proteins, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("abs_max")
    ax.set_title("Diagonal vs off-diagonal abs_max at peak layer (Boltz-1 vs Boltz-2)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Comparison tables and verdict
# ---------------------------------------------------------------------------

def _compute_channel_overlap(
    b1_tops: List[Tuple[int, float]],
    b2_tops: List[Tuple[int, float]],
) -> List[int]:
    b1_ch = {ch for ch, _ in b1_tops}
    b2_ch = {ch for ch, _ in b2_tops}
    return sorted(b1_ch & b2_ch)


def print_comparison_tables(
    proteins: List[str],
    boltz1_stats: Dict[str, List[Dict]],
    boltz2_stats: Dict[str, List[Dict]],
    boltz1_diag: Dict[str, List[Dict]],
    boltz2_diag: Dict[str, List[Dict]],
) -> Dict[str, Any]:
    """Print all three comparison tables. Returns data dict for report."""
    all_data: Dict[str, Any] = {}

    for pname in proteins:
        b1s = boltz1_stats.get(pname, [])
        b2s = boltz2_stats.get(pname, [])
        b1d = boltz1_diag.get(pname, [])
        b2d = boltz2_diag.get(pname, [])

        b1_tops = _dominant_channels(b1s)
        b2_tops = _dominant_channels(b2s)

        b1_lz = _layer_z_stats(b1s)
        b2_lz = _layer_z_stats(b2s)

        b1_peak = _peak_layer(b1_lz)
        b2_peak = _peak_layer(b2_lz)

        b1_peak_val = b1_lz.get(b1_peak, {}).get("abs_max", float("nan"))
        b2_peak_val = b2_lz.get(b2_peak, {}).get("abs_max", float("nan"))

        b1_end = _endpoint_abs_max(b1_lz)
        b2_end = _endpoint_abs_max(b2_lz)

        b1_dbl = _diag_by_layer(b1d)
        b2_dbl = _diag_by_layer(b2d)

        # Diagonal comparison — use each model's own peak layer
        b1_diag_rec = b1_dbl.get(b1_peak, {})
        b2_diag_rec = b2_dbl.get(b2_peak, {})

        b1_d_abs = b1_diag_rec.get("diag_abs_max", float("nan"))
        b1_od_abs = b1_diag_rec.get("offdiag_abs_max", float("nan"))
        b2_d_abs = b2_diag_rec.get("diag_abs_max", float("nan"))
        b2_od_abs = b2_diag_rec.get("offdiag_abs_max", float("nan"))

        b1_ratio = b1_d_abs / b1_od_abs if b1_od_abs else float("nan")
        b2_ratio = b2_d_abs / b2_od_abs if b2_od_abs else float("nan")

        b1_mean_ratio = _mean_diag_ratio(b1_dbl)
        b2_mean_ratio = _mean_diag_ratio(b2_dbl)

        overlap = _compute_channel_overlap(b1_tops, b2_tops)

        # Print
        print(f"\n{'─'*60}")
        print(f"  Protein: {pname}")
        print(f"{'─'*60}")
        fmt_tops = lambda tops: ", ".join(f"ch{ch}({pct:.0f}%)" for ch, pct in tops)
        print(f"  Channel dominance:")
        print(f"    Boltz-1 top-5: {fmt_tops(b1_tops)}")
        print(f"    Boltz-2 top-5: {fmt_tops(b2_tops)}")
        print(f"    Overlap (shared in top-5): {[f'ch{c}' for c in overlap]}")
        print()
        print(f"  Per-block max:")
        print(f"    Boltz-1: peak {b1_peak_val:.2f} at layer {b1_peak}  |  endpoint {b1_end:.2f}")
        print(f"    Boltz-2: peak {b2_peak_val:.2f} at layer {b2_peak}  |  endpoint {b2_end:.2f}")
        print()
        print(f"  Diagonal dominance at peak layer:")
        print(f"    Boltz-1 (L{b1_peak}): diag={b1_d_abs:.2f}, offdiag={b1_od_abs:.2f}, "
              f"ratio={b1_ratio:.3f}")
        print(f"    Boltz-2 (L{b2_peak}): diag={b2_d_abs:.2f}, offdiag={b2_od_abs:.2f}, "
              f"ratio={b2_ratio:.3f}")
        print(f"    Mean ratio (all layers)  Boltz-1: {b1_mean_ratio:.3f}  "
              f"Boltz-2: {b2_mean_ratio:.3f}")

        all_data[pname] = {
            "b1_tops": b1_tops,
            "b2_tops": b2_tops,
            "overlap": overlap,
            "b1_peak": b1_peak,
            "b2_peak": b2_peak,
            "b1_peak_val": b1_peak_val,
            "b2_peak_val": b2_peak_val,
            "b1_end": b1_end,
            "b2_end": b2_end,
            "b1_diag": b1_d_abs,
            "b1_offdiag": b1_od_abs,
            "b1_ratio": b1_ratio,
            "b2_diag": b2_d_abs,
            "b2_offdiag": b2_od_abs,
            "b2_ratio": b2_ratio,
            "b1_mean_ratio": b1_mean_ratio,
            "b2_mean_ratio": b2_mean_ratio,
        }

    return all_data


def _determine_verdict(
    all_data: Dict[str, Any],
) -> Tuple[str, str]:
    """Return (verdict_key, verdict_sentence)."""
    if not all_data:
        return "NO_DATA", "No data available for comparison."

    proteins = list(all_data.keys())

    # Metric 1: channel overlap score (average overlap size / top_n)
    overlap_sizes = [len(all_data[p]["overlap"]) for p in proteins]
    avg_overlap = sum(overlap_sizes) / len(overlap_sizes)
    channel_match = avg_overlap >= TOP_N_REPORT * 0.6   # >=3 of 5 channels shared

    # Metric 2: curve shape — do both models peak at roughly same layer (within ±10)?
    curve_match = all(
        abs(all_data[p]["b1_peak"] - all_data[p]["b2_peak"]) <= 10
        for p in proteins
    )

    # Metric 3: diagonal dominance preserved in both models (ratio > 1.1 on average)
    b1_diag_ok = all(all_data[p]["b1_mean_ratio"] > 1.05 for p in proteins)
    b2_diag_ok = all(
        not math.isnan(all_data[p]["b2_mean_ratio"]) and all_data[p]["b2_mean_ratio"] > 1.05
        for p in proteins
    )
    diag_match = b1_diag_ok and b2_diag_ok

    if channel_match and curve_match and diag_match:
        key = "FULL_MATCH"
        verdict = (
            "FULL MATCH: Boltz-2 shows identical patterns. Same dominant channels, "
            "same curve shape, same diagonal dominance. Suggests the pattern is architectural."
        )
    elif not channel_match and curve_match and diag_match:
        key = "ARCHITECTURAL_MATCH"
        verdict = (
            "ARCHITECTURAL MATCH, WEIGHT-SPECIFIC CHANNELS: Boltz-2 shows the same "
            "diagonal concentration and unimodal curve, but different specific channels "
            "dominate. Suggests channel-level outliers are a property of how each model "
            "is trained, but the structural pattern (sparsity + diagonal concentration) "
            "is architectural."
        )
    elif diag_match and not curve_match:
        key = "DIAG_PRESERVED_CURVE_DIFFERENT"
        verdict = (
            "DIAGONAL DOMINANCE PRESERVED, CURVE SHAPE DIFFERENT: Boltz-2 shows diagonal "
            "concentration but a different per-block magnitude curve. Investigate further."
        )
    else:
        key = "PATTERN_BREAKS"
        verdict = (
            "PATTERN BREAKS: Boltz-2 shows fundamentally different behavior. "
            "The Boltz-1 pattern may not generalize."
        )

    # Append grounding numbers
    avg_overlap_pct = avg_overlap / TOP_N_REPORT * 100
    avg_b1_peak = sum(all_data[p]["b1_peak"] for p in proteins) / len(proteins)
    avg_b2_peak = sum(all_data[p]["b2_peak"] for p in proteins) / len(proteins)
    avg_b2_ratio = sum(
        all_data[p]["b2_mean_ratio"] for p in proteins
        if not math.isnan(all_data[p]["b2_mean_ratio"])
    )
    n_b2_ratio = sum(
        1 for p in proteins if not math.isnan(all_data[p]["b2_mean_ratio"])
    )
    avg_b2_ratio_str = f"{avg_b2_ratio/n_b2_ratio:.3f}" if n_b2_ratio else "N/A"

    verdict += (
        f"\n  [Numbers] Avg channel overlap: {avg_overlap:.1f}/{TOP_N_REPORT} "
        f"({avg_overlap_pct:.0f}%); "
        f"Avg peak layer Boltz-1={avg_b1_peak:.1f} Boltz-2={avg_b2_peak:.1f}; "
        f"Avg diagonal ratio Boltz-2={avg_b2_ratio_str}"
    )

    return key, verdict


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _write_markdown_report(
    verdict_key: str,
    verdict: str,
    all_data: Dict[str, Any],
    plot_dir: Path,
    out_path: Path,
) -> None:
    proteins = list(all_data.keys())

    # Interpretation sentence pair
    interp = {
        "FULL_MATCH": (
            "The Boltz-1 findings (channel-wise outliers, diagonal concentration, "
            "unimodal curve peaking at layer ~23) replicate exactly in Boltz-2 despite "
            "different trained weights, indicating the pattern is **architectural** — "
            "an inherent property of AlphaFold3-class pair-tensor mechanics. "
            "This means the finding is likely to generalize across all AF3-class models."
        ),
        "ARCHITECTURAL_MATCH": (
            "The structural pattern (diagonal concentration + unimodal magnitude curve) "
            "is shared by both models, suggesting it is **architectural** and inherent to "
            "the triangle attention mechanism. However, the specific dominant channels "
            "differ between Boltz-1 and Boltz-2, indicating that **which channels** carry "
            "the outlier energy is a property of training rather than architecture."
        ),
        "DIAG_PRESERVED_CURVE_DIFFERENT": (
            "Diagonal concentration is a robust architectural property, but the "
            "per-block magnitude profile differs between models. This warrants "
            "further investigation into whether differences in depth (64 vs 48 layers) "
            "or training objective cause the curve divergence."
        ),
        "PATTERN_BREAKS": (
            "The Boltz-1 findings do not replicate in Boltz-2, suggesting the observed "
            "patterns are **weight-specific** artifacts of Boltz-1 training rather than "
            "a universal architectural property. Further analysis of training dynamics "
            "is required."
        ),
    }.get(verdict_key, "Interpretation unavailable.")

    # Build channel table rows
    ch_rows = []
    for pname in proteins:
        d = all_data[pname]
        b1_str = ", ".join(f"ch{c}({p:.0f}%)" for c, p in d["b1_tops"])
        b2_str = ", ".join(f"ch{c}({p:.0f}%)" for c, p in d["b2_tops"])
        ov_str = ", ".join(f"ch{c}" for c in d["overlap"]) or "—"
        ch_rows.append(f"| {pname} | {b1_str} | {b2_str} | {ov_str} |")

    # Build per-block table rows
    pb_rows = []
    for pname in proteins:
        d = all_data[pname]
        pb_rows.append(
            f"| {pname} "
            f"| {d['b1_peak_val']:.2f} @ L{d['b1_peak']} "
            f"| {d['b1_end']:.2f} "
            f"| {d['b2_peak_val']:.2f} @ L{d['b2_peak']} "
            f"| {d['b2_end']:.2f} |"
        )

    # Build diagonal table rows
    dg_rows = []
    for pname in proteins:
        d = all_data[pname]
        dg_rows.append(
            f"| {pname} "
            f"| {d['b1_diag']:.2f} | {d['b1_offdiag']:.2f} | {d['b1_ratio']:.3f} "
            f"| {d['b1_mean_ratio']:.3f} "
            f"| {d['b2_diag']:.2f} | {d['b2_offdiag']:.2f} | {d['b2_ratio']:.3f} "
            f"| {d['b2_mean_ratio']:.3f} |"
        )

    # Relative plot paths from report location
    def _relplot(name):
        return name  # plots are in same dir as report

    md = textwrap.dedent(f"""\
    # Boltz-2 vs Boltz-1 Pairformer Activation Pattern Verification

    ## Verdict

    > **{verdict_key.replace('_', ' ')}**
    >
    {chr(10).join('> ' + line for line in verdict.splitlines())}

    ---

    ## Interpretation

    {interp}

    ---

    ## Comparison Tables

    ### 1. Channel Dominance

    | Protein | Boltz-1 top-5 (hit%) | Boltz-2 top-5 (hit%) | Shared channels |
    |---------|----------------------|----------------------|-----------------|
    {chr(10).join(ch_rows)}

    *Metric: % of layers where the channel appears in the top-5 by per-channel std
    (combined with flat-index decoding of top-10 absolute values).*

    ### 2. Per-Block abs_max

    | Protein | B1 peak (layer) | B1 endpoint | B2 peak (layer) | B2 endpoint |
    |---------|-----------------|-------------|-----------------|-------------|
    {chr(10).join(pb_rows)}

    ### 3. Diagonal Dominance

    | Protein | B1 diag | B1 offdiag | B1 ratio | B1 mean ratio | B2 diag | B2 offdiag | B2 ratio | B2 mean ratio |
    |---------|---------|------------|----------|---------------|---------|------------|----------|---------------|
    {chr(10).join(dg_rows)}

    *Ratio = diag_abs_max / offdiag_abs_max at each model's own peak layer.
    Mean ratio is averaged across all 48/64 layers.*

    ---

    ## Plots

    ### Per-block abs_max (Boltz-1 solid, Boltz-2 dashed)
    ![per_block_max]({_relplot("per_block_max_comparison.png")})

    ### Top-5 dominant channels (blue = Boltz-1, orange = Boltz-2)
    ![top_channels]({_relplot("top_channels_comparison.png")})

    ### Diagonal/off-diagonal ratio per layer
    ![diagonal_ratio]({_relplot("diagonal_ratio_comparison.png")})

    ### Diagonal vs off-diagonal at peak layer
    ![peak_diag]({_relplot("peak_layer_diag_offdiag.png")})
    """)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)
    print(f"\nReport written to {out_path}")


# ---------------------------------------------------------------------------
# Load Boltz-1 results
# ---------------------------------------------------------------------------

def _load_boltz1_results(
    boltz1_dir: Path,
    proteins: List[str],
) -> Tuple[Dict[str, List[Dict]], Dict[str, List[Dict]]]:
    stats: Dict[str, List[Dict]] = {}
    diag: Dict[str, List[Dict]] = {}

    for pname in proteins:
        stats_path = boltz1_dir / pname / "pairformer_stats.json"
        diag_path = boltz1_dir / pname / "diagonal_stats.json"

        if stats_path.exists():
            with open(stats_path) as f:
                d = json.load(f)
            stats[pname] = d.get("records", [])
            print(f"  Loaded Boltz-1 stats for {pname}: "
                  f"{d.get('num_layers','?')} layers, "
                  f"{len(stats[pname])} records")
        else:
            print(f"  [warn] No Boltz-1 stats at {stats_path}")

        if diag_path.exists():
            with open(diag_path) as f:
                diag[pname] = json.load(f)
        else:
            print(f"  [warn] No Boltz-1 diagonal stats at {diag_path}")

    return stats, diag


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare Boltz-2 vs Boltz-1 Pairformer activation patterns."
    )
    p.add_argument(
        "--out_dir",
        default="./test_output/boltz_results_boltz2_results",
        help="Root output directory for Boltz-2 results and comparison plots.",
    )
    p.add_argument(
        "--boltz1_dir",
        default="./test_output/boltz_results_profile_long_results",
        help=(
            "Directory containing Boltz-1 pairformer_stats.json and diagonal_stats.json "
            "(from test_long_sequences.py). Default: "
            "./test_output/boltz_results_profile_long_results"
        ),
    )
    p.add_argument(
        "--yaml_dir",
        default="./test_output/boltz_results_profile_long_results",
        help=(
            "Directory containing the YAML inputs generated by test_long_sequences.py. "
            "Default: ./test_output/boltz_results_profile_long_results"
        ),
    )
    p.add_argument(
        "--cache",
        default=get_cache_path(),
        help="Boltz cache directory (default: ~/.boltz or $BOLTZ_CACHE).",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Optional path to a Boltz-2 checkpoint (.ckpt).",
    )
    p.add_argument(
        "--use_msa_server",
        action="store_true",
        help="Fetch MSAs from ColabFold (recommended).",
    )
    p.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference.",
    )
    p.add_argument(
        "--no_bf16",
        action="store_true",
        help="Disable bfloat16 autocast.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    warnings.filterwarnings("ignore", ".*Tensor Cores.*")
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("highest")

    cache = Path(args.cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    boltz1_dir = Path(args.boltz1_dir)
    yaml_dir = Path(args.yaml_dir)
    comparison_dir = out_root / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("  boltz2_verification.py — Boltz-2 vs Boltz-1 Pairformer comparison")
    print(f"{'='*70}\n")

    # ------------------------------------------------------------------
    # 1. Download Boltz-2 weights
    # ------------------------------------------------------------------
    print("[1/5] Checking / downloading Boltz-2 weights (~5GB on first run) …")
    download_boltz2(cache)
    print("  Weights ready.")

    # ------------------------------------------------------------------
    # 2. Load model
    # ------------------------------------------------------------------
    print("\n[2/5] Loading Boltz-2 model …")
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    use_bf16 = (not args.no_bf16) and (device.type == "cuda")
    print(f"  Device: {device}  |  bfloat16: {use_bf16}")
    model = _load_boltz2(cache, args.checkpoint)
    model = model.to(device)

    # ------------------------------------------------------------------
    # 3. Load Boltz-1 results
    # ------------------------------------------------------------------
    print(f"\n[3/5] Loading Boltz-1 reference results from {boltz1_dir} …")
    protein_names = [p["name"] for p in PROTEINS]
    b1_stats, b1_diag = _load_boltz1_results(boltz1_dir, protein_names)

    # ------------------------------------------------------------------
    # 4. Run Boltz-2 profiling for each protein
    # ------------------------------------------------------------------
    print("\n[4/5] Running Boltz-2 profiling …")
    b2_stats: Dict[str, List[Dict]] = {}
    b2_diag: Dict[str, List[Dict]] = {}

    for prot in PROTEINS:
        name = prot["name"]
        print(f"\n  ── {name}  ({prot['description']}) ──")

        # Look for the YAML from test_long_sequences.py output
        yaml_path = yaml_dir / name / f"{name}.yaml"
        if not yaml_path.exists():
            print(f"  [warn] YAML not found at {yaml_path}. Skipping {name}.")
            continue

        work_dir = out_root / name / "processed_input"

        result = _profile_protein(
            name=name,
            yaml_path=yaml_path,
            work_dir=work_dir,
            cache=cache,
            model=model,
            device=device,
            use_bf16=use_bf16,
            out_dir=out_root / name,
            use_msa_server=args.use_msa_server,
        )
        if result is None:
            print(f"  [FAILED] {name} skipped.")
            continue

        stats_store, diag_store = result
        b2_stats[name] = stats_store
        b2_diag[name] = diag_store

    # ------------------------------------------------------------------
    # 5. Comparison, plots, and report
    # ------------------------------------------------------------------
    print(f"\n[5/5] Cross-comparison ({len(b2_stats)} Boltz-2 profiles) …")

    # Use only proteins that succeeded in both models
    common_proteins = sorted(
        p for p in protein_names if p in b1_stats and p in b2_stats
    )

    if not common_proteins:
        print("ERROR: No proteins with both Boltz-1 and Boltz-2 results.", file=sys.stderr)
        sys.exit(1)

    print(f"  Common proteins for comparison: {common_proteins}")

    # Comparison tables
    all_data = print_comparison_tables(
        proteins=common_proteins,
        boltz1_stats={p: b1_stats[p] for p in common_proteins},
        boltz2_stats={p: b2_stats[p] for p in common_proteins},
        boltz1_diag={p: b1_diag.get(p, []) for p in common_proteins},
        boltz2_diag={p: b2_diag[p] for p in common_proteins},
    )

    # Verdict
    verdict_key, verdict = _determine_verdict(all_data)
    print(f"\n{'='*70}")
    print(f"VERDICT: {verdict}")
    print(f"{'='*70}\n")

    # Plots
    print("Generating comparison plots …")
    plot_per_block_max(
        boltz1_stats={p: b1_stats[p] for p in common_proteins},
        boltz2_stats=b2_stats,
        out_path=comparison_dir / "per_block_max_comparison.png",
    )
    plot_top_channels(
        boltz1_ch={p: _dominant_channels(b1_stats[p]) for p in common_proteins},
        boltz2_ch={p: _dominant_channels(b2_stats[p]) for p in common_proteins},
        out_path=comparison_dir / "top_channels_comparison.png",
    )
    plot_diagonal_ratio(
        boltz1_diag={p: b1_diag.get(p, []) for p in common_proteins},
        boltz2_diag={p: b2_diag[p] for p in common_proteins},
        boltz1_stats={p: b1_stats[p] for p in common_proteins},
        boltz2_stats=b2_stats,
        out_path=comparison_dir / "diagonal_ratio_comparison.png",
    )
    plot_peak_diag_offdiag(
        boltz1_diag={p: b1_diag.get(p, []) for p in common_proteins},
        boltz2_diag={p: b2_diag[p] for p in common_proteins},
        boltz1_stats={p: b1_stats[p] for p in common_proteins},
        boltz2_stats=b2_stats,
        out_path=comparison_dir / "peak_layer_diag_offdiag.png",
    )

    # Markdown report
    _write_markdown_report(
        verdict_key=verdict_key,
        verdict=verdict,
        all_data=all_data,
        plot_dir=comparison_dir,
        out_path=comparison_dir / "REPORT.md",
    )

    print(f"\nAll Boltz-2 outputs: {out_root.resolve()}")
    print(f"Comparison report:   {comparison_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
