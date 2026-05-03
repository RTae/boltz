#!/usr/bin/env python3
"""
test_long_sequences.py – Verify that Pairformer channel-wise outlier patterns
hold across different sequence lengths.

Tests three proteins of increasing size:
  small  (~113 aa) — boltz examples/prot.yaml
  medium (~386 aa) — HSP70 fragment (PDB 3HSC chain A)
  large  (~668 aa) — Glutamate synthase (PDB 3C7N chain A)

For each protein:
  1. Fetches / hard-codes the FASTA sequence and writes a minimal YAML.
  2. Processes the input (Boltz-1 format) via process_inputs.
  3. Runs a single Pairformer trunk forward pass with two hook types:
     a. Standard stats hooks (captures per-channel std, abs_max per layer).
     b. Diagonal-concentration hooks (records diagonal vs off-diagonal abs_max).
  4. Saves pairformer_stats.json and diagonal_stats.json per protein.

Analysis outputs (written to --out_dir, default ./test_output_long_results/):
  long_sequence_per_block_max.png         – per-block abs_max curves overlaid
  long_sequence_diagonal_concentration.png – diag vs off-diag max at peak layer
  Printed channel comparison table and final verdict.

Usage:
    python test_long_sequences.py [--use_msa_server] [--cpu] [--no_bf16]
                                  [--out_dir ./test_output_long_results]
                                  [--cache ~/.boltz]
"""

import importlib.util
import json
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
# Bootstrap: import shared helpers from profile_pairformer.py
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_PP_PATH = _SCRIPT_DIR / "profile_pairformer.py"

if not _PP_PATH.exists():
    print(
        f"ERROR: profile_pairformer.py not found at {_PP_PATH}\n"
        "Please run this script from the boltz repository root.",
        file=sys.stderr,
    )
    sys.exit(1)

_spec = importlib.util.spec_from_file_location("profile_pairformer", _PP_PATH)
_pp = importlib.util.module_from_spec(_spec)   # type: ignore[arg-type]
_spec.loader.exec_module(_pp)                  # type: ignore[union-attr]

# Re-use helpers from profile_pairformer
_tensor_stats = _pp._tensor_stats             # noqa: SLF001
_json_safe = _pp._json_safe                   # noqa: SLF001
run_trunk_forward_orig = _pp.run_trunk_forward
remove_hooks = _pp.remove_hooks

# Boltz internals
from boltz.main import (                       # noqa: E402
    BoltzDiffusionParams,
    BoltzProcessedInput,
    BoltzSteeringParams,
    MSAModuleArgs,
    PairformerArgs,
    check_inputs,
    download_boltz1,
    get_cache_path,
    process_inputs,
)
from boltz.data.module.inference import BoltzInferenceDataModule
from boltz.data.types import Manifest
from boltz.model.models.boltz1 import Boltz1

import argparse

# ---------------------------------------------------------------------------
# Protein catalogue
# ---------------------------------------------------------------------------

# RCSB fallback sequences (pre-fetched so the script runs offline too).
_SMALL_SEQ = (
    "QLEDSEVEAVAKGLEEMYANGVTEDNFKNYVKNNFAQQEISSVEEELNVNISDSCVANKIKDEFFAMISIS"
    "AIVKAAQKKAWKELAVTVLRFAKANGLKTNAIIVAGQLALWAVQCG"
)  # 113 aa – same as examples/prot.yaml

_MEDIUM_SEQ = (
    "MSKGPAVGIDLGTTYSCVGVFQHGKVEIIANDQGNRTTPSYVAFTDTERLIGDAAKNQVAMNPTNTVFDA"
    "KRLIGRRFDDAVVQSDMKHWPFMVVNDAGRPKVQVEYKGETKSFYPEEVSSMVLTKMKEIAEAYLGKTVTN"
    "AVVTVPAYFNDSQRQATKDAGTIAGLNVLRIINEPTAAAIAYGLDKKVGAERNVLIFDLGGGTFDVSILTIED"
    "GIFEVKSTAGDTHLGGEDFDNRMVNHFIAEFKRKHKKDISENKRAVRRLRTACERAKRTLSSSTQASIEIDS"
    "LYEGIDFYTSITRARFEELNADLFRGTLDPVEKALRDAKLDKSQIHDIVLVGGSTRIPKIQKLLQDFFNGKE"
    "LNKSINPDEAVAYGAAVQAAILSGDKSE"
)  # 386 aa – HSP70 (PDB 3HSC chain A)

_LARGE_SEQ = (
    "GPMSTPFGLDLGNNNSVLAVARNRGIDIVVNEVSNRSTPSVVGFGPKNRYLGETGKNKQTSNIKNTVANLK"
    "RIIGLDYHHPDFEQESKHFTSKLVELDDKKTGAEVRFAGEKHVFSATQLAAMFIDKVKDTVKQDTKANITDV"
    "CIAVPPWYTEEQRYNIADAARIAGLNPVRIVNDVTAAGVSYGIFKTDLPEGEEKPRIVAFVDIGHSSYTCSIM"
    "AFKKGQLKVLGTACDKHFGGRDFDLAITEHFADEFKTKYKIDIRENPKAYNRILTAAEKLKKVLSANTNAPF"
    "SVESVMNDVDVSSQLSREELEELVKPLLERVTEPVTKALAQAKLSAEEVDFVEIIGGTTRIPTLKQSISEAFG"
    "KPLSTTLNQDEAIAKGAAFICAIHSPTLRVRPFKFEDIHPYSVSYSWDKQVEDEDHMEVFPAGSSFPSTKLI"
    "TLNRTGDFSMAASYTDITQLPPNTPEQIANWEITGVQLPEGQDSVPVKLKLRCDPSGLHTIEEAYTIEDIEVE"
    "EPIPLPEDAPEDAEQEFKKVTKTVKKDDLTIVAHTFGLDAKKLNELIEKENEMLAQDKLVAETEDRKNTLEEY"
    "IYTLRGKLEEEYAPFASDAEKTKLQGMLNKAEEWLYDEGFDSIKAKYIAKYEELASLGNIIRGRYLAKEEEKKQ"
    "AIRSKQEASQMAAMA"
)  # 668 aa – Glutamate synthase subunit (PDB 3C7N chain A)

PROTEINS: List[Dict[str, Any]] = [
    {
        "name": "small_113aa",
        "pdb_id": None,          # use existing prot.yaml
        "yaml_src": "examples/prot.yaml",
        "fallback_seq": _SMALL_SEQ,
        "needs_msa_server": True,
        "description": "~113 aa (examples/prot.yaml)",
    },
    {
        "name": "medium_386aa",
        "pdb_id": "3HSC",
        "pdb_entity": "1",
        "fallback_seq": _MEDIUM_SEQ,
        "needs_msa_server": True,
        "description": "~386 aa (PDB 3HSC, HSP70 chaperone)",
    },
    {
        "name": "large_668aa",
        "pdb_id": "3C7N",
        "pdb_entity": "1",
        "fallback_seq": _LARGE_SEQ,
        "needs_msa_server": True,
        "description": "~668 aa (PDB 3C7N, glutamate synthase)",
    },
]

# ---------------------------------------------------------------------------
# Sequence fetching
# ---------------------------------------------------------------------------

def _fetch_rcsb_seq(pdb_id: str, entity: str = "1") -> Optional[str]:
    """Try to fetch the canonical one-letter sequence from RCSB."""
    try:
        import urllib.request
        url = (
            f"https://data.rcsb.org/rest/v1/core/polymer_entity"
            f"/{pdb_id.upper()}/{entity}"
        )
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read())
        seq: str = data.get("entity_poly", {}).get(
            "pdbx_seq_one_letter_code_can", ""
        )
        return seq or None
    except Exception as exc:
        print(f"  [warn] Could not fetch {pdb_id} entity {entity} from RCSB: {exc}")
        return None


def _resolve_sequence(p: Dict[str, Any], repo_root: Path) -> Tuple[str, str]:
    """Return (sequence, source_description)."""
    # Small protein: read from existing YAML
    if p.get("yaml_src"):
        yaml_path = repo_root / p["yaml_src"]
        if yaml_path.exists():
            import re
            text = yaml_path.read_text()
            m = re.search(r"sequence:\s*(\S+)", text)
            if m:
                return m.group(1), f"from {p['yaml_src']}"
    # Fetch from RCSB
    if p.get("pdb_id"):
        seq = _fetch_rcsb_seq(p["pdb_id"], p.get("pdb_entity", "1"))
        if seq:
            return seq, f"RCSB {p['pdb_id']} entity {p.get('pdb_entity','1')}"
    # Fall back to hardcoded sequence
    return p["fallback_seq"], "hardcoded fallback"


# ---------------------------------------------------------------------------
# YAML generation
# ---------------------------------------------------------------------------

_YAML_TEMPLATE = textwrap.dedent("""\
    version: 1
    sequences:
      - protein:
          id: A
          sequence: {seq}
""")


def _write_yaml(yaml_path: Path, seq: str) -> None:
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(_YAML_TEMPLATE.format(seq=seq))


# ---------------------------------------------------------------------------
# Input processing (Boltz-1 format)
# ---------------------------------------------------------------------------

def _process_input(
    yaml_path: Path,
    work_dir: Path,
    cache: Path,
    use_msa_server: bool,
) -> Optional[Path]:
    """Run process_inputs and return the processed/ subdirectory, or None on failure."""
    processed_dir = work_dir / "processed"

    if (processed_dir / "manifest.json").exists():
        print(f"    [cache] Reusing existing processed data at {processed_dir}")
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
        boltz2=False,
        preprocessing_threads=min(4, multiprocessing.cpu_count()),
    )

    # process_inputs may return None but still write the manifest
    if not manifest or not manifest.records:
        mp = processed_dir / "manifest.json"
        if mp.exists():
            manifest = Manifest.load(mp)

    if not manifest or not manifest.records:
        print(
            f"    ERROR: manifest is empty for {yaml_path.name}. "
            "If this protein needs MSA, pass --use_msa_server.",
            file=sys.stderr,
        )
        return None

    return processed_dir


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model(cache: Path, checkpoint: Optional[str]) -> Boltz1:
    ckpt = checkpoint or str(cache / "boltz1_conf.ckpt")
    print(f"  Loading checkpoint: {ckpt}")
    model: Boltz1 = Boltz1.load_from_checkpoint(
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
        diffusion_process_args=asdict(BoltzDiffusionParams()),
        ema=False,
        use_kernels=False,
        pairformer_args=asdict(
            PairformerArgs(num_blocks=48, num_heads=16, dropout=0.0,
                           activation_checkpointing=False)
        ),
        msa_args=asdict(MSAModuleArgs(subsample_msa=False, use_paired_feature=False)),
        steering_args=asdict(BoltzSteeringParams()),
    )
    model.eval()
    model.use_kernels = False
    return model


# ---------------------------------------------------------------------------
# Hook infrastructure — standard stats
# ---------------------------------------------------------------------------

def _make_stats_hook(layer_idx: int, storage: List[Dict]) -> Any:
    """Record per-channel std and abs_max for layer_output_z and layer_output_s."""
    def hook(module: nn.Module, inputs: Any, output: Any) -> None:
        if not isinstance(output, (tuple, list)) or len(output) < 2:
            return
        s_out, z_out = output[0], output[1]
        for tag, t in (("layer_output_z", z_out), ("layer_output_s", s_out)):
            if not isinstance(t, torch.Tensor):
                continue
            try:
                stats = _tensor_stats(t)
            except Exception as exc:
                stats = {"error": str(exc)}
            storage.append({"layer_idx": layer_idx, "op": tag, **stats})
    return hook


# ---------------------------------------------------------------------------
# Hook infrastructure — diagonal concentration
# ---------------------------------------------------------------------------

def _make_diag_hook(layer_idx: int, diag_storage: List[Dict]) -> Any:
    """Record diagonal vs off-diagonal abs_max of the z pair tensor."""
    def hook(module: nn.Module, inputs: Any, output: Any) -> None:
        if not isinstance(output, (tuple, list)) or len(output) < 2:
            return
        z_out = output[1]
        if not isinstance(z_out, torch.Tensor) or z_out.ndim < 4:
            return

        # z_out: [B, N, N, D] — work in float32 to avoid bfloat16 edge cases
        z = z_out.detach().float()
        B, N, N2, D = z.shape

        # Build diagonal index mask
        diag_idx = torch.arange(N, device=z.device)

        # Diagonal positions: z[:, i, i, :]
        diag_vals = z[:, diag_idx, diag_idx, :]        # [B, N, D]
        diag_abs_max = float(diag_vals.abs().max())

        # Off-diagonal: use a triu/tril mask trick — zero diagonal then take max
        # Create a mask for off-diagonal (N x N, True = off-diagonal)
        eye_mask = torch.eye(N, dtype=torch.bool, device=z.device)  # [N, N]
        # Expand to [1, N, N, 1] and broadcast
        off_mask = (~eye_mask).unsqueeze(0).unsqueeze(-1)  # [1, N, N, 1]
        offdiag_abs_max_val = float(z.abs().masked_fill(~off_mask, 0.0).max())

        diag_storage.append({
            "layer_idx": layer_idx,
            "diag_abs_max": diag_abs_max,
            "offdiag_abs_max": offdiag_abs_max_val,
            "N": N,
        })

    return hook


def _attach_all_hooks(
    pairformer_module: nn.Module,
) -> Tuple[List, List[Dict], List[Dict]]:
    """Attach both stats hooks and diagonal hooks to all PairformerLayer instances.

    Returns
    -------
    handles     : list of hook handles (call .remove() on each)
    stats_store : list that stats hooks will fill
    diag_store  : list that diagonal hooks will fill
    """
    stats_store: List[Dict] = []
    diag_store: List[Dict] = []
    handles = []

    layers = list(pairformer_module.layers)
    for layer_idx, layer in enumerate(layers):
        h = layer.register_forward_hook(
            _make_stats_hook(layer_idx, stats_store)
        )
        handles.append(h)
        h2 = layer.register_forward_hook(
            _make_diag_hook(layer_idx, diag_store)
        )
        handles.append(h2)

    return handles, stats_store, diag_store


# ---------------------------------------------------------------------------
# Trunk forward pass  (same logic as profile_pairformer.run_trunk_forward)
# ---------------------------------------------------------------------------

def _run_trunk(
    model: Boltz1,
    feats: Dict[str, torch.Tensor],
    device: torch.device,
    use_bf16: bool,
) -> None:
    ctx_mgr = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if use_bf16 and device.type == "cuda"
        else torch.no_grad()
    )
    with torch.no_grad():
        with ctx_mgr:
            s_inputs = model.input_embedder(feats)
            s_init = model.s_init(s_inputs)
            z_init = (
                model.z_init_1(s_inputs)[:, :, None]
                + model.z_init_2(s_inputs)[:, None, :]
            )
            z_init = z_init + model.rel_pos(feats)
            z_init = z_init + model.token_bonds(feats["token_bonds"].float())

            s = torch.zeros_like(s_init)
            z = torch.zeros_like(z_init)
            mask = feats["token_pad_mask"].float()
            pair_mask = mask[:, :, None] * mask[:, None, :]

            recycling_steps = 1
            for _i in range(recycling_steps + 1):
                s = s_init + model.s_recycle(model.s_norm(s))
                z = z_init + model.z_recycle(model.z_norm(z))

                if not model.no_msa:
                    z = z + model.msa_module(
                        z, s_inputs, feats, use_kernels=False
                    )

                pairformer_module = (
                    model.pairformer_module._orig_mod   # noqa: SLF001
                    if getattr(model, "is_pairformer_compiled", False)
                    else model.pairformer_module
                )
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

def _to_device(obj: Any, dev: torch.device) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.to(dev)
    if isinstance(obj, dict):
        return {k: _to_device(v, dev) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_device(v, dev) for v in obj]
    return obj


def profile_protein(
    name: str,
    processed_dir: Path,
    model: Boltz1,
    device: torch.device,
    use_bf16: bool,
    out_dir: Path,
) -> Optional[Tuple[List[Dict], List[Dict]]]:
    """Run trunk forward with both hook types. Returns (stats_store, diag_store)."""
    manifest_path = processed_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"    ERROR: manifest.json missing in {processed_dir}", file=sys.stderr)
        return None

    manifest = Manifest.load(manifest_path)
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

    dm = BoltzInferenceDataModule(
        manifest=processed.manifest,
        target_dir=processed.targets_dir,
        msa_dir=processed.msa_dir,
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
    print(f"    Tokens: {token_count}  |  pair tensor ~[1,{token_count},{token_count},D]")

    # Locate pairformer
    pairformer_module = (
        model.pairformer_module._orig_mod            # noqa: SLF001
        if getattr(model, "is_pairformer_compiled", False)
        else model.pairformer_module
    )

    print(f"    Attaching hooks to {len(list(pairformer_module.layers))} layers …")
    handles, stats_store, diag_store = _attach_all_hooks(pairformer_module)

    # --- Forward pass (with OOM guard) ---
    try:
        t0 = time.time()
        _run_trunk(model, feats, device, use_bf16)
        elapsed = time.time() - t0
        print(f"    Forward pass took {elapsed:.1f}s")
    except torch.cuda.OutOfMemoryError:
        remove_hooks(handles)
        torch.cuda.empty_cache()
        print(
            f"    [OOM] CUDA out-of-memory for {name} (N={token_count}). "
            "Skipping.",
            file=sys.stderr,
        )
        return None
    except Exception as exc:
        remove_hooks(handles)
        print(f"    ERROR during forward pass: {exc}", file=sys.stderr)
        return None
    finally:
        remove_hooks(handles)

    print(f"    Collected {len(stats_store)} stats records, {len(diag_store)} diag records.")

    # Save stats JSON
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = out_dir / "pairformer_stats.json"
    with open(stats_path, "w") as f:
        json.dump(
            _json_safe({
                "protein": name,
                "token_count": token_count,
                "num_layers": len([r for r in stats_store if r.get("op") == "layer_output_z"]),
                "device": str(device),
                "bf16_autocast": use_bf16,
                "records": stats_store,
            }),
            f, indent=2,
        )

    diag_path = out_dir / "diagonal_stats.json"
    with open(diag_path, "w") as f:
        json.dump(_json_safe(diag_store), f, indent=2)

    print(f"    Saved {stats_path} and {diag_path}")
    return stats_store, diag_store


# ---------------------------------------------------------------------------
# Channel dominance analysis
# ---------------------------------------------------------------------------

TOP_K_PER_LAYER = 5
TOP_N_REPORT = 5


def _dominant_channels(
    stats_store: List[Dict],
    top_k_per_layer: int = TOP_K_PER_LAYER,
    report_n: int = TOP_N_REPORT,
) -> List[Tuple[int, float]]:
    """Return top-N (channel_idx, hit_pct) across all layers.

    Two methods are combined:
    1. Per-channel std per layer (per_channel_std field) — selects top-k channels
       by variance each layer and counts how often each channel appears.
    2. Flat-index decoding (top10_abs_flat_indices % shape[-1]) — counts raw
       absolute-value hits.
    Both vote equally; the union is sorted by combined hit count.
    """
    layer_z_recs = [r for r in stats_store if r.get("op") == "layer_output_z"]
    if not layer_z_recs:
        return []

    num_layers = len(layer_z_recs)
    std_hits: Counter = Counter()
    flat_hits: Counter = Counter()

    for rec in layer_z_recs:
        # Method 1: per-channel std
        ch_std = rec.get("per_channel_std")
        if ch_std:
            ranked = sorted(range(len(ch_std)), key=lambda i: ch_std[i], reverse=True)
            for ch in ranked[:top_k_per_layer]:
                std_hits[ch] += 1

        # Method 2: flat-index decoding
        shape = rec.get("shape")
        flat_idxs = rec.get("top10_abs_flat_indices")
        if shape and flat_idxs:
            D = shape[-1]
            for fi in flat_idxs[:top_k_per_layer]:
                ch = fi % D
                flat_hits[ch] += 1

    # Combine: total_hits = std_hits + flat_hits (both over num_layers)
    all_channels = set(std_hits.keys()) | set(flat_hits.keys())
    combined: Dict[int, float] = {}
    for ch in all_channels:
        pct_std = std_hits.get(ch, 0) / num_layers * 100.0
        pct_flat = flat_hits.get(ch, 0) / num_layers * 100.0
        combined[ch] = (pct_std + pct_flat) / 2.0

    top = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:report_n]
    return top


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_per_block_max(
    all_stats: Dict[str, List[Dict]],
    out_path: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available – skipping per-block-max plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    for name, storage in all_stats.items():
        layer_z = {
            r["layer_idx"]: r
            for r in storage
            if r.get("op") == "layer_output_z"
        }
        if not layer_z:
            continue
        layers = sorted(layer_z.keys())
        abs_maxes = [layer_z[i]["abs_max"] for i in layers]
        ax.plot(layers, abs_maxes, marker="o", linewidth=1.5, markersize=3, label=name)

    ax.set_xlabel("Pairformer layer index")
    ax.set_ylabel("abs_max of pair tensor z")
    ax.set_title("Per-block abs_max of pair tensor z across sequence lengths")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def _plot_diagonal_concentration(
    all_diag: Dict[str, List[Dict]],
    all_stats: Dict[str, List[Dict]],
    out_path: Path,
) -> None:
    """Bar chart: diagonal vs off-diagonal abs_max at the peak layer, per protein."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available – skipping diagonal concentration plot.")
        return

    # Determine peak layer per protein from stats (layer with max abs_max of z)
    proteins = [p for p in all_diag if p in all_stats]
    if not proteins:
        return

    x = np.arange(len(proteins))
    diag_vals = []
    offdiag_vals = []
    peak_layers = []

    for pname in proteins:
        # Find peak layer
        layer_z = {
            r["layer_idx"]: r
            for r in all_stats[pname]
            if r.get("op") == "layer_output_z"
        }
        if not layer_z:
            diag_vals.append(0.0)
            offdiag_vals.append(0.0)
            peak_layers.append(-1)
            continue

        peak_layer = max(layer_z, key=lambda i: layer_z[i]["abs_max"])
        peak_layers.append(peak_layer)

        # Find corresponding diagonal record
        diag_rec = next(
            (d for d in all_diag[pname] if d["layer_idx"] == peak_layer), None
        )
        if diag_rec:
            diag_vals.append(diag_rec["diag_abs_max"])
            offdiag_vals.append(diag_rec["offdiag_abs_max"])
        else:
            diag_vals.append(0.0)
            offdiag_vals.append(0.0)

    bar_w = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - bar_w / 2, diag_vals, bar_w, label="Diagonal (i=j)", color="steelblue")
    bars2 = ax.bar(x + bar_w / 2, offdiag_vals, bar_w, label="Off-diagonal (i≠j)", color="coral")
    ax.bar_label(bars1, fmt="%.3g", padding=3, fontsize=8)
    ax.bar_label(bars2, fmt="%.3g", padding=3, fontsize=8)

    labels = [
        f"{p}\n(peak L{pl})" for p, pl in zip(proteins, peak_layers)
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("abs_max")
    ax.set_title("Diagonal vs Off-diagonal abs_max in pair tensor z at peak layer")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Comparison table + verdict
# ---------------------------------------------------------------------------

def _print_comparison_table(
    channel_results: Dict[str, List[Tuple[int, float]]],
) -> None:
    proteins = list(channel_results.keys())
    print("\n" + "=" * 80)
    print("TOP-5 DOMINANT CHANNELS IN pair tensor z BY SEQUENCE LENGTH")
    print("=" * 80)

    col_w = 18
    header = f"{'Protein':<22}" + "".join(
        f"{'  top-'+str(i+1):<{col_w}}" for i in range(TOP_N_REPORT)
    )
    print(header)
    print("-" * 80)

    for pname in proteins:
        tops = channel_results.get(pname, [])
        cells = []
        for i in range(TOP_N_REPORT):
            if i < len(tops):
                ch, pct = tops[i]
                cells.append(f"ch{ch}({pct:.0f}%)")
            else:
                cells.append("—")
        row = f"  {pname:<20}" + "".join(f"  {c:<{col_w-2}}" for c in cells)
        print(row)
    print("=" * 80)


def _print_verdict(channel_results: Dict[str, List[Tuple[int, float]]]) -> None:
    if not channel_results:
        return

    # Collect top-1 and top-3 channels per protein
    top1 = {p: tops[0][0] if tops else -1 for p, tops in channel_results.items()}
    top3 = {
        p: {tops[i][0] for i in range(min(3, len(tops)))}
        for p, tops in channel_results.items()
    }

    proteins = [p for p in channel_results if top1[p] >= 0]
    if not proteins:
        print("\nVERDICT: insufficient data.")
        return

    top1_vals = [top1[p] for p in proteins]
    unique_top1 = set(top1_vals)

    # Check if top-3 sets overlap significantly
    top3_sets = [top3[p] for p in proteins]
    intersection_top3 = set.intersection(*top3_sets) if top3_sets else set()

    print("\n" + "=" * 70)
    if len(unique_top1) == 1:
        ch = list(unique_top1)[0]
        print(
            f"PATTERN HOLDS: same dominant channel (ch{ch}) at all N tested."
        )
        if len(intersection_top3) >= 2:
            others = sorted(intersection_top3 - {ch})
            print(f"  Common top-3: ch{ch} + {', '.join(f'ch{c}' for c in others)}")
    elif intersection_top3:
        common = sorted(intersection_top3)
        dominant_ch = Counter(top1_vals).most_common(1)[0][0]
        print(
            f"PATTERN PARTIALLY HOLDS: channel(s) {common} appear in top-3 at all N, "
            f"but top-1 varies."
        )
        # List which N deviate
        for p in proteins:
            if top1[p] != dominant_ch:
                print(f"  {p}: top channel = ch{top1[p]}")
    else:
        print("PATTERN BREAKS: dominant channels differ significantly at large N.")
        for p in proteins:
            tops_str = ", ".join(f"ch{c}" for c, _ in channel_results[p][:3])
            print(f"  {p}: {tops_str}")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Verify Pairformer channel outlier pattern at longer sequence lengths."
    )
    p.add_argument(
        "--out_dir",
        default="./test_output/boltz_results_profile_long_results",
        help="Output directory (default: ./test_output/boltz_results_profile_long_results).",
    )
    p.add_argument(
        "--cache",
        default=get_cache_path(),
        help="Boltz cache directory.",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Optional Boltz-1 checkpoint path.",
    )
    p.add_argument(
        "--use_msa_server",
        action="store_true",
        help="Fetch MSAs from ColabFold server (recommended for accuracy).",
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
    repo_root = _SCRIPT_DIR

    print(f"\n{'='*70}")
    print("  test_long_sequences.py — Pairformer channel pattern vs sequence length")
    print(f"{'='*70}\n")

    # ------------------------------------------------------------------
    # 1. Download weights
    # ------------------------------------------------------------------
    print("[1/5] Checking / downloading Boltz-1 weights …")
    download_boltz1(cache)

    # ------------------------------------------------------------------
    # 2. Load model (once, shared across all proteins)
    # ------------------------------------------------------------------
    print("\n[2/5] Loading Boltz-1 model …")
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    use_bf16 = (not args.no_bf16) and (device.type == "cuda")
    print(f"  Device: {device}  |  bfloat16: {use_bf16}")
    model = _load_model(cache, args.checkpoint)
    model = model.to(device)

    # ------------------------------------------------------------------
    # 3. Prepare inputs
    # ------------------------------------------------------------------
    print("\n[3/5] Preparing YAML inputs …")
    targets = []
    for prot in PROTEINS:
        print(f"  {prot['name']}: {prot['description']}")
        seq, src = _resolve_sequence(prot, repo_root)
        # Compact sequence (remove any whitespace from multi-line hardcodes)
        seq = seq.replace("\n", "").replace(" ", "")
        print(f"    sequence length = {len(seq)} aa  (source: {src})")

        work_dir = out_root / prot["name"] / "processed_input"
        yaml_path = out_root / prot["name"] / f"{prot['name']}.yaml"

        if not yaml_path.exists():
            _write_yaml(yaml_path, seq)
            print(f"    Wrote {yaml_path}")
        else:
            print(f"    YAML exists: {yaml_path}")

        targets.append({
            **prot,
            "yaml_path": yaml_path,
            "work_dir": work_dir,
            "seq_len": len(seq),
        })

    # ------------------------------------------------------------------
    # 4. Process + profile each protein
    # ------------------------------------------------------------------
    print("\n[4/5] Processing and profiling …")
    all_stats: Dict[str, List[Dict]] = {}
    all_diag: Dict[str, List[Dict]] = {}

    for t in targets:
        name = t["name"]
        print(f"\n  ── {name}  ({t['description']}) ──")

        # Process input
        print("    Processing input (may fetch MSA) …")
        t0 = time.time()
        processed_dir = _process_input(
            yaml_path=t["yaml_path"],
            work_dir=t["work_dir"],
            cache=cache,
            use_msa_server=t["needs_msa_server"] and args.use_msa_server,
        )
        if processed_dir is None:
            print(f"    [FAILED] Skipping {name}.")
            continue
        print(f"    Input processed in {time.time()-t0:.1f}s")

        # Run profiling
        print("    Running trunk forward pass …")
        result = profile_protein(
            name=name,
            processed_dir=processed_dir,
            model=model,
            device=device,
            use_bf16=use_bf16,
            out_dir=out_root / name,
        )
        if result is None:
            print(f"    [FAILED] Skipping {name}.")
            continue

        stats_store, diag_store = result
        all_stats[name] = stats_store
        all_diag[name] = diag_store

    # ------------------------------------------------------------------
    # 5. Analysis and plots
    # ------------------------------------------------------------------
    print(f"\n[5/5] Analysis — {len(all_stats)} protein(s) profiled successfully.\n")

    if not all_stats:
        print("No successful profiles — nothing to compare.", file=sys.stderr)
        sys.exit(1)

    # Channel dominance per protein
    channel_results: Dict[str, List[Tuple[int, float]]] = {}
    for name, storage in all_stats.items():
        tops = _dominant_channels(storage)
        channel_results[name] = tops

    _print_comparison_table(channel_results)

    # Plots
    print("\nGenerating plots …")

    _plot_per_block_max(
        all_stats=all_stats,
        out_path=out_root / "long_sequence_per_block_max.png",
    )

    _plot_diagonal_concentration(
        all_diag=all_diag,
        all_stats=all_stats,
        out_path=out_root / "long_sequence_diagonal_concentration.png",
    )

    # Verdict
    _print_verdict(channel_results)

    print(f"All outputs written to: {out_root.resolve()}")


if __name__ == "__main__":
    main()
