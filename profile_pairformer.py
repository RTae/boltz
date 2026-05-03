#!/usr/bin/env python3
"""
profile_pairformer.py – Empirical study of Pairformer pair-tensor activations.

Usage:
    python profile_pairformer.py [--input examples/prot.yaml] [--cache ~/.boltz]
                                 [--checkpoint path/to/boltz1_conf.ckpt]
                                 [--cpu]

Outputs:
    pairformer_stats.json
    plot1_per_block_max.png
    plot2_channel_std.png
    plot3_per_op_max.png
"""

import argparse
import json
import multiprocessing
import os
import sys
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Import Boltz internals
# ---------------------------------------------------------------------------
from boltz.main import (
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

# ---------------------------------------------------------------------------
# Activation statistics helpers
# ---------------------------------------------------------------------------

TOP_K = 10
HIST_BINS = 50


def _tensor_stats(x: torch.Tensor) -> Dict[str, Any]:
    """Compute lightweight statistics for a tensor (detached, cast to float32)."""
    xf = x.detach().float()

    shape = list(xf.shape)
    dtype = str(x.dtype)
    flat = xf.reshape(-1)

    mn = float(flat.min())
    mx = float(flat.max())
    mean = float(flat.mean())
    std = float(flat.std())
    abs_max = float(flat.abs().max())

    stats: Dict[str, Any] = {
        "shape": shape,
        "dtype": dtype,
        "min": mn,
        "max": mx,
        "mean": mean,
        "std": std,
        "abs_max": abs_max,
    }

    # Per-channel std along last dim for rank >= 3 tensors (z: [B,N,N,D] or s: [B,N,D])
    if xf.ndim >= 3:
        ch_std = xf.reshape(-1, xf.shape[-1]).std(dim=0).cpu().tolist()
        stats["per_channel_std"] = ch_std

        # Top-K absolute values
        abs_flat = xf.abs().reshape(-1)
        topk_vals, topk_idx = torch.topk(abs_flat, min(TOP_K, abs_flat.numel()))
        stats["top10_abs_values"] = topk_vals.cpu().tolist()
        stats["top10_abs_flat_indices"] = topk_idx.cpu().tolist()

    # Histogram (50 bins)
    min_val = float(flat.min())
    max_val = float(flat.max())
    if min_val == max_val:
        max_val = min_val + 1e-6
    hist = torch.histc(flat, bins=HIST_BINS, min=min_val, max=max_val)
    bin_edges = torch.linspace(min_val, max_val, HIST_BINS + 1).tolist()
    stats["histogram"] = {
        "bin_edges": bin_edges,
        "counts": hist.cpu().tolist(),
    }

    return stats


# ---------------------------------------------------------------------------
# Hook infrastructure
# ---------------------------------------------------------------------------

SUBMODULE_NAMES = [
    "tri_mul_out",
    "tri_mul_in",
    "tri_att_start",
    "tri_att_end",
    "transition_z",
    "attention",
    "transition_s",
]


def make_submodule_hook(layer_idx: int, op_name: str, storage: List[Dict]):
    """Return a forward hook that records activation stats for a submodule output."""

    def hook(module: nn.Module, inputs: Any, output: Any) -> None:
        # Some submodules return tuples – take the first element
        if isinstance(output, (tuple, list)):
            tensor = output[0]
        else:
            tensor = output

        if not isinstance(tensor, torch.Tensor):
            return

        try:
            stats = _tensor_stats(tensor)
        except Exception as exc:  # noqa: BLE001
            stats = {"error": str(exc)}

        storage.append({
            "layer_idx": layer_idx,
            "op": op_name,
            **stats,
        })

    return hook


def make_layer_hook(layer_idx: int, storage: List[Dict]):
    """Return a forward hook that records stats for the full layer output (s, z)."""

    def hook(module: nn.Module, inputs: Any, output: Any) -> None:
        # PairformerLayer returns (s, z)
        if not isinstance(output, (tuple, list)) or len(output) < 2:
            return
        s_out, z_out = output[0], output[1]

        s_stats = _tensor_stats(s_out) if isinstance(s_out, torch.Tensor) else {}
        z_stats = _tensor_stats(z_out) if isinstance(z_out, torch.Tensor) else {}

        storage.append({
            "layer_idx": layer_idx,
            "op": "layer_output_z",
            **z_stats,
        })
        storage.append({
            "layer_idx": layer_idx,
            "op": "layer_output_s",
            **s_stats,
        })

    return hook


def attach_hooks(pairformer_module: nn.Module):
    """Attach forward hooks to all PairformerLayer instances and their submodules.

    Returns
    -------
    handles : list
        List of hook handles (call .remove() on each to clean up).
    storage : list
        Mutable list that hooks will append stats dicts into.
    """
    storage: List[Dict] = []
    handles = []

    layers = list(pairformer_module.layers)  # nn.ModuleList

    for layer_idx, layer in enumerate(layers):
        # Hook the layer itself (to capture final (s, z) outputs)
        h = layer.register_forward_hook(make_layer_hook(layer_idx, storage))
        handles.append(h)

        # Hook each named submodule that exists on this layer
        for op_name in SUBMODULE_NAMES:
            submod = getattr(layer, op_name, None)
            if submod is not None:
                h = submod.register_forward_hook(
                    make_submodule_hook(layer_idx, op_name, storage)
                )
                handles.append(h)

    return handles, storage


def remove_hooks(handles):
    for h in handles:
        h.remove()


# ---------------------------------------------------------------------------
# Trunk-only forward pass (avoids running expensive diffusion)
# ---------------------------------------------------------------------------

def run_trunk_forward(model: Boltz1, feats: Dict[str, torch.Tensor],
                      device: torch.device, use_bf16: bool) -> None:
    """Run the embedding + pairformer trunk forward pass on *feats*.

    Parameters
    ----------
    model : Boltz1
    feats : dict of tensors already on *device*
    device : torch.device
    use_bf16 : bool
        Whether to wrap the pass in bfloat16 autocast.
    """
    # Match the recycling logic used during inference (single recycling step for profiling)
    recycling_steps = 1

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
            relative_position_encoding = model.rel_pos(feats)
            z_init = z_init + relative_position_encoding
            z_init = z_init + model.token_bonds(feats["token_bonds"].float())

            s = torch.zeros_like(s_init)
            z = torch.zeros_like(z_init)

            mask = feats["token_pad_mask"].float()
            pair_mask = mask[:, :, None] * mask[:, None, :]

            for _i in range(recycling_steps + 1):
                s = s_init + model.s_recycle(model.s_norm(s))
                z = z_init + model.z_recycle(model.z_norm(z))

                if not model.no_msa:
                    z = z + model.msa_module(
                        z, s_inputs, feats, use_kernels=False
                    )

                # Use uncompiled version if compiled
                if getattr(model, "is_pairformer_compiled", False):
                    pairformer_module = model.pairformer_module._orig_mod  # noqa: SLF001
                else:
                    pairformer_module = model.pairformer_module

                s, z = pairformer_module(
                    s,
                    z,
                    mask=mask,
                    pair_mask=pair_mask,
                    use_kernels=False,
                )

    print(f"Trunk forward complete. z shape: {z.shape}, s shape: {s.shape}")


# ---------------------------------------------------------------------------
# JSON serialisation helper (torch types are not JSON-serialisable by default)
# ---------------------------------------------------------------------------

def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        # Handle inf / nan
        if obj != obj:  # nan
            return "NaN"
        if obj == float("inf"):
            return "Inf"
        if obj == float("-inf"):
            return "-Inf"
        return obj
    return obj


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def generate_plots(storage: List[Dict], out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available – skipping plots.")
        return

    # Organise per-layer z output stats
    layer_z_stats = {
        rec["layer_idx"]: rec
        for rec in storage
        if rec.get("op") == "layer_output_z"
    }
    if not layer_z_stats:
        print("No layer_output_z stats found – skipping plots.")
        return

    layer_indices = sorted(layer_z_stats.keys())
    N = len(layer_indices)
    mid_idx = layer_indices[N // 2]

    # ------------------------------------------------------------------
    # Plot 1 – per-block abs_max of z
    # ------------------------------------------------------------------
    abs_maxes = [layer_z_stats[i]["abs_max"] for i in layer_indices]
    peak_layer = layer_indices[int(np.argmax(abs_maxes))]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(layer_indices, abs_maxes, marker="o", linewidth=1.5, markersize=4)
    ax.axvline(peak_layer, color="red", linestyle="--", alpha=0.6,
               label=f"Peak layer {peak_layer}")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("abs_max")
    ax.set_title("Per-block max absolute value of pair tensor z")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot1_path = out_dir / "plot1_per_block_max.png"
    fig.savefig(plot1_path, dpi=150)
    plt.close(fig)
    print(f"Saved {plot1_path}")

    # ------------------------------------------------------------------
    # Plot 2 – per-channel std distribution at middle layer
    # ------------------------------------------------------------------
    mid_stats = layer_z_stats[mid_idx]
    ch_std = mid_stats.get("per_channel_std")
    if ch_std:
        ch_std_sorted = sorted(ch_std, reverse=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.semilogy(range(len(ch_std_sorted)), ch_std_sorted)
        ax.set_xlabel("Channel rank (sorted)")
        ax.set_ylabel("std (log scale)")
        ax.set_title(f"Channel-wise std distribution at layer {mid_idx}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plot2_path = out_dir / "plot2_channel_std.png"
        fig.savefig(plot2_path, dpi=150)
        plt.close(fig)
        print(f"Saved {plot2_path}")
    else:
        print("No per_channel_std available for middle layer – skipping plot 2.")

    # ------------------------------------------------------------------
    # Plot 3 – per-operation abs_max at middle layer
    # ------------------------------------------------------------------
    op_stats_mid = {
        rec["op"]: rec
        for rec in storage
        if rec.get("layer_idx") == mid_idx and rec.get("op") in SUBMODULE_NAMES
    }
    if op_stats_mid:
        ops = [op for op in SUBMODULE_NAMES if op in op_stats_mid]
        vals = [op_stats_mid[op]["abs_max"] for op in ops]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(ops, vals, color="steelblue")
        ax.bar_label(bars, fmt="%.3g", padding=3, fontsize=8)
        ax.set_xlabel("Operation")
        ax.set_ylabel("abs_max")
        ax.set_title(f"Per-operation max absolute value at layer {mid_idx}")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        plot3_path = out_dir / "plot3_per_op_max.png"
        fig.savefig(plot3_path, dpi=150)
        plt.close(fig)
        print(f"Saved {plot3_path}")
    else:
        print("No submodule stats for middle layer – skipping plot 3.")


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(storage: List[Dict]) -> None:
    layer_z_stats = {
        rec["layer_idx"]: rec
        for rec in storage
        if rec.get("op") == "layer_output_z"
    }
    n_layers = len(layer_z_stats)
    print(f"\n{'='*60}")
    print(f"Number of Pairformer layers found and hooked: {n_layers}")
    print(f"{'='*60}")

    if not layer_z_stats:
        print("No layer output statistics collected.")
        return

    peak_val = -1.0
    peak_layer = -1

    for idx in sorted(layer_z_stats.keys()):
        abs_max = layer_z_stats[idx]["abs_max"]
        z_shape = layer_z_stats[idx].get("shape", "?")
        z_dtype = layer_z_stats[idx].get("dtype", "?")
        print(f"  Layer {idx:3d}: abs_max={abs_max:.4f}  shape={z_shape}  dtype={z_dtype}")
        if abs_max > peak_val:
            peak_val = abs_max
            peak_layer = idx

    print(f"\n  >> Peak abs_max: {peak_val:.4f} at layer {peak_layer} <<")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile Boltz-1 Pairformer activations."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="examples/prot.yaml",
        help="Path to the YAML input file (default: examples/prot.yaml).",
    )
    parser.add_argument(
        "--processed_dir",
        type=str,
        default=None,
        help=(
            "Path to an already-processed boltz predict output directory "
            "(e.g. test_output/boltz_results_prot/processed). "
            "When set, skips input processing entirely."
        ),
    )
    parser.add_argument(
        "--use_msa_server",
        action="store_true",
        help="Use the MMSeqs2 server for MSA generation (same as boltz predict --use_msa_server).",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default=get_cache_path(),
        help="Boltz cache directory (default: ~/.boltz or $BOLTZ_CACHE).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional path to a Boltz-1 checkpoint (.ckpt). "
             "If not provided, uses <cache>/boltz1_conf.ckpt.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./test_output/boltz_results_profile_output",
        help="Directory where JSON stats and plots are saved.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference (slow but works without a GPU).",
    )
    parser.add_argument(
        "--no_bf16",
        action="store_true",
        help="Disable bfloat16 autocast (run in float32).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    warnings.filterwarnings("ignore", ".*Tensor Cores.*")
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("highest")

    cache = Path(args.cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Download model weights if needed
    # ------------------------------------------------------------------
    print("Checking / downloading Boltz-1 model weights …")
    download_boltz1(cache)

    ccd_path = cache / "ccd.pkl"
    mol_dir = cache / "mols"

    # ------------------------------------------------------------------
    # 2. Process input (or reuse existing processed output)
    # ------------------------------------------------------------------
    if args.processed_dir:
        # Fast path: reuse an already-processed boltz predict directory.
        processed_dir = Path(args.processed_dir).expanduser()
        manifest_path = processed_dir / "manifest.json"
        if not manifest_path.exists():
            print(f"ERROR: No manifest.json found in {processed_dir}", file=sys.stderr)
            sys.exit(1)
        manifest = Manifest.load(manifest_path)
        print(f"Reusing pre-processed data from {processed_dir} "
              f"({len(manifest.records)} record(s)).")
    else:
        data_path = Path(args.input).expanduser()
        if not data_path.exists():
            print(f"ERROR: Input file not found: {data_path}", file=sys.stderr)
            sys.exit(1)

        work_dir = out_dir / "processed_input"
        work_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing input: {data_path} …")
        data_list = check_inputs(data_path)

        manifest = process_inputs(
            data=data_list,
            out_dir=work_dir,
            ccd_path=ccd_path,
            mol_dir=mol_dir,
            msa_server_url="https://api.colabfold.com",
            msa_pairing_strategy="greedy",
            use_msa_server=args.use_msa_server,
            boltz2=False,
            preprocessing_threads=min(4, multiprocessing.cpu_count()),
        )

        if not manifest or not manifest.records:
            manifest_path = work_dir / "processed" / "manifest.json"
            if manifest_path.exists():
                manifest = Manifest.load(manifest_path)

        if not manifest or not manifest.records:
            print(
                "ERROR: Manifest has no records. "
                "If your input needs MSA, either:\n"
                "  • pass --use_msa_server   (calls the MMSeqs2 server), or\n"
                "  • pass --processed_dir test_output/boltz_results_prot/processed\n"
                "    to reuse a previous 'boltz predict --use_msa_server' run.",
                file=sys.stderr,
            )
            sys.exit(1)

        processed_dir = work_dir / "processed"
        print(f"Manifest loaded with {len(manifest.records)} record(s).")

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

    # ------------------------------------------------------------------
    # 3. Load model
    # ------------------------------------------------------------------
    checkpoint = args.checkpoint or str(cache / "boltz1_conf.ckpt")
    print(f"Loading Boltz-1 checkpoint: {checkpoint} …")

    pairformer_args = PairformerArgs(
        num_blocks=48,
        num_heads=16,
        dropout=0.0,
        activation_checkpointing=False,
    )
    msa_args = MSAModuleArgs(
        subsample_msa=False,
        use_paired_feature=False,
    )
    diffusion_params = BoltzDiffusionParams()
    steering_args = BoltzSteeringParams()

    predict_args = {
        "recycling_steps": 1,
        "sampling_steps": 10,
        "diffusion_samples": 1,
        "max_parallel_samples": 1,
        "write_confidence_summary": False,
        "write_full_pae": False,
        "write_full_pde": False,
    }

    model: Boltz1 = Boltz1.load_from_checkpoint(
        checkpoint,
        strict=True,
        predict_args=predict_args,
        map_location="cpu",
        diffusion_process_args=asdict(diffusion_params),
        ema=False,
        use_kernels=False,           # disable cuEquivariance kernels so hooks work
        pairformer_args=asdict(pairformer_args),
        msa_args=asdict(msa_args),
        steering_args=asdict(steering_args),
    )
    model.eval()
    model.use_kernels = False        # redundant safety measure

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model = model.to(device)
    use_bf16 = (not args.no_bf16) and (device.type == "cuda")
    print(f"Running on device: {device}  |  bfloat16 autocast: {use_bf16}")

    # ------------------------------------------------------------------
    # 4. Get one batch from the data module
    # ------------------------------------------------------------------
    data_module = BoltzInferenceDataModule(
        manifest=processed.manifest,
        target_dir=processed.targets_dir,
        msa_dir=processed.msa_dir,
        num_workers=0,
        constraints_dir=processed.constraints_dir,
    )
    predict_loader = data_module.predict_dataloader()
    batch = next(iter(predict_loader))

    # Move all tensors to device
    def to_device(obj, dev):
        if isinstance(obj, torch.Tensor):
            return obj.to(dev)
        if isinstance(obj, dict):
            return {k: to_device(v, dev) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_device(v, dev) for v in obj]
        return obj

    feats = to_device(batch, device)
    print(f"Batch loaded. Token count: {feats['token_pad_mask'].sum().item():.0f}")

    # ------------------------------------------------------------------
    # 5. Attach hooks
    # ------------------------------------------------------------------
    # Locate the pairformer module (may be compiled)
    if getattr(model, "is_pairformer_compiled", False):
        pairformer_module = model.pairformer_module._orig_mod  # noqa: SLF001
    else:
        pairformer_module = model.pairformer_module

    print(f"Attaching hooks to {len(list(pairformer_module.layers))} PairformerLayer(s) …")
    handles, storage = attach_hooks(pairformer_module)

    # ------------------------------------------------------------------
    # 6. Run trunk forward pass (hooks fire here)
    # ------------------------------------------------------------------
    print("Running trunk forward pass …")
    run_trunk_forward(model, feats, device, use_bf16)

    # ------------------------------------------------------------------
    # 7. Remove hooks
    # ------------------------------------------------------------------
    remove_hooks(handles)
    print(f"Collected {len(storage)} hook records.")

    # ------------------------------------------------------------------
    # 8. Save stats to JSON
    # ------------------------------------------------------------------
    json_path = out_dir / "pairformer_stats.json"
    # Serialise layer index → list of op records for readability
    out_records: Dict[str, Any] = {
        "num_layers": len([r for r in storage if r.get("op") == "layer_output_z"]),
        "device": str(device),
        "bf16_autocast": use_bf16,
        "records": _json_safe(storage),
    }
    with open(json_path, "w") as f:
        json.dump(out_records, f, indent=2)
    print(f"Stats saved to {json_path}")

    # ------------------------------------------------------------------
    # 9. Print summary
    # ------------------------------------------------------------------
    print_summary(storage)

    # ------------------------------------------------------------------
    # 10. Generate plots
    # ------------------------------------------------------------------
    print("Generating plots …")
    generate_plots(storage, out_dir)
    print(f"\nAll outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
