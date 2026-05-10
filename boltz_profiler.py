#!/usr/bin/env python3
"""
boltz_profiler.py — Pair representation activation profiler for Boltz-1/2.

Profiles z (pair representation) captured after each PairformerLayer block
during a trunk-only forward pass. Computes M1–M5 metrics for MX-format
quantization research.

Supports both Boltz-1 (48 blocks) and Boltz-2 (64 blocks) checkpoints.
Robustly handles checkpoint ↔ codebase version mismatches.

Usage:
    # With the .venv bundled in this repo:
    .venv/bin/python boltz_profiler.py \\
        --boltz_src src \\
        --ckpt_path ~/.boltz/boltz2_conf.ckpt \\
        --input_dir /path/to/boltz_output_dir \\
        --output_dir ./profiler_results \\
        --device cuda

    # Or after activating the venv:
    source .venv/bin/activate
    python boltz_profiler.py --boltz_src src --ckpt_path ... --input_dir ... --output_dir ...

input_dir layout (Boltz processed output):
    input_dir/
        processed/
            manifest.json
            structures/   <- {protein}.npz
            msa/          <- {msa_id}.npz
            constraints/  <- {protein}.npz   [optional]

  OR a parent dir of per-protein Boltz-output subdirs, each with the above layout.
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from scipy.stats import kurtosis as scipy_kurtosis, pearsonr

warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# Model loading — handles Boltz1 and Boltz2 + version mismatches
# ============================================================

def _patch_checkpoint_hp(hp: dict, cls) -> dict:
    """
    Remove hyper_parameters keys that the current codebase no longer accepts.

    The released boltz2_conf.ckpt may have been built with a slightly different
    codebase (e.g. extra kwargs in score_model_args such as
    'mse_rotational_alignment').  We strip unknown kwargs so the model can be
    instantiated without errors.  Weights are loaded with strict=False to
    tolerate any minor structural differences.
    """
    import inspect  # noqa: PLC0415

    sig = inspect.signature(cls.__init__)
    known_top = set(sig.parameters.keys()) - {"self"}

    patched = {}
    for k, v in hp.items():
        if k not in known_top:
            print(f"[load_model] Dropping unknown hyper-parameter: {k!r}")
            continue
        patched[k] = v
    return patched


def load_model(ckpt_path: str, boltz_src: str, device: str):
    """
    Load a Boltz-1 or Boltz-2 checkpoint, move to device, set eval mode.

    Auto-detects which model class to use by inspecting the checkpoint's
    hyper_parameters (Boltz-2 has pairformer_args with v2=True or 64 blocks).
    Handles minor codebase/checkpoint version mismatches via strict=False.

    Returns (model, model_class_name).
    """
    ckpt_path = os.path.expanduser(ckpt_path)
    if boltz_src not in sys.path:
        sys.path.insert(0, boltz_src)

    from boltz.model.models.boltz1 import Boltz1  # noqa: PLC0415
    from boltz.model.models.boltz2 import Boltz2  # noqa: PLC0415

    print(f"[load_model] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters", {})

    # Detect model class: Boltz2 has 'use_templates' or num_blocks>=64 in pairformer_args
    pf_args = hp.get("pairformer_args", {})
    is_boltz2 = (
        isinstance(pf_args, dict) and (
            pf_args.get("v2", False) or
            pf_args.get("num_blocks", 0) >= 64 or
            "use_templates" in hp
        )
    )
    model_cls = Boltz2 if is_boltz2 else Boltz1
    model_name = "Boltz-2" if is_boltz2 else "Boltz-1"
    print(f"[load_model] Detected: {model_name}")

    # Try standard load first; fall back to manual instantiation on kwarg errors
    try:
        model = model_cls.load_from_checkpoint(ckpt_path, map_location="cpu")
    except TypeError as e:
        print(f"[load_model] Standard load failed ({e}); retrying with patched HP ...")
        patched_hp = _patch_checkpoint_hp(hp, model_cls)
        model = model_cls(**patched_hp)
        state_dict = ckpt.get("state_dict", {})
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[load_model] Missing keys ({len(missing)}): {missing[:5]} ...")
        if unexpected:
            print(f"[load_model] Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")

    model = model.to(device).eval()
    num_blocks = len(model.pairformer_module.layers)
    print(f"[load_model] {model_name} ready on {device}. "
          f"pairformer_module.layers: {num_blocks} blocks, token_z=128.")
    return model, model_name


# ============================================================
# Hook registration
# ============================================================

def register_hooks(model):
    """
    Register forward hooks on every PairformerLayer in
    model.pairformer_module.layers.

    Both Boltz-1 (trunk.PairformerLayer) and Boltz-2 (layers.pairformer.PairformerLayer)
    forward() signatures return (s, z):
      output[0] = s  (single repr)  [B, N, token_s]
      output[1] = z  (pair repr)    [B, N, N, 128]

    The hook squeezes the batch dimension (B=1 during inference) and
    casts to float32 before storing.

    Returns
    -------
    activations : dict[int -> Tensor]  block_idx -> z [N, N, 128]
    handles     : list of hook handles (call .remove() to detach)
    """
    activations: dict = {}
    handles = []

    def make_hook(block_idx: int):
        def hook(module, input, output):
            # output[1] is z; cast bf16->f32, drop batch dim
            z_out = output[1].float().squeeze(0).detach().cpu()  # [N, N, 128]
            activations[block_idx] = z_out
        return hook

    for idx, layer in enumerate(model.pairformer_module.layers):
        h = layer.register_forward_hook(make_hook(idx))
        handles.append(h)

    print(f"[register_hooks] Hooked {len(handles)} PairformerLayer blocks.")
    return activations, handles


def remove_hooks(handles: list) -> None:
    for h in handles:
        h.remove()


# ============================================================
# Input loading helpers
# ============================================================

def _load_features_from_processed_dir(
    processed_dir: Path, boltz_src: str
) -> list[tuple[str, dict]]:
    """
    Load all protein records from a Boltz-processed directory.

    Returns list of (protein_name, feature_dict) pairs ready for inference.
    Each feature_dict has a batch dimension of 1 (collated).
    """
    if boltz_src not in sys.path:
        sys.path.insert(0, boltz_src)

    from boltz.data.module.inference import PredictionDataset  # noqa: PLC0415
    from boltz.data.types import Manifest  # noqa: PLC0415

    manifest_path = processed_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in {processed_dir}")

    manifest = Manifest.load(manifest_path)
    target_dir = processed_dir / "structures"
    msa_dir = processed_dir / "msa"
    constraints_dir = processed_dir / "constraints"
    if not constraints_dir.exists():
        constraints_dir = None

    dataset = PredictionDataset(
        manifest=manifest,
        target_dir=target_dir,
        msa_dir=msa_dir,
        constraints_dir=constraints_dir,
    )

    results = []
    for idx, record in enumerate(manifest.records):
        feats = dataset[idx]
        # Add batch dimension (collate single item)
        feats_batched = {}
        for k, v in feats.items():
            if isinstance(v, torch.Tensor):
                feats_batched[k] = v.unsqueeze(0)
            else:
                feats_batched[k] = v
        results.append((record.id, feats_batched))

    return results


def _discover_protein_inputs(input_dir: Path, boltz_src: str) -> list[tuple[str, dict]]:
    """
    Discover protein inputs from input_dir.

    Supported layouts:
      1. input_dir/processed/manifest.json  — single Boltz output dir
      2. input_dir/<name>/processed/manifest.json  — per-protein subdirs
    """
    all_proteins: list[tuple[str, dict]] = []

    # Layout 1: input_dir itself is a Boltz output dir
    if (input_dir / "processed" / "manifest.json").exists():
        print(f"[input] Single processed Boltz dir: {input_dir}")
        proteins = _load_features_from_processed_dir(
            input_dir / "processed", boltz_src
        )
        all_proteins.extend(proteins)
        return all_proteins

    # Layout 2: per-protein subdirectories
    subdirs = sorted([p for p in input_dir.iterdir() if p.is_dir()])
    if subdirs:
        for subdir in subdirs:
            processed = subdir / "processed"
            if processed.is_dir() and (processed / "manifest.json").exists():
                print(f"[input] Loading protein subdir: {subdir.name}")
                proteins = _load_features_from_processed_dir(processed, boltz_src)
                all_proteins.extend(proteins)

    if all_proteins:
        return all_proteins

    raise RuntimeError(
        f"Could not find any Boltz-processed inputs in {input_dir}.\n"
        "Expected either:\n"
        "  {input_dir}/processed/manifest.json\n"
        "  {input_dir}/<protein>/processed/manifest.json\n"
        "Pre-process proteins with 'boltz predict' before profiling."
    )


# ============================================================
# Trunk-only forward pass
# ============================================================

def run_trunk_only(model, feats: dict, device: str) -> None:
    """
    Execute the Boltz trunk (InputEmbedder + MSA/templates + Pairformer) without
    the expensive diffusion/confidence modules.  All registered PairformerLayer
    hooks fire during this call, populating the activations dict.

    Supports both Boltz-1 and Boltz-2 models: extra Boltz-2 components
    (contact_conditioning, bond_type_feature, template_module) are detected
    at runtime via hasattr/getattr and only used when present.
    """
    feats_dev = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in feats.items()
    }

    with torch.no_grad():
        s_inputs = model.input_embedder(feats_dev)

        s_init = model.s_init(s_inputs)
        z_init = (
            model.z_init_1(s_inputs)[:, :, None]
            + model.z_init_2(s_inputs)[:, None, :]
        )
        rel_pos = model.rel_pos(feats_dev)
        z_init = z_init + rel_pos
        z_init = z_init + model.token_bonds(feats_dev["token_bonds"].float())

        # Boltz-2 only: bond type feature
        if getattr(model, "bond_type_feature", False):
            z_init = z_init + model.token_bonds_type(feats_dev["type_bonds"].long())

        # Boltz-2 only: contact conditioning
        if hasattr(model, "contact_conditioning"):
            z_init = z_init + model.contact_conditioning(feats_dev)

        mask = feats_dev["token_pad_mask"].float()          # [B, N]
        pair_mask = mask[:, :, None] * mask[:, None, :]    # [B, N, N]

        s = torch.zeros_like(s_init)
        z = torch.zeros_like(z_init)

        recycling_steps = 3
        for _ in range(recycling_steps + 1):
            s = s_init + model.s_recycle(model.s_norm(s))
            z = z_init + model.z_recycle(model.z_norm(z))

            # Boltz-2 only: template module
            if getattr(model, "use_templates", False):
                z = z + model.template_module(z, feats_dev, pair_mask, use_kernels=False)

            # MSA module (Boltz-1 may have no_msa flag; Boltz-2 always has MSA)
            if not getattr(model, "no_msa", False):
                z = z + model.msa_module(z, s_inputs, feats_dev, use_kernels=False)

            # PairformerLayer hooks fire here
            s, z = model.pairformer_module(
                s, z, mask=mask, pair_mask=pair_mask, use_kernels=False
            )


# ============================================================
# M1: Basic Statistics
# ============================================================

def compute_m1_stats(z: torch.Tensor) -> dict:
    """z: float32 [N, N, 128]"""
    z_abs = z.abs()
    nonzero_mask = z_abs > 0
    absmin_nz = z_abs[nonzero_mask].amin().item() if nonzero_mask.any() else 0.0
    absmax = z_abs.amax().item()
    return {
        "mean":             float(z.mean().item()),
        "std":              float(z.std().item()),
        "absmax":           float(absmax),
        "absmin_nonzero":   float(absmin_nz),
        "dynamic_range":    float(absmax / max(absmin_nz, 1e-10)),
        "outlier_fraction": float((z_abs > 3.0 * z.std()).float().mean().item()),
    }


# ============================================================
# M2: Kurtosis
# ============================================================

def compute_m2_kurtosis(z: torch.Tensor) -> dict:
    """z: float32 [N, N, 128]"""
    z_np = z.numpy()

    global_kurt = float(scipy_kurtosis(z_np.flatten(), fisher=True))

    per_hz_block = [
        float(scipy_kurtosis(z_np[..., b * 32:(b + 1) * 32].flatten(), fisher=True))
        for b in range(4)
    ]

    per_channel = [
        scipy_kurtosis(z_np[:, :, c].flatten(), fisher=True)
        for c in range(128)
    ]

    return {
        "global":              global_kurt,
        "per_hz_block":        per_hz_block,
        "per_channel_mean":    float(np.mean(per_channel)),
        "per_channel_std":     float(np.std(per_channel)),
    }


# ============================================================
# M3: Outlier Structure
# ============================================================

def compute_m3_outlier_structure(z: torch.Tensor) -> dict:
    """z: float32 [N, N, 128]"""
    z_abs = z.abs()

    # Channel-wise: which Hz channels are persistently hot?
    channel_absmax = z_abs.amax(dim=(0, 1))  # [128]
    channel_mean   = z_abs.mean(dim=(0, 1))  # [128]
    outlier_channels = int((channel_absmax > 3.0 * channel_absmax.mean()).sum().item())

    # Spatial-wise: which (i,j) positions are hot?
    spatial_absmax = z_abs.amax(dim=-1)      # [N, N]
    spatial_mean   = z_abs.mean(dim=-1)      # [N, N]
    outlier_positions = float(
        (spatial_absmax > 3.0 * spatial_absmax.mean()).float().mean().item()
    )

    channel_outlier_fraction = outlier_channels / 128.0
    spatial_outlier_fraction = outlier_positions

    if channel_outlier_fraction > spatial_outlier_fraction:
        dominant = "channel"
    elif spatial_outlier_fraction > channel_outlier_fraction:
        dominant = "spatial"
    else:
        dominant = "mixed"

    return {
        "channel_absmax":            channel_absmax.numpy().tolist(),
        "spatial_absmax":            spatial_absmax.numpy().tolist(),
        "outlier_channels_count":    outlier_channels,
        "outlier_spatial_fraction":  spatial_outlier_fraction,
        "dominant":                  dominant,
    }


# ============================================================
# M4: Within-Tile Scale Variance (TAMX hypothesis)
# ============================================================

def compute_m4_tile_coherence(
    z: torch.Tensor, tile_sizes: list = None
) -> dict:
    """
    z: float32 [N, N, 128]
    cv_mean < 0.1  -> TAMX very safe
    cv_mean < 0.3  -> TAMX acceptable
    cv_mean > 0.5  -> TAMX too coarse
    """
    if tile_sizes is None:
        tile_sizes = [4, 8, 16, 32]

    result = {}
    N = z.shape[0]

    for T in tile_sizes:
        N_pad = ((N + T - 1) // T) * T
        pad_len = N_pad - N
        # Pad the row dimension (dim=0) with zeros
        z_pad = F.pad(z, (0, 0, 0, 0, 0, pad_len))  # [N_pad, N, 128]

        cvs_row = []
        for b in range(4):
            hz = z_pad[..., b * 32:(b + 1) * 32]             # [N_pad, N, 32]
            hz_tiled = hz.reshape(-1, T, N, 32)               # [N_pad/T, T, N, 32]
            scales = hz_tiled.abs().amax(dim=-1)              # [N_pad/T, T, N]
            denom = scales.mean(dim=1, keepdim=True).clamp(min=1e-8)
            cv = scales.std(dim=1) / denom.squeeze(1)         # [N_pad/T, N]
            cvs_row.append(float(cv.mean().item()))

        result[f"T{T}_row_cv_mean"] = float(np.mean(cvs_row))
        result[f"T{T}_row_cv_std"]  = float(np.std(cvs_row))

    return result


# ============================================================
# M5: Sequence Distance Correlation
# ============================================================

def compute_m5_seq_dist_corr(z: torch.Tensor) -> dict:
    """z: float32 [N, N, 128]"""
    N = z.shape[0]
    scale_map = z.abs().amax(dim=-1).numpy()   # [N, N]

    idx = np.arange(N)
    dist_matrix = np.abs(idx[:, None] - idx[None, :])  # [N, N]
    proximity = 1.0 / (1.0 + dist_matrix)

    r, p = pearsonr(scale_map.flatten(), proximity.flatten())
    return {
        "pearson_r": float(r),
        "p_value":   float(p),
    }


# ============================================================
# Full metrics computation for one (protein, block) pair
# ============================================================

def compute_all_metrics(z: torch.Tensor, protein: str, block_idx: int) -> dict:
    """Run M1–M5 on z [N, N, 128] (float32)."""
    print(f"  [metrics] {protein} block {block_idx:02d} — N={z.shape[0]}", end="  ")

    m1 = compute_m1_stats(z)
    print("M1", end=" ")

    m2 = compute_m2_kurtosis(z)
    print("M2", end=" ")

    m3 = compute_m3_outlier_structure(z)
    print("M3", end=" ")

    m4 = compute_m4_tile_coherence(z)
    print("M4", end=" ")

    m5 = compute_m5_seq_dist_corr(z)
    print("M5")

    return {
        "stats":             m1,
        "kurtosis":          m2,
        "outlier_structure": m3,
        "tile_coherence":    m4,
        "seq_dist_corr":     m5,
    }


# ============================================================
# Save results
# ============================================================

def save_results(
    all_results: dict,
    output_dir: Path,
    raw_activations: dict,
) -> None:
    """
    Save raw per-block NPZ files and a summary JSON.

    all_results : { protein_name: { block_idx: { M1..M5 metrics } } }
    raw_activations : { protein_name: { block_idx: z tensor [N, N, 128] } }
    """
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for protein, blocks in raw_activations.items():
        for block_idx, z in blocks.items():
            npz_path = raw_dir / f"{protein}_{block_idx:02d}.npz"
            metrics = all_results[protein][block_idx]

            np.savez_compressed(
                str(npz_path),
                z_channel_absmax=np.array(
                    metrics["outlier_structure"]["channel_absmax"]
                ),
                z_spatial_absmax=np.array(
                    metrics["outlier_structure"]["spatial_absmax"]
                ),
                **{f"m1_{k}": np.array(v)
                   for k, v in metrics["stats"].items()},
                **{f"m2_{k}": np.array(v)
                   for k, v in metrics["kurtosis"].items()
                   if not isinstance(v, list)},
                m2_per_hz_block=np.array(metrics["kurtosis"]["per_hz_block"]),
                **{f"m4_{k}": np.array(v)
                   for k, v in metrics["tile_coherence"].items()},
                **{f"m5_{k}": np.array(v)
                   for k, v in metrics["seq_dist_corr"].items()},
            )

    # summary.json — replace non-serialisable list-of-lists in spatial_absmax
    summary: dict = {}
    for protein, blocks in all_results.items():
        summary[protein] = {}
        for block_idx, metrics in blocks.items():
            m = dict(metrics)
            os_copy = dict(m["outlier_structure"])
            # Truncate large spatial_absmax for JSON readability
            # (full data is in the NPZ)
            os_copy["spatial_absmax"] = "<saved in NPZ>"
            m = dict(m)
            m["outlier_structure"] = os_copy
            summary[protein][str(block_idx)] = m

    json_path = output_dir / "summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] Summary written to {json_path}")


# ============================================================
# Plotting helpers
# ============================================================

def _subsample_spatial(arr: np.ndarray, max_size: int = 128) -> np.ndarray:
    """Subsample a 2-D spatial array to at most max_size × max_size for plotting."""
    N = arr.shape[0]
    if N <= max_size:
        return arr
    step = N // max_size
    return arr[::step, ::step][:max_size, :max_size]


def plot_all(all_results: dict, output_dir: Path) -> None:
    sns.set_theme(style="whitegrid")
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    proteins = list(all_results.keys())
    num_blocks = max(
        int(b) for blocks in all_results.values() for b in blocks
    ) + 1
    block_indices = list(range(num_blocks))

    palette = sns.color_palette("tab10", n_colors=len(proteins))

    # ------------------------------------------------------------------
    # PLOT 1 — Kurtosis across blocks
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5))
    for color, protein in zip(palette, proteins):
        kurt_values = [
            all_results[protein][b]["kurtosis"]["global"]
            for b in block_indices
        ]
        ax.plot(block_indices, kurt_values, label=protein, color=color, linewidth=1.5)
    ax.axhline(0.0, linestyle="--", color="gray",  linewidth=1.0, label="Gaussian (k=0)")
    ax.axhline(3.0, linestyle="--", color="orange", linewidth=1.0, label="Laplace (k=3)")
    ax.set_xlabel("Block index")
    ax.set_ylabel("Excess kurtosis (Fisher)")
    ax.set_title("Pair Representation Kurtosis Across Pairformer Blocks")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "kurtosis_across_blocks.png", dpi=300)
    plt.close(fig)
    print("[plot] kurtosis_across_blocks.png")

    # ------------------------------------------------------------------
    # PLOT 2 — Outlier structure heatmaps (per protein)
    # ------------------------------------------------------------------
    for protein in proteins:
        # Pick the last block for the representative heatmap
        last_block = max(int(b) for b in all_results[protein])
        metrics_last = all_results[protein][last_block]
        channel_absmax = np.array(metrics_last["outlier_structure"]["channel_absmax"])
        spatial_absmax_raw = np.array(metrics_last["outlier_structure"]["spatial_absmax"])
        spatial_sub = _subsample_spatial(spatial_absmax_raw)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: channel bar chart
        ax_ch = axes[0]
        ax_ch.bar(np.arange(128), channel_absmax, color="steelblue", width=1.0)
        ax_ch.axhline(
            3.0 * channel_absmax.mean(), linestyle="--", color="red",
            linewidth=1.2, label="3× mean"
        )
        ax_ch.set_xlabel("Hz channel index")
        ax_ch.set_ylabel("|z| absmax")
        ax_ch.set_title("Channel-wise |z| absmax (block 47)")
        ax_ch.legend(fontsize=8)

        # Right: spatial heatmap
        ax_sp = axes[1]
        im = ax_sp.imshow(spatial_sub, cmap="viridis", aspect="auto")
        fig.colorbar(im, ax=ax_sp, shrink=0.8)
        N_orig = spatial_absmax_raw.shape[0]
        ax_sp.set_title(
            f"Spatial |z| absmax (block 47, subsampled {spatial_sub.shape[0]}×{spatial_sub.shape[1]}"
            f" from {N_orig}×{N_orig})"
        )
        ax_sp.set_xlabel("j (token)")
        ax_sp.set_ylabel("i (token)")

        fig.suptitle(f"Outlier Location: Channel vs Spatial — {protein}", fontsize=13)
        fig.tight_layout()
        fname = plots_dir / f"outlier_structure_{protein}.png"
        fig.savefig(fname, dpi=300)
        plt.close(fig)
        print(f"[plot] outlier_structure_{protein}.png")

    # ------------------------------------------------------------------
    # PLOT 3 — TAMX tile coherence grouped bar chart
    # ------------------------------------------------------------------
    tile_sizes = [4, 8, 16, 32]
    x = np.arange(len(tile_sizes))
    bar_width = 0.8 / len(proteins)

    fig, ax = plt.subplots(figsize=(10, 5))
    for j, (color, protein) in enumerate(zip(palette, proteins)):
        # Average CV across all blocks for each tile size
        cv_means = []
        for T in tile_sizes:
            key = f"T{T}_row_cv_mean"
            vals = [all_results[protein][b]["tile_coherence"][key]
                    for b in block_indices]
            cv_means.append(float(np.mean(vals)))
        offset = (j - len(proteins) / 2.0 + 0.5) * bar_width
        ax.bar(x + offset, cv_means, width=bar_width, label=protein, color=color)

    ax.axhline(0.1, linestyle="--", color="green",  linewidth=1.2, label="CV=0.1 (safe)")
    ax.axhline(0.3, linestyle="--", color="orange", linewidth=1.2, label="CV=0.3 (acceptable)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"T={t}" for t in tile_sizes])
    ax.set_xlabel("Tile size T")
    ax.set_ylabel("Mean CV (within-tile scale variation)")
    ax.set_title("Within-Tile Scale Variance vs Tile Size")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "tamx_tile_coherence.png", dpi=300)
    plt.close(fig)
    print("[plot] tamx_tile_coherence.png")

    # ------------------------------------------------------------------
    # PLOT 4 — Dynamic range across blocks (log scale)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5))
    for color, protein in zip(palette, proteins):
        dr_values = [
            all_results[protein][b]["stats"]["dynamic_range"]
            for b in block_indices
        ]
        ax.plot(block_indices, dr_values, label=protein, color=color, linewidth=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("Block index")
    ax.set_ylabel("Dynamic range (absmax / absmin_nonzero)")
    ax.set_title("Pair Representation Dynamic Range Across Blocks")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "dynamic_range_across_blocks.png", dpi=300)
    plt.close(fig)
    print("[plot] dynamic_range_across_blocks.png")

    # ------------------------------------------------------------------
    # PLOT 5 — Sequence distance correlation across blocks
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5))
    for color, protein in zip(palette, proteins):
        r_values = [
            all_results[protein][b]["seq_dist_corr"]["pearson_r"]
            for b in block_indices
        ]
        ax.plot(block_indices, r_values, label=protein, color=color, linewidth=1.5)
    ax.axhline(0.3, linestyle="--", color="red", linewidth=1.0, label="r=0.3 (threshold)")
    ax.set_xlabel("Block index")
    ax.set_ylabel("Pearson r (scale_map vs proximity)")
    ax.set_title("Scale Map vs Sequence Proximity Correlation")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "seq_dist_correlation.png", dpi=300)
    plt.close(fig)
    print("[plot] seq_dist_correlation.png")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Profile Boltz-1 pair representation activations for MX quantization research."
    )
    parser.add_argument(
        "--boltz_src",
        required=True,
        help="Path to boltz/src directory (added to sys.path).",
    )
    parser.add_argument(
        "--ckpt_path",
        default="~/.boltz/boltz1_conf.ckpt",
        help=(
            "Path to Boltz-1 or Boltz-2 checkpoint (.ckpt). "
            "Default: ~/.boltz/boltz1_conf.ckpt  "
            "(Boltz-2: ~/.boltz/boltz2_conf.ckpt)"
        ),
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help=(
            "Boltz-processed output directory, or parent directory of "
            "per-protein Boltz-processed subdirectories.\n"
            "Each entry must have processed/manifest.json."
        ),
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where raw NPZ, summary JSON, and plots are saved.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Compute device: 'cuda', 'cpu', or 'cuda:N'. Default: cuda.",
    )
    parser.add_argument(
        "--proteins",
        nargs="*",
        default=None,
        help=(
            "Optional whitelist of protein IDs to process. "
            "If omitted, all proteins found in input_dir are processed."
        ),
    )
    args = parser.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA not available, falling back to CPU.")
        device = "cpu"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load model ----
    model, model_name = load_model(args.ckpt_path, args.boltz_src, device)

    # ---- Register hooks ----
    activations, hook_handles = register_hooks(model)

    # ---- Discover inputs ----
    print(f"\n[input] Scanning {args.input_dir} ...")
    all_proteins = _discover_protein_inputs(Path(args.input_dir), args.boltz_src)

    if args.proteins:
        whitelist = set(args.proteins)
        all_proteins = [(name, feats) for name, feats in all_proteins
                        if name in whitelist]
        print(f"[input] Filtered to: {[n for n, _ in all_proteins]}")

    if not all_proteins:
        print("[error] No proteins found. Exiting.")
        sys.exit(1)

    num_blocks = len(model.pairformer_module.layers)
    print(f"[input] Will profile {len(all_proteins)} protein(s) × {num_blocks} blocks ({model_name}).")

    # ---- Profile each protein ----
    all_results: dict = {}        # protein -> block_idx -> metrics
    raw_activations_store: dict = {}  # protein -> block_idx -> z

    for protein_name, feats in all_proteins:
        N = feats["token_pad_mask"].shape[-1]
        print(f"\n{'='*60}")
        print(f"[run] Protein: {protein_name}  (N={N} tokens)")
        print(f"{'='*60}")

        # Clear activations from previous protein
        activations.clear()

        # Forward pass — hooks populate activations dict
        run_trunk_only(model, feats, device)

        num_blocks_captured = len(activations)
        print(f"[run] Captured {num_blocks_captured} block activations.")

        if num_blocks_captured == 0:
            print(f"[warn] No activations captured for {protein_name}, skipping.")
            continue

        # Compute metrics for each block
        protein_results: dict = {}
        protein_raw: dict = {}

        for block_idx in sorted(activations.keys()):
            z = activations[block_idx]  # float32 [N, N, 128]
            assert z.dtype == torch.float32, f"Expected float32, got {z.dtype}"
            assert z.ndim == 3 and z.shape[-1] == 128, f"Unexpected z shape: {z.shape}"

            metrics = compute_all_metrics(z, protein_name, block_idx)
            protein_results[block_idx] = metrics
            protein_raw[block_idx] = z

        all_results[protein_name] = protein_results
        raw_activations_store[protein_name] = protein_raw

        print(f"[run] {protein_name}: done ({num_blocks_captured} blocks).")

    # ---- Save results ----
    print(f"\n[save] Writing results to {output_dir} ...")
    save_results(all_results, output_dir, raw_activations_store)

    # ---- Plot ----
    print("\n[plot] Generating plots ...")
    plot_all(all_results, output_dir)

    print(f"\n[done] All outputs written to {output_dir}/")
    print("  raw/              — per-block NPZ files with all metric arrays")
    print("  summary.json      — all scalar metrics (JSON)")
    print("  plots/            — 5 analysis plots (PNG, 300 DPI)")


if __name__ == "__main__":
    main()
