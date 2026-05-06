#!/usr/bin/env python3
"""
fp8_pairformer_simulation.py

FP8 mixed-precision simulation on Boltz-2's 64-layer Pairformer.
Validates the doubly-sparse outlier hypothesis by measuring per-layer
error accumulation under five quantization strategies.

Conditions
----------
  1. BF16 baseline        — no quantization (reference)
  2. Naive FP8            — quantize entire z tensor
  3. Channel-only         — keep LOUD_CHANNELS in BF16, quantize rest
  4. Diagonal-only        — keep diagonal (i==j) in BF16, quantize rest
  5. Doubly-sparse        — keep diagonal AND LOUD_CHANNELS in BF16, quantize rest

Outputs (written to --out_dir)
-------------------------------
  fp8_simulation_results.json
  plot1_per_layer_mae.png
  plot2_per_layer_cosine.png
  plot3_ablation.png
  plot4_summary_table.png

Usage
-----
  python fp8_pairformer_simulation.py \\
      [--processed_dir test_output_boltz2/medium_386aa/processed] \\
      [--cache ~/.boltz] [--out_dir ./fp8_results] \\
      [--use_msa_server] [--cpu] [--no_bf16] [--seed 42]
"""

import argparse
import json
import multiprocessing
import os
import sys
import time
import traceback
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Disable cuEquivariance kernel tuning before any Boltz imports
os.environ.setdefault("CUEQ_DEFAULT_CONFIG", "1")
os.environ.setdefault("CUEQ_DISABLE_AOT_TUNING", "1")

# ---------------------------------------------------------------------------
# Bootstrap path so we can import boltz from the src/ directory
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

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
from boltz.data.module.inferencev2 import Boltz2InferenceDataModule  # noqa: E402
from boltz.data.types import Manifest                                  # noqa: E402
from boltz.model.models.boltz2 import Boltz2                          # noqa: E402
from boltz.model.layers.pairformer import PairformerLayer              # noqa: E402

# ============================================================================
# Constants
# ============================================================================

LOUD_CHANNELS: List[int] = [51, 36, 105, 65]

FP8_MAX: float = 448.0          # E4M3 max representable value
FP8_MIN_NORMAL: float = 6.103515625e-05  # E4M3 min normal value

CONDITIONS: List[str] = ["naive_fp8", "channel_only", "diagonal_only", "doubly_sparse"]

CONDITION_COLORS: Dict[str, str] = {
    "naive_fp8":     "red",
    "channel_only":  "orange",
    "diagonal_only": "purple",
    "doubly_sparse": "green",
}

CONDITION_LABELS: Dict[str, str] = {
    "naive_fp8":     "Naive FP8",
    "channel_only":  "Channel-only",
    "diagonal_only": "Diagonal-only",
    "doubly_sparse": "Doubly-sparse",
}

# Layers to report in the headline table
REPORT_LAYERS: List[int] = [0, 16, 32, 48, 63]

# HSP70 fragment, 386 aa (PDB 3HSC chain A) — used when data is not pre-processed
_MEDIUM_SEQ: str = (
    "MSKGPAVGIDLGTTYSCVGVFQHGKVEIIANDQGNRTTPSYVAFTDTERLIGDAAKNQVAMNPTNTVFDA"
    "KRLIGRRFDDAVVQSDMKHWPFMVVNDAGRPKVQVEYKGETKSFYPEEVSSMVLTKMKEIAEAYLGKTVTN"
    "AVVTVPAYFNDSQRQATKDAGTIAGLNVLRIINEPTAAAIAYGLDKKVGAERNVLIFDLGGGTFDVSILTIED"
    "GIFEVKSTAGDTHLGGEDFDNRMVNHFIAEFKRKHKKDISENKRAVRRLRTACERAKRTLSSSTQASIEIDS"
    "LYEGIDFYTSITRARFEELNADLFRGTLDPVEKALRDAKLDKSQIHDIVLVGGSTRIPKIQKLLQDFFNGKE"
    "LNKSINPDEAVAYGAAVQAAILSGDKSE"
)

# ============================================================================
# FP8 support detection
# ============================================================================

def _check_native_fp8() -> bool:
    """Return True if PyTorch supports torch.float8_e4m3fn natively."""
    try:
        _ = torch.zeros(1, dtype=torch.float8_e4m3fn)
        return True
    except (AttributeError, TypeError, RuntimeError):
        return False


HAS_NATIVE_FP8: bool = _check_native_fp8()

# ============================================================================
# Quantization functions
# ============================================================================

def _simulate_fp8_bf16(x: torch.Tensor, n_levels: int = 240) -> torch.Tensor:
    """BF16 simulation of FP8 E4M3 quantize-dequantize (uniform approximation).

    Parameters
    ----------
    x : torch.Tensor
        Input tensor (any float dtype).
    n_levels : int
        Number of positive quantization levels.  E4M3 has ~240 distinct positive
        values, so 240 is a reasonable approximation.

    Returns
    -------
    torch.Tensor
        Simulated-quantized tensor in the same dtype as *x*.
    """
    scale = x.abs().max() / 448.0
    scale = torch.clamp(scale, min=1e-8)
    x_scaled = (x / scale).clamp(-448.0, 448.0)
    step = 448.0 / n_levels
    x_q = (x_scaled / step).round() * step
    return x_q * scale


def _native_fp8_quant(x: torch.Tensor) -> torch.Tensor:
    """Cast x to torch.float8_e4m3fn and back (native hardware path)."""
    scale = x.abs().max() / FP8_MAX
    scale = torch.clamp(scale, min=FP8_MIN_NORMAL)
    x_scaled = x / scale
    return x_scaled.to(torch.float8_e4m3fn).to(x.dtype) * scale


def fp8_quantize(
    x_bf16: torch.Tensor,
    mask_keep_bf16: Optional[torch.Tensor] = None,
    use_native: bool = False,
) -> torch.Tensor:
    """Quantize *x_bf16* to FP8 E4M3 and back.

    Parameters
    ----------
    x_bf16 : torch.Tensor
        Input tensor (BF16 or FP32).
    mask_keep_bf16 : bool tensor or None
        If provided, positions where True are kept in BF16 (not quantized).
    use_native : bool
        Use torch.float8_e4m3fn if True, otherwise BF16 simulation.

    Returns
    -------
    torch.Tensor
        Quantized tensor in the same dtype as *x_bf16*.
    """
    if mask_keep_bf16 is None:
        # Quantize everything
        if use_native:
            return _native_fp8_quant(x_bf16)
        return _simulate_fp8_bf16(x_bf16)

    # Zero out masked (kept-in-BF16) positions so the scale is computed
    # only from the values that will actually be quantized.
    x_quant_only = torch.where(mask_keep_bf16, torch.zeros_like(x_bf16), x_bf16)

    if use_native:
        scale = x_quant_only.abs().max() / FP8_MAX
        scale = torch.clamp(scale, min=FP8_MIN_NORMAL)
        x_scaled = x_quant_only / scale
        x_fp8 = x_scaled.to(torch.float8_e4m3fn).to(x_bf16.dtype) * scale
    else:
        x_fp8 = _simulate_fp8_bf16(x_quant_only)

    # Restore original BF16 values at masked positions
    return torch.where(mask_keep_bf16, x_bf16, x_fp8)

# ============================================================================
# Mask creation helpers
# ============================================================================

def make_diag_mask(
    shape: Tuple[int, int, int, int],
    device: torch.device,
) -> torch.Tensor:
    """Boolean mask: True at diagonal positions (i==j), all channels."""
    B, N1, N2, D = shape
    mask = torch.zeros(shape, dtype=torch.bool, device=device)
    diag_idx = torch.arange(min(N1, N2), device=device)
    mask[:, diag_idx, diag_idx, :] = True
    return mask


def make_channel_mask(
    shape: Tuple[int, int, int, int],
    channels: List[int],
    device: torch.device,
) -> torch.Tensor:
    """Boolean mask: True for the given channels at all spatial positions."""
    B, N1, N2, D = shape
    mask = torch.zeros(shape, dtype=torch.bool, device=device)
    for ch in channels:
        if ch < D:
            mask[:, :, :, ch] = True
    return mask


def make_doubly_sparse_mask(
    shape: Tuple[int, int, int, int],
    device: torch.device,
) -> torch.Tensor:
    """Boolean mask: True at diagonal OR loud-channel positions (union)."""
    return make_diag_mask(shape, device) | make_channel_mask(shape, LOUD_CHANNELS, device)


def get_mask_for_condition(
    condition: str,
    z_shape: Tuple[int, int, int, int],
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Return the appropriate mask_keep_bf16 tensor for a quantization condition."""
    if condition == "naive_fp8":
        return None  # quantize everything — no mask
    if condition == "channel_only":
        return make_channel_mask(z_shape, LOUD_CHANNELS, device)
    if condition == "diagonal_only":
        return make_diag_mask(z_shape, device)
    if condition == "doubly_sparse":
        return make_doubly_sparse_mask(z_shape, device)
    raise ValueError(f"Unknown condition: {condition!r}")

# ============================================================================
# Per-layer metrics
# ============================================================================

def compute_metrics(
    z_baseline: torch.Tensor,
    z_condition: torch.Tensor,
) -> Dict[str, float]:
    """Compare two pair tensors and return accuracy metrics.

    All arithmetic is done in float32 for numerical accuracy.
    """
    z_b = z_baseline.float()
    z_c = z_condition.float()
    diff = (z_c - z_b).abs()
    cosine_sim = F.cosine_similarity(z_c.flatten(), z_b.flatten(), dim=0).item()
    baseline_mean_abs = z_b.abs().mean().item()
    return {
        "mean_abs_error": diff.mean().item(),
        "max_abs_error":  diff.max().item(),
        "rms_error":      (diff ** 2).mean().sqrt().item(),
        "cosine_sim":     cosine_sim,
        "relative_error": diff.mean().item() / (baseline_mean_abs + 1e-8),
    }

# ============================================================================
# Model loading
# ============================================================================

def load_boltz2(cache: Path, checkpoint: Optional[str] = None) -> Boltz2:
    """Load Boltz-2 from checkpoint into CPU memory (hooks need eval mode)."""
    ckpt = checkpoint or str(cache / "boltz2_conf.ckpt")
    print(f"  Loading Boltz-2 from: {ckpt}")

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
    steering_args   = BoltzSteeringParams()

    model: Boltz2 = Boltz2.load_from_checkpoint(
        ckpt,
        strict=True,
        predict_args={
            "recycling_steps":         0,
            "sampling_steps":          10,
            "diffusion_samples":       1,
            "max_parallel_samples":    1,
            "write_confidence_summary": False,
            "write_full_pae":           False,
            "write_full_pde":           False,
        },
        map_location="cpu",
        diffusion_process_args=asdict(diffusion_params),
        ema=False,
        use_kernels=False,   # must be False for hooks to fire
        pairformer_args=asdict(pairformer_args),
        msa_args=asdict(msa_args),
        steering_args=asdict(steering_args),
    )
    model.eval()
    model.use_kernels = False  # belt-and-suspenders
    return model

# ============================================================================
# Data preprocessing / loading
# ============================================================================

def ensure_processed_data(
    processed_dir: Path,
    cache: Path,
    work_dir: Path,
    use_msa_server: bool,
) -> Path:
    """Ensure `processed_dir/manifest.json` exists.

    If the directory already contains preprocessed data it is reused.
    Otherwise, a YAML is written from `_MEDIUM_SEQ` and `process_inputs`
    is called (requires either a pre-generated MSA or --use_msa_server).
    """
    manifest_path = processed_dir / "manifest.json"
    if manifest_path.exists():
        print(f"  Found preprocessed data at {processed_dir}")
        return processed_dir

    print(f"  Preprocessed data not found at {processed_dir}, building it…")
    work_dir.mkdir(parents=True, exist_ok=True)

    # Write a minimal Boltz-2 YAML for the medium (386 aa) protein.
    # msa: empty avoids requiring the MSA server while still exercising the
    # full Pairformer stack (MSA module receives zero-filled features).
    yaml_path = work_dir / "medium_386aa.yaml"
    seq = _MEDIUM_SEQ.replace("\n", "").replace(" ", "")
    yaml_path.write_text(
        "version: 1\n"
        "sequences:\n"
        "  - protein:\n"
        "      id: A\n"
        f"      sequence: {seq}\n"
        "      msa: empty\n"
    )
    print(f"  Created YAML: {yaml_path}  ({len(seq)} aa)")

    try:
        data_list = check_inputs(yaml_path)
    except Exception as exc:
        raise RuntimeError(f"check_inputs failed: {exc}") from exc

    manifest = process_inputs(
        data=data_list,
        out_dir=work_dir,
        ccd_path=cache / "ccd.pkl",
        mol_dir=cache / "mols",
        msa_server_url="https://api.colabfold.com",
        msa_pairing_strategy="greedy",
        use_msa_server=use_msa_server,
        boltz2=True,
        preprocessing_threads=min(4, multiprocessing.cpu_count()),
    )

    # Fallback: try to load the manifest that process_inputs may have written
    if not manifest or not getattr(manifest, "records", None):
        if manifest_path.exists():
            manifest = Manifest.load(manifest_path)
    if not manifest or not getattr(manifest, "records", None):
        raise RuntimeError(
            "Preprocessing produced an empty manifest.\n"
            "If MSA generation is needed, pass --use_msa_server."
        )

    return processed_dir


def load_batch(
    processed_dir: Path,
    cache: Path,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Load the first (and typically only) batch from the processed data."""
    manifest = Manifest.load(processed_dir / "manifest.json")
    constraints_dir = processed_dir / "constraints"
    processed = BoltzProcessedInput(
        manifest=manifest,
        targets_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa",
        constraints_dir=constraints_dir if constraints_dir.exists() else None,
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
        raise RuntimeError("DataLoader is empty — no records loaded.")

    def _to(obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        if isinstance(obj, dict):
            return {k: _to(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to(v) for v in obj]
        return obj

    return _to(batch)

# ============================================================================
# Trunk-only forward pass (no diffusion / confidence modules)
# ============================================================================

def run_trunk_forward(
    model: Boltz2,
    feats: Dict[str, torch.Tensor],
    device: torch.device,
    use_bf16: bool,
    recycling_steps: int = 0,
) -> None:
    """Run Boltz-2 embedding + MSA + Pairformer without diffusion or confidence."""
    ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if use_bf16 and device.type == "cuda"
        else torch.no_grad()
    )

    with torch.no_grad(), ctx:
        s_inputs = model.input_embedder(feats)
        s_init   = model.s_init(s_inputs)
        z_init   = (
            model.z_init_1(s_inputs)[:, :, None]
            + model.z_init_2(s_inputs)[:, None, :]
        )
        z_init = z_init + model.rel_pos(feats)
        z_init = z_init + model.token_bonds(feats["token_bonds"].float())
        if getattr(model, "bond_type_feature", False):
            z_init = z_init + model.token_bonds_type(feats["type_bonds"].long())
        z_init = z_init + model.contact_conditioning(feats)

        s = torch.zeros_like(s_init)
        z = torch.zeros_like(z_init)
        mask      = feats["token_pad_mask"].float()
        pair_mask = mask[:, :, None] * mask[:, None, :]

        msa_mod = (
            model.msa_module._orig_mod  # noqa: SLF001
            if getattr(model, "is_msa_compiled", False) and not model.training
            else model.msa_module
        )
        pf_mod = (
            model.pairformer_module._orig_mod  # noqa: SLF001
            if getattr(model, "is_pairformer_compiled", False) and not model.training
            else model.pairformer_module
        )

        for _ in range(recycling_steps + 1):
            s = s_init + model.s_recycle(model.s_norm(s))
            z = z_init + model.z_recycle(model.z_norm(z))

            # MSA module (use_kernels=False to avoid custom CUDA kernels)
            z = z + msa_mod(z, s_inputs, feats, use_kernels=False)

            # Pairformer trunk (hooks fire here)
            s, z = pf_mod(s, z, mask=mask, pair_mask=pair_mask, use_kernels=False)

# ============================================================================
# Hook infrastructure
# ============================================================================

def make_baseline_hook(
    layer_idx: int,
    baseline_zs: Dict[int, torch.Tensor],
):
    """Capture-only hook for the BF16 baseline pass (no output modification)."""
    def hook(module: nn.Module, inputs: Any, output: Any) -> None:
        if isinstance(output, (tuple, list)) and len(output) >= 2:
            z_out = output[1]
            if isinstance(z_out, torch.Tensor):
                baseline_zs[layer_idx] = z_out.detach().clone()
    return hook


def make_quant_hook(
    layer_idx: int,
    condition: str,
    baseline_zs: Dict[int, torch.Tensor],
    metrics_store: List[Dict],
    use_native_fp8: bool,
):
    """Quantizing hook for condition passes.

    Applies the appropriate FP8 quantization to z_out, computes accuracy
    metrics vs the stored baseline, and returns the modified (s, z) tuple
    so the quantized z propagates to the next layer.
    """
    def hook(module: nn.Module, inputs: Any, output: Any):
        if not isinstance(output, (tuple, list)) or len(output) < 2:
            return None
        s_out = output[0]
        z_out = output[1]
        if not isinstance(z_out, torch.Tensor) or z_out.ndim != 4:
            return None

        # Build the keep-in-BF16 mask for this condition
        mask = get_mask_for_condition(condition, z_out.shape, z_out.device)

        # Quantize z
        z_q = fp8_quantize(z_out, mask_keep_bf16=mask, use_native=use_native_fp8)

        # Compute metrics vs BF16 baseline
        if layer_idx in baseline_zs:
            z_base = baseline_zs[layer_idx]
            if z_base.shape == z_q.shape:
                m = compute_metrics(z_base, z_q)
                m["layer"] = layer_idx
                metrics_store.append(m)

        # Return modified output — PyTorch replaces the module output with this
        return (s_out, z_q)

    return hook


def _get_pairformer_layers(model: Boltz2):
    """Return the raw (uncompiled) PairformerModule."""
    if getattr(model, "is_pairformer_compiled", False):
        return model.pairformer_module._orig_mod  # noqa: SLF001
    return model.pairformer_module


def attach_baseline_hooks(
    pairformer_mod: nn.Module,
    baseline_zs: Dict[int, torch.Tensor],
) -> List:
    handles = []
    for layer_idx, layer in enumerate(pairformer_mod.layers):
        h = layer.register_forward_hook(make_baseline_hook(layer_idx, baseline_zs))
        handles.append(h)
    return handles


def attach_quant_hooks(
    pairformer_mod: nn.Module,
    condition: str,
    baseline_zs: Dict[int, torch.Tensor],
    metrics_store: List[Dict],
    use_native_fp8: bool,
) -> List:
    handles = []
    for layer_idx, layer in enumerate(pairformer_mod.layers):
        h = layer.register_forward_hook(
            make_quant_hook(
                layer_idx, condition, baseline_zs, metrics_store, use_native_fp8
            )
        )
        handles.append(h)
    return handles


def remove_hooks(handles: List) -> None:
    for h in handles:
        h.remove()

# ============================================================================
# JSON serialisation helper
# ============================================================================

def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        if obj != obj:
            return "NaN"
        if obj == float("inf"):
            return "Inf"
        if obj == float("-inf"):
            return "-Inf"
        return obj
    return obj

# ============================================================================
# Plotting — four publication-quality figures
# ============================================================================

def _setup_mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        return plt, np
    except ImportError:
        return None, None


def generate_plots(
    per_layer_metrics: Dict[str, List[Dict]],
    N: int,
    out_dir: Path,
) -> None:
    plt, np = _setup_mpl()
    if plt is None:
        print("matplotlib not available — skipping plots.")
        return

    num_layers = max(
        (r["layer"] for cond_list in per_layer_metrics.values() for r in cond_list),
        default=63,
    ) + 1
    layer_idx = list(range(num_layers))

    def get_arr(cond: str, key: str) -> List[float]:
        recs = {r["layer"]: r for r in per_layer_metrics.get(cond, [])}
        return [recs.get(i, {}).get(key, float("nan")) for i in layer_idx]

    # ------------------------------------------------------------------
    # Plot 1 — per-layer MAE, log scale  (the headline plot)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 6))
    for cond in CONDITIONS:
        vals = get_arr(cond, "mean_abs_error")
        # filter NaN for semilogy
        ys = [v if v == v and v > 0 else None for v in vals]
        ax.semilogy(
            layer_idx, vals,
            color=CONDITION_COLORS[cond],
            label=CONDITION_LABELS[cond],
            linewidth=2.0,
        )
    ax.set_xlabel("Layer index", fontsize=13)
    ax.set_ylabel("Mean absolute error vs BF16 baseline (log scale)", fontsize=12)
    ax.set_title(
        f"Per-layer mean absolute error vs BF16 baseline\n"
        f"(Boltz-2 Pairformer, N={N})",
        fontsize=13,
    )
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim(0, num_layers - 1)
    fig.tight_layout()
    p1 = out_dir / "plot1_per_layer_mae.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"Saved {p1}")

    # ------------------------------------------------------------------
    # Plot 2 — per-layer cosine similarity, linear
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1.2, label="Perfect (1.0)")
    for cond in CONDITIONS:
        vals = get_arr(cond, "cosine_sim")
        ax.plot(
            layer_idx, vals,
            color=CONDITION_COLORS[cond],
            label=CONDITION_LABELS[cond],
            linewidth=2.0,
        )
    ax.set_xlabel("Layer index", fontsize=13)
    ax.set_ylabel("Cosine similarity vs BF16 baseline", fontsize=12)
    ax.set_title("Per-layer cosine similarity vs BF16 baseline", fontsize=13)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, num_layers - 1)

    # Zoom y-axis to [min - 0.005, 1.001]
    all_vals = [
        v
        for cond in CONDITIONS
        for v in get_arr(cond, "cosine_sim")
        if v == v  # skip NaN
    ]
    if all_vals:
        ax.set_ylim(max(min(all_vals) - 0.005, 0.0), 1.001)

    fig.tight_layout()
    p2 = out_dir / "plot2_per_layer_cosine.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"Saved {p2}")

    # ------------------------------------------------------------------
    # Plot 3 — final-layer ablation bar chart
    # ------------------------------------------------------------------
    last_layer = num_layers - 1
    metric_keys   = ["mean_abs_error", "max_abs_error", "rms_error", "1-cosine_sim"]
    metric_labels = ["Mean Abs Error", "Max Abs Error", "RMS Error", "1 − Cosine Sim"]

    final: Dict[str, Dict[str, float]] = {}
    for cond in CONDITIONS:
        recs = {r["layer"]: r for r in per_layer_metrics.get(cond, [])}
        rec  = recs.get(last_layer, {})
        final[cond] = {
            "mean_abs_error": rec.get("mean_abs_error", float("nan")),
            "max_abs_error":  rec.get("max_abs_error",  float("nan")),
            "rms_error":      rec.get("rms_error",      float("nan")),
            "1-cosine_sim":   1.0 - rec.get("cosine_sim", float("nan")),
        }

    x      = np.arange(len(metric_keys))
    width  = 0.19
    fig, ax = plt.subplots(figsize=(14, 7))
    for i, cond in enumerate(CONDITIONS):
        vals = [final[cond].get(mk, float("nan")) for mk in metric_keys]
        ax.bar(
            x + i * width,
            vals,
            width,
            label=CONDITION_LABELS[cond],
            color=CONDITION_COLORS[cond],
            alpha=0.85,
        )
    ax.set_xlabel("Metric", fontsize=13)
    ax.set_ylabel("Value (log scale)", fontsize=12)
    ax.set_title("Final layer accuracy comparison (lower is better)", fontsize=13)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y", which="both")
    fig.tight_layout()
    p3 = out_dir / "plot3_ablation.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    print(f"Saved {p3}")

    # ------------------------------------------------------------------
    # Plot 4 — memory / compute / accuracy summary table
    # ------------------------------------------------------------------
    D = 128
    total_elem = N * N * D

    # Memory estimates (bytes then to MB)
    def _mb(bf16_n: int, fp8_n: int) -> float:
        return (bf16_n * 2 + fp8_n * 1) / 1e6

    # Element counts for each condition
    ch_bf16  = 4 * N * N
    diag_bf16 = N * D
    ds_bf16  = 4 * N * N + N * D - 4 * N   # channels ∪ diagonal (subtract overlap)

    mem_data = {
        "BF16 baseline":  _mb(total_elem, 0),
        "Naive FP8":      _mb(0, total_elem),
        "Channel-only":   _mb(ch_bf16,   total_elem - ch_bf16),
        "Diagonal-only":  _mb(diag_bf16, total_elem - diag_bf16),
        "Doubly-sparse":  _mb(ds_bf16,   total_elem - ds_bf16),
    }

    def _flops(bf16_n: int, fp8_n: int) -> str:
        frac_bf16 = bf16_n / total_elem
        frac_fp8  = fp8_n  / total_elem
        r = 1.0 / (frac_bf16 + frac_fp8 / 2.0)
        return f"{r:.2f}×"

    flops_data = {
        "BF16 baseline":  "1.00×",
        "Naive FP8":      "~2.00×",
        "Channel-only":   _flops(ch_bf16,   total_elem - ch_bf16),
        "Diagonal-only":  _flops(diag_bf16, total_elem - diag_bf16),
        "Doubly-sparse":  f"~{_flops(ds_bf16, total_elem - ds_bf16)}",
    }

    def _cos(cond_key: Optional[str]) -> str:
        if cond_key is None:
            return "1.0000"
        recs = {r["layer"]: r for r in per_layer_metrics.get(cond_key, [])}
        v = recs.get(last_layer, {}).get("cosine_sim", float("nan"))
        return f"{v:.4f}" if v == v else "N/A"

    cos_data = {
        "BF16 baseline":  "1.0000",
        "Naive FP8":      _cos("naive_fp8"),
        "Channel-only":   _cos("channel_only"),
        "Diagonal-only":  _cos("diagonal_only"),
        "Doubly-sparse":  _cos("doubly_sparse"),
    }

    rows_display = list(mem_data.keys())
    table_data = [
        [name, f"{mem_data[name]:.2f}", flops_data[name], cos_data[name]]
        for name in rows_display
    ]
    col_labels = ["Condition", "Memory / layer (MB)", "FLOPs vs BF16", "Final cosine sim"]

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axis("off")
    tbl = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.2, 2.0)

    # Highlight the doubly-sparse row (index 5 because row 0 is the header)
    highlight_row = rows_display.index("Doubly-sparse") + 1
    for j in range(len(col_labels)):
        tbl[(highlight_row, j)].set_facecolor("#d4edda")

    ax.set_title(
        f"Memory and Compute Summary — Boltz-2 Pairformer FP8 Simulation  (N={N}, D={D})",
        fontsize=13,
        pad=24,
    )
    fig.tight_layout()
    p4 = out_dir / "plot4_summary_table.png"
    fig.savefig(p4, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {p4}")

# ============================================================================
# Console summary
# ============================================================================

def print_summary(
    per_layer_metrics: Dict[str, List[Dict]],
    N: int,
    D: int = 128,
) -> None:
    num_layers = max(
        (r["layer"] for cond_list in per_layer_metrics.values() for r in cond_list),
        default=63,
    ) + 1
    last_layer = num_layers - 1
    total_elem = N * N * D

    def _mae(cond: str, lyr: int) -> float:
        recs = {r["layer"]: r for r in per_layer_metrics.get(cond, [])}
        return recs.get(lyr, {}).get("mean_abs_error", float("nan"))

    def _cos(cond: str) -> float:
        recs = {r["layer"]: r for r in per_layer_metrics.get(cond, [])}
        return recs.get(last_layer, {}).get("cosine_sim", float("nan"))

    print()
    print("=" * 68)
    print(f"FP8 SIMULATION RESULTS — Boltz-2 N={N}, H100")
    print()
    print("Per-layer error growth (mean absolute error):")
    header_cols = ["Layer 0", "Layer 16", "Layer 32", "Layer 48", "Layer 63"]
    print(f"{'':22s}" + "".join(f"{c:>12s}" for c in header_cols))
    for cond in CONDITIONS:
        label = CONDITION_LABELS[cond]
        vals  = [_mae(cond, l) for l in REPORT_LAYERS]
        row   = f"{label:<22s}"
        for v in vals:
            if v == v:
                row += f"{v:>12.3f}"
            else:
                row += f"{'N/A':>12s}"
        print(row)

    print()
    print("Final layer cosine similarity vs BF16 baseline:")
    cosine_vals = {cond: _cos(cond) for cond in CONDITIONS}
    best_cond = max(cosine_vals, key=lambda c: cosine_vals[c] if cosine_vals[c] == cosine_vals[c] else -1.0)
    for cond in CONDITIONS:
        cosine = cosine_vals[cond]
        suffix = "  ← best structured condition" if cond == best_cond else ""
        if cosine == cosine:
            print(f"  {CONDITION_LABELS[cond]:<22s} {cosine:.4f}{suffix}")
        else:
            print(f"  {CONDITION_LABELS[cond]:<22s} N/A{suffix}")

    print()
    print("Memory and compute summary:")
    bf16_mb = total_elem * 2 / 1e6
    naive_mb = total_elem * 1 / 1e6
    ds_bf16 = 4 * N * N + N * D - 4 * N
    ds_mb = (ds_bf16 * 2 + (total_elem - ds_bf16) * 1) / 1e6
    print(
        f"  {'BF16 baseline:':<24s} {bf16_mb:5.1f} MB/layer"
        f"  1.00x FLOPs  cosine=1.0000"
    )
    c_naive = _cos("naive_fp8")
    print(
        f"  {'Naive FP8:':<24s} {naive_mb:5.1f} MB/layer"
        f"  2.00x FLOPs  cosine={c_naive:.4f}" if c_naive == c_naive
        else f"  {'Naive FP8:':<24s} {naive_mb:5.1f} MB/layer  2.00x FLOPs  cosine=N/A"
    )
    c_ds = _cos("doubly_sparse")
    print(
        f"  {'Doubly-sparse:':<24s} {ds_mb:5.1f} MB/layer"
        f"  ~2.00x FLOPs  cosine={c_ds:.4f}" if c_ds == c_ds
        else f"  {'Doubly-sparse:':<24s} {ds_mb:5.1f} MB/layer  ~2.00x FLOPs  cosine=N/A"
    )

    print()
    print("Verdict:")

    # Guard against NaN results
    if c_naive != c_naive or c_ds != c_ds:
        print("  WARNING: NaN values in results — something may have gone wrong.")
        print("=" * 68)
        return

    # Unexpected: naive FP8 is very accurate
    if c_naive > 0.9999:
        print(
            "  WARNING: Naive FP8 unexpectedly preserves accuracy (cosine > 0.9999)."
        )
        print(
            "  The original BF16 activations may already be small enough to fit"
            " within the FP8 E4M3 range (~448 max), meaning no significant clipping"
            " occurs. Consider profiling per-layer abs_max values to confirm."
        )
        print("=" * 68)
        return

    err_naive = 1.0 - c_naive
    err_ds    = 1.0 - c_ds

    imp = err_naive / err_ds if err_ds > 0 else float("inf")
    mem_overhead_pct = (ds_mb / naive_mb - 1.0) * 100.0

    # diagonal_only  = doubly_sparse with CHANNEL protection removed
    # channel_only   = doubly_sparse with DIAGONAL protection removed
    c_diag_only    = _cos("diagonal_only")   # kept diag, dropped channels
    c_channel_only = _cos("channel_only")    # kept channels, dropped diag
    err_diag_only    = (1.0 - c_diag_only)    if c_diag_only    == c_diag_only    else float("nan")
    err_channel_only = (1.0 - c_channel_only) if c_channel_only == c_channel_only else float("nan")

    # Factor: how much worse is "removed channel protection" vs doubly-sparse?
    remove_ch_factor  = err_diag_only    / err_ds if err_ds > 0 and err_diag_only    == err_diag_only    else float("nan")
    # Factor: how much worse (or better) is "removed diagonal protection" vs doubly-sparse?
    remove_diag_factor = err_channel_only / err_ds if err_ds > 0 and err_channel_only == err_channel_only else float("nan")

    if c_ds > c_naive:
        print(
            f"  Doubly-sparse achieves {imp:.1f}x better accuracy than naive FP8"
        )
        print(
            f"  with only {mem_overhead_pct:+.1f}% memory overhead vs naive FP8."
        )
        if remove_ch_factor == remove_ch_factor:
            print(
                f"  Ablation — removing channel sparsity (→ diagonal-only):"
                f"  {remove_ch_factor:.1f}× error growth."
            )
        if remove_diag_factor == remove_diag_factor:
            if remove_diag_factor >= 1.0:
                print(
                    f"  Ablation — removing diagonal sparsity (→ channel-only):"
                    f"  {remove_diag_factor:.1f}× error growth."
                )
            else:
                print(
                    f"  Ablation — removing diagonal sparsity (→ channel-only) is"
                    f" slightly BETTER ({remove_diag_factor:.2f}× error):"
                )
                print(
                    "  Non-loud-channel diagonal values fit in FP8 range;"
                    " channel protection alone is the primary driver."
                )
    else:
        print(
            f"  NOTE: Doubly-sparse did NOT outperform naive FP8 in this run."
        )
        print(
            f"  Doubly-sparse cosine: {c_ds:.4f},  Naive FP8 cosine: {c_naive:.4f}"
        )
        print(
            "  Possible explanations:"
        )
        print(
            "  • LOUD_CHANNELS may not be the dominant outlier channels for this input."
        )
        print(
            "  • Outlier magnitudes may not exceed FP8_MAX=448 in this configuration."
        )
        print(
            "  • Try profiling with boltz2_verification.py to check activation stats."
        )

    print("=" * 68)

# ============================================================================
# Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FP8 mixed-precision simulation on Boltz-2 Pairformer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--processed_dir",
        type=str,
        default="test_output_boltz2/medium_386aa/processed",
        help="Path to preprocessed data directory (created if absent)",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default=None,
        help="Boltz cache directory (default: ~/.boltz or $BOLTZ_CACHE)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional path to a Boltz-2 .ckpt file",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./fp8_results",
        help="Output directory for JSON and PNG files",
    )
    parser.add_argument(
        "--use_msa_server",
        action="store_true",
        help="Use the MMSeqs2 server for MSA generation if preprocessing is needed",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU execution (very slow, for debugging only)",
    )
    parser.add_argument(
        "--no_bf16",
        action="store_true",
        help="Disable BF16 autocast (run in FP32)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--recycling_steps",
        type=int,
        default=0,
        help="Number of recycling steps (0 = single pass, faster for experiments)",
    )
    args = parser.parse_args()

    # ── Reproducibility ──────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("highest")
    warnings.filterwarnings("ignore", ".*Tensor Cores.*")

    # ── Device ───────────────────────────────────────────────────────────────
    if args.cpu or not torch.cuda.is_available():
        device  = torch.device("cpu")
        use_bf16 = False
        print("Running on CPU (no GPU or --cpu flag set). This will be slow.")
    else:
        device   = torch.device("cuda")
        use_bf16 = not args.no_bf16
        props    = torch.cuda.get_device_properties(device)
        print(f"GPU: {props.name}  ({props.total_memory / 1e9:.0f} GB VRAM)")
        print(f"BF16 autocast: {use_bf16}")

    use_native_fp8 = HAS_NATIVE_FP8 and device.type == "cuda"
    if use_native_fp8:
        print("FP8 mode: native torch.float8_e4m3fn")
    else:
        print("FP8 mode: BF16 simulation of E4M3 quantization")

    # ── Paths ────────────────────────────────────────────────────────────────
    cache_root = Path(args.cache).expanduser() if args.cache else Path(get_cache_path()).expanduser()
    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    processed_dir_base = Path(args.processed_dir)
    if processed_dir_base.name == "processed":
        work_dir      = processed_dir_base.parent
        processed_dir = processed_dir_base
    else:
        work_dir      = processed_dir_base
        processed_dir = processed_dir_base / "processed"

    # ── Download weights ─────────────────────────────────────────────────────
    print(f"\nEnsuring Boltz-2 weights are in {cache_root} …")
    download_boltz2(cache_root)

    # ── Preprocess input ─────────────────────────────────────────────────────
    print("\nChecking preprocessed data …")
    processed_dir = ensure_processed_data(
        processed_dir=processed_dir,
        cache=cache_root,
        work_dir=work_dir,
        use_msa_server=args.use_msa_server,
    )

    # ── Load model ────────────────────────────────────────────────────────────
    print("\nLoading model …")
    t0    = time.time()
    model = load_boltz2(cache_root, args.checkpoint)
    model = model.to(device)
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\nLoading data batch …")
    feats = load_batch(processed_dir, cache_root, device)
    N     = int(feats["token_pad_mask"].sum().item())
    D     = 128
    print(f"  Sequence length N = {N}")
    print(f"  Pair tensor shape: [1, {N}, {N}, {D}]")
    print(f"  BF16 storage per layer: {N * N * D * 2 / 1e6:.1f} MB")
    print(f"  Baseline total (64 layers): {N * N * D * 2 * 64 / 1e6:.0f} MB")

    pf_mod    = _get_pairformer_layers(model)
    num_layers = len(list(pf_mod.layers))
    print(f"  Pairformer has {num_layers} layers")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1 — BF16 baseline: capture z at every layer, no modification
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 56}")
    print("PHASE 1: BF16 baseline forward pass …")
    baseline_zs: Dict[int, torch.Tensor] = {}
    handles = attach_baseline_hooks(pf_mod, baseline_zs)

    try:
        t0 = time.time()
        run_trunk_forward(model, feats, device, use_bf16, args.recycling_steps)
        elapsed = time.time() - t0
        print(f"  Captured {len(baseline_zs)} layer z tensors in {elapsed:.1f}s")
    except torch.cuda.OutOfMemoryError:
        remove_hooks(handles)
        raise RuntimeError(
            "OOM during baseline pass. Reduce sequence length or use --cpu."
        )
    except Exception as exc:
        remove_hooks(handles)
        traceback.print_exc()
        raise RuntimeError(f"Baseline pass failed: {exc}") from exc
    finally:
        remove_hooks(handles)

    if not baseline_zs:
        raise RuntimeError(
            "No baseline z tensors captured.\n"
            "Hooks did not fire — ensure use_kernels=False and that the model "
            "contains PairformerLayer instances."
        )

    # Sanity-check baseline
    z0     = baseline_zs[0]
    z0_max = z0.abs().max().item()
    print(f"  Layer 0 baseline: abs_max={z0_max:.3f}  dtype={z0.dtype}  shape={tuple(z0.shape)}")
    if z0_max > 10_000:
        print(
            f"  WARNING: Very large layer-0 z values ({z0_max:.1f}). "
            "Naive FP8 will likely show large errors."
        )
    if z0_max < 1e-6:
        print(
            "  WARNING: Near-zero layer-0 z values — hooks may not be firing "
            "correctly or the embeddings collapsed."
        )

    # ══════════════════════════════════════════════════════════════════════════
    # PHASES 2-5 — one forward pass per quantization condition
    # ══════════════════════════════════════════════════════════════════════════
    per_layer_metrics: Dict[str, List[Dict]] = {}

    for condition in CONDITIONS:
        print(f"\n{'═' * 56}")
        print(f"Running condition: {CONDITION_LABELS[condition]} …")
        metrics_store: List[Dict] = []
        handles = attach_quant_hooks(
            pf_mod, condition, baseline_zs, metrics_store, use_native_fp8
        )

        try:
            t0 = time.time()
            run_trunk_forward(model, feats, device, use_bf16, args.recycling_steps)
            elapsed = time.time() - t0
            metrics_store.sort(key=lambda r: r["layer"])

            n_captured  = len(metrics_store)
            final_cos   = metrics_store[-1]["cosine_sim"] if metrics_store else float("nan")
            final_mae   = metrics_store[-1]["mean_abs_error"] if metrics_store else float("nan")
            print(
                f"  Done in {elapsed:.1f}s | {n_captured} layer metrics | "
                f"final cosine={final_cos:.4f} | final MAE={final_mae:.4e}"
            )

        except torch.cuda.OutOfMemoryError:
            print(f"  OOM during {condition} — skipping.", file=sys.stderr)
            metrics_store = []
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            traceback.print_exc()
            metrics_store = []
        finally:
            remove_hooks(handles)

        per_layer_metrics[condition] = metrics_store

        # Free GPU cache between conditions
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════════════
    # Save JSON
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 56}")
    print("Saving results …")

    last_layer = num_layers - 1

    def _final_summary(cond: str) -> Dict:
        recs = {r["layer"]: r for r in per_layer_metrics.get(cond, [])}
        rec  = recs.get(last_layer, {})
        return {k: v for k, v in rec.items() if k != "layer"}

    results = {
        "config": {
            "model":               "boltz2",
            "N":                   N,
            "D":                   D,
            "loud_channels":       LOUD_CHANNELS,
            "fp8_max":             FP8_MAX,
            "fp8_min_normal":      FP8_MIN_NORMAL,
            "native_fp8":          use_native_fp8,
            "bf16_autocast":       use_bf16,
            "num_layers":          num_layers,
            "recycling_steps":     args.recycling_steps,
            "device":              str(device),
            "seed":                args.seed,
        },
        "per_layer_metrics": {
            cond: [
                {"layer": r["layer"], **{k: v for k, v in r.items() if k != "layer"}}
                for r in per_layer_metrics[cond]
            ]
            for cond in CONDITIONS
        },
        "final_layer_summary": {cond: _final_summary(cond) for cond in CONDITIONS},
    }

    json_path = out_dir / "fp8_simulation_results.json"
    with open(json_path, "w") as fh:
        json.dump(_json_safe(results), fh, indent=2)
    print(f"Saved {json_path}")

    # ══════════════════════════════════════════════════════════════════════════
    # Generate plots
    # ══════════════════════════════════════════════════════════════════════════
    print("\nGenerating plots …")
    generate_plots(per_layer_metrics, N, out_dir)

    # ══════════════════════════════════════════════════════════════════════════
    # Console summary
    # ══════════════════════════════════════════════════════════════════════════
    print_summary(per_layer_metrics, N, D)

    print(f"\nAll outputs saved to: {out_dir.resolve()}/")


if __name__ == "__main__":
    main()
