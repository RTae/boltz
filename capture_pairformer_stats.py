"""
capture_pairformer_stats.py
===========================
On-the-fly numerical distribution characteriser for Boltz-2's Pairformer module.

Goal: identify activation/weight outlier patterns that affect MXFP8 mixed-precision
training and inference.  No full tensors are saved to disk; only small float statistics
are accumulated per hook fire and written as JSON-lines.

Key MXFP8 concerns captured here:
  - abs_max / kurtosis: heavy-tailed distributions blow up FP8 E4M3 (max ~448)
    and E5M2 (max ~57344) representations.
  - fraction_gt6 / fraction_gt57344: direct overflow-risk counters.
  - channel_max_to_median_ratio: SmoothQuant/RoMeo-style per-channel outlier
    indicator; ratio >> 1 means smooth-quant channel migration is needed.
  - symmetry_score: pair matrix asymmetry; MXFP8 quantisers may exploit symmetry
    assumptions that break here.
  - pre_softmax_score: the logit distribution before softmax.  High kurtosis or
    large magnitude scores indicate attention sharpening that hurts low-precision.

Operator inventory (from Phase-1 discovery):
  Pairformer blocks (64 × PairformerLayer):
    - TriangleMultiplicationOutgoing  (pair, z-track)
    - TriangleMultiplicationIncoming  (pair, z-track)
    - TriangleAttentionStartingNode   (pair, z-track) + internal pre-softmax
    - TriangleAttentionEndingNode     (pair, z-track) + internal pre-softmax
    - AttentionPairBias [v2]          (single, s-track) + internal pre-softmax
    - triangle_bias linear inside each TriangleAttention (separate capture)
  MSA blocks (4 × MSALayer, each containing):
    - OuterProductMean
    - PairWeightedAveraging
    - PairformerNoSeqLayer (inherits same triangle ops)
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import pickle
import platform
import tarfile
import traceback
import urllib.request
import warnings
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger("capture_stats")

# ---------------------------------------------------------------------------
# Record dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ActivationRecord:
    """One hook-fire record written as a JSONL line.

    Fields are chosen to answer three MXFP8 questions:
      1. Range overflow risk   → abs_max, fraction_gt6, fraction_gt57344
      2. Distribution shape    → kurtosis, percentiles
      3. Channel outlier risk  → channel_max_to_median_ratio (SmoothQuant indicator)
      4. Pair symmetry         → symmetry_score, symmetric_outlier_correlation
    """

    # Identity
    layer_index: int          # Pairformer block index (0-based); -1 for MSA layers
    msa_layer_index: int      # MSA block index (0-based); -1 for Pairformer layers
    operator_name: str        # Class name of the operator
    tensor_role: str          # "input" | "output" | "pre_softmax_score" | "tri_bias_output"
    seqlen: int               # N (token sequence length used as x-axis variable)

    # Shape / dtype
    tensor_shape: List[int]
    dtype: str

    # Global stats
    min: float
    max: float
    mean: float
    std: float
    abs_max: float
    kurtosis: float           # Fisher definition: E[((x-mu)/sigma)^4] - 3

    # Percentiles (absolute values)
    p50: float
    p90: float
    p99: float
    p99_9: float
    p99_99: float

    # MXFP8 range counters
    fraction_gt6: float       # E4M3 dynamic-range concern: |x| > 6
    fraction_gt57344: float   # E5M2 dynamic-range concern: |x| > 57344

    # Channel-wise stats (along last dim D)
    channel_max_to_median_ratio: float   # max(per-channel abs_max) / median(per-channel abs_max)

    # Token-pair extremes (only for [B, N, N, D] tensors, else empty)
    top_k_abs_values: List[float]
    top_k_indices: List[List[int]]       # each entry: [i, j, d]

    # Symmetry (only for [B, N, N, D] tensors, else -1.0)
    symmetry_score: float     # mean |z[i,j,d]-z[j,i,d]| / (|z[i,j,d]|+|z[j,i,d]|+1e-6)
    symmetry_max: float
    symmetric_outlier_correlation: float # fraction of top-1000 that have (j,i) also in top-1000

    # Diagonal vs off-diagonal (only for [B, N, N, D] tensors, else -1.0)
    diag_abs_max: float
    diag_mean: float
    diag_std: float
    offdiag_abs_max: float
    offdiag_mean: float
    offdiag_std: float
    diag_to_offdiag_ratio: float


@dataclass
class WeightRecord:
    """One weight-analysis record written as a JSONL line.

    Per-output and per-input channel stats expose anisotropic weight distributions
    that cause quantisation error under naive per-tensor MXFP8 scaling.
    """

    layer_index: int
    msa_layer_index: int
    operator_name: str
    parameter_name: str       # e.g. "tri_mul_out.p_in.weight"

    shape: List[int]
    dtype: str

    # Global
    min: float
    max: float
    mean: float
    std: float
    abs_max: float
    kurtosis: float

    # Per-output-channel (dim 0) abs-max list — for large weights summarised as quantiles
    out_channel_abs_max_p50: float
    out_channel_abs_max_p90: float
    out_channel_abs_max_p99: float
    out_channel_abs_max_max: float
    channel_max_to_median_ratio: float   # max / median of per-output-channel abs_max


# ---------------------------------------------------------------------------
# GPU statistics helpers (all under torch.no_grad())
# ---------------------------------------------------------------------------

@torch.no_grad()
def _kurtosis(x: Tensor) -> float:
    """Fisher kurtosis (excess): E[((x-mu)/sigma)^4] - 3.

    Values > 0 indicate heavier tails than Gaussian — a sign that naive
    per-tensor MXFP8 clipping will incur significant error.
    """
    x = x.float().flatten()
    if x.numel() < 4:
        return 0.0
    mu = x.mean()
    sigma = x.std(unbiased=False)
    if sigma < 1e-8:
        return 0.0
    kurt = ((((x - mu) / sigma) ** 4).mean() - 3.0).item()
    return float(kurt)


@torch.no_grad()
def _compute_tensor_stats(
    t: Tensor,
    top_k: int = 100,
) -> Dict[str, Any]:
    """Compute the full statistics dictionary for a single tensor on GPU.

    Returns a plain Python dict suitable for ActivationRecord construction.
    No tensor is retained after this call.
    """
    t_f = t.float()
    flat = t_f.flatten()
    numel = flat.numel()

    # --- Global stats ---
    t_min  = flat.min().item()
    t_max  = flat.max().item()
    t_mean = flat.mean().item()
    t_std  = flat.std(unbiased=False).item()
    t_abs  = flat.abs()
    abs_max = t_abs.max().item()
    kurt = _kurtosis(flat)

    # Percentiles (of absolute values)
    percs = torch.quantile(
        t_abs,
        torch.tensor([0.50, 0.90, 0.99, 0.999, 0.9999], device=t.device),
    )
    p50, p90, p99, p99_9, p99_99 = percs.tolist()

    # MXFP8 overflow fractions
    frac_gt6     = (t_abs > 6.0).float().mean().item()
    frac_gt57344 = (t_abs > 57344.0).float().mean().item()

    # --- Channel-wise stats (last dimension) ---
    last_dim = t_f.shape[-1] if t_f.ndim >= 1 else 1
    chan_view = t_f.reshape(-1, last_dim)   # [*, D]
    chan_abs_max = chan_view.abs().max(dim=0).values   # [D]
    cmax_median = chan_abs_max.median().item()
    cmax_max    = chan_abs_max.max().item()
    channel_max_to_median_ratio = (
        cmax_max / (cmax_median + 1e-8)
    )
    del chan_view, chan_abs_max

    # --- Token-pair stats (only for [B, N, N, D]) ---
    top_k_abs_values: List[float] = []
    top_k_indices:    List[List[int]] = []

    sym_score   = -1.0
    sym_max     = -1.0
    sym_outlier = -1.0
    diag_abs_max = -1.0
    diag_mean    = -1.0
    diag_std     = -1.0
    offdiag_abs_max = -1.0
    offdiag_mean    = -1.0
    offdiag_std     = -1.0
    diag_to_offdiag = -1.0

    if t_f.ndim == 4 and t_f.shape[1] == t_f.shape[2]:
        B, N, _, D = t_f.shape

        # Top-k over (i, j, d) — take the first batch item to avoid OOM
        t0 = t_f[0]   # [N, N, D]
        flat0 = t0.abs().flatten()
        k_actual = min(top_k, flat0.numel())
        top_vals, top_idx_flat = torch.topk(flat0, k_actual)
        # Convert flat indices back to (i, j, d)
        i_idx = (top_idx_flat // (N * D))
        j_idx = (top_idx_flat % (N * D)) // D
        d_idx = top_idx_flat % D
        top_k_abs_values = top_vals.tolist()
        top_k_indices     = torch.stack([i_idx, j_idx, d_idx], dim=1).tolist()
        del flat0, top_vals, top_idx_flat, i_idx, j_idx, d_idx

        # Symmetry stats
        t0_T = t0.transpose(0, 1)   # z[j,i,d] → z_T[i,j,d]
        diff  = (t0 - t0_T).abs()
        denom = (t0.abs() + t0_T.abs() + 1e-6)
        sym_ratio = diff / denom
        sym_score = sym_ratio.mean().item()
        sym_max   = sym_ratio.max().item()
        del diff, denom, sym_ratio

        # symmetric_outlier_correlation: of top-1000 abs values in t0, what fraction
        # have their (j,i,d) counterpart also in the top-1000?
        K_sym = min(1000, t0.numel())
        flat0_sym = t0.abs().flatten()
        _, sym_top_idx = torch.topk(flat0_sym, K_sym)
        sym_set_flat = set(sym_top_idx.tolist())
        # Compute transposed flat indices for each top index
        i_s = sym_top_idx // (N * D)
        j_s = (sym_top_idx % (N * D)) // D
        d_s = sym_top_idx % D
        transposed_flat = (j_s * N * D + i_s * D + d_s).tolist()
        matches = sum(1 for fi in transposed_flat if fi in sym_set_flat)
        sym_outlier = matches / K_sym
        del flat0_sym, sym_top_idx, sym_set_flat, i_s, j_s, d_s

        # Diagonal vs off-diagonal
        diag_mask = torch.eye(N, dtype=torch.bool, device=t_f.device)  # [N, N]
        t0_abs = t0.abs()
        diag_vals    = t0_abs[diag_mask]           # [N*D] conceptually [N, D] flattened
        offdiag_vals = t0_abs[~diag_mask]
        if diag_vals.numel() > 0:
            diag_abs_max = diag_vals.max().item()
            diag_mean    = diag_vals.mean().item()
            diag_std     = diag_vals.std(unbiased=False).item() if diag_vals.numel() > 1 else 0.0
        if offdiag_vals.numel() > 0:
            offdiag_abs_max = offdiag_vals.max().item()
            offdiag_mean    = offdiag_vals.mean().item()
            offdiag_std     = offdiag_vals.std(unbiased=False).item() if offdiag_vals.numel() > 1 else 0.0
        diag_to_offdiag = diag_abs_max / (offdiag_abs_max + 1e-8)
        del t0_abs, diag_vals, offdiag_vals, diag_mask, t0, t0_T

    del t_f, flat, t_abs

    return dict(
        min=t_min, max=t_max, mean=t_mean, std=t_std, abs_max=abs_max,
        kurtosis=kurt,
        p50=p50, p90=p90, p99=p99, p99_9=p99_9, p99_99=p99_99,
        fraction_gt6=frac_gt6, fraction_gt57344=frac_gt57344,
        channel_max_to_median_ratio=channel_max_to_median_ratio,
        top_k_abs_values=top_k_abs_values,
        top_k_indices=top_k_indices,
        symmetry_score=sym_score, symmetry_max=sym_max,
        symmetric_outlier_correlation=sym_outlier,
        diag_abs_max=diag_abs_max, diag_mean=diag_mean, diag_std=diag_std,
        offdiag_abs_max=offdiag_abs_max, offdiag_mean=offdiag_mean, offdiag_std=offdiag_std,
        diag_to_offdiag_ratio=diag_to_offdiag,
    )


@torch.no_grad()
def _compute_weight_stats(w: Tensor) -> Dict[str, Any]:
    """Compute weight statistics including per-output-channel analysis."""
    w_f = w.float()
    flat = w_f.flatten()

    t_min  = flat.min().item()
    t_max  = flat.max().item()
    t_mean = flat.mean().item()
    t_std  = flat.std(unbiased=False).item()
    abs_max = flat.abs().max().item()
    kurt = _kurtosis(flat)

    # Per-output-channel (dim 0) abs-max
    if w_f.ndim >= 2:
        out_chan_abs = w_f.reshape(w_f.shape[0], -1).abs().max(dim=1).values
    else:
        out_chan_abs = w_f.abs().unsqueeze(0)

    percs = torch.quantile(
        out_chan_abs,
        torch.tensor([0.50, 0.90, 0.99], device=w.device),
    )
    p50_oc, p90_oc, p99_oc = percs.tolist()
    max_oc = out_chan_abs.max().item()
    med_oc = out_chan_abs.median().item()
    ratio  = max_oc / (med_oc + 1e-8)

    del w_f, flat, out_chan_abs
    return dict(
        min=t_min, max=t_max, mean=t_mean, std=t_std, abs_max=abs_max,
        kurtosis=kurt,
        out_channel_abs_max_p50=p50_oc,
        out_channel_abs_max_p90=p90_oc,
        out_channel_abs_max_p99=p99_oc,
        out_channel_abs_max_max=max_oc,
        channel_max_to_median_ratio=ratio,
    )


# ---------------------------------------------------------------------------
# Hook manager
# ---------------------------------------------------------------------------

class HookManager:
    """Registers, maintains and removes PyTorch forward hooks.

    Architecture:
    - forward_hook on every operator module  → capture input & output tensors
    - monkey-patched forward on sub-modules  → capture pre-softmax logits
    - forward_hook on triangle-bias linear   → capture triangle bias output
    """

    def __init__(
        self,
        output_file: Path,
        layers_to_capture: List[int],
        top_k: int = 100,
        seqlen: int = 0,
    ) -> None:
        self.output_file = output_file
        self.layers_to_capture = set(layers_to_capture)
        self.top_k = top_k
        self.seqlen = seqlen
        self._handles: List[torch.utils.hooks.RemovableHook] = []
        self._patched: List[Tuple[nn.Module, str, Callable]] = []  # (mod, 'forward', orig_fn)

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def _write_record(self, record: ActivationRecord) -> None:
        """Append one JSONL record to the output file."""
        try:
            line = json.dumps(asdict(record)) + "\n"
            with self.output_file.open("a") as fh:
                fh.write(line)
        except Exception:
            log.warning("Failed to write record: %s", traceback.format_exc())

    def _build_record(
        self,
        t: Tensor,
        layer_index: int,
        msa_layer_index: int,
        operator_name: str,
        tensor_role: str,
    ) -> Optional[ActivationRecord]:
        try:
            stats = _compute_tensor_stats(t, self.top_k)
            return ActivationRecord(
                layer_index=layer_index,
                msa_layer_index=msa_layer_index,
                operator_name=operator_name,
                tensor_role=tensor_role,
                seqlen=self.seqlen,
                tensor_shape=list(t.shape),
                dtype=str(t.dtype),
                **stats,
            )
        except Exception:
            log.warning(
                "Stats failed for %s/%s layer=%d: %s",
                operator_name, tensor_role, layer_index, traceback.format_exc(),
            )
            return None

    # ------------------------------------------------------------------
    # Hook factories
    # ------------------------------------------------------------------

    def _make_forward_hook(
        self,
        layer_index: int,
        msa_layer_index: int,
        operator_name: str,
    ) -> Callable:
        """Return a hook that captures the first input tensor and the output tensor."""
        def hook(module: nn.Module, inputs: tuple, output: Tensor) -> None:
            try:
                with torch.no_grad():
                    # Input: take first tensor arg
                    for inp in inputs:
                        if isinstance(inp, Tensor):
                            rec = self._build_record(
                                inp, layer_index, msa_layer_index, operator_name, "input"
                            )
                            if rec is not None:
                                self._write_record(rec)
                            break

                    # Output
                    if isinstance(output, Tensor):
                        rec = self._build_record(
                            output, layer_index, msa_layer_index, operator_name, "output"
                        )
                        if rec is not None:
                            self._write_record(rec)
            except Exception:
                log.warning("Hook error %s: %s", operator_name, traceback.format_exc())
        return hook

    def _make_tri_bias_hook(
        self,
        layer_index: int,
        msa_layer_index: int,
        operator_name: str,
    ) -> Callable:
        """Hook that captures the OUTPUT of the triangle-bias linear projection.

        This is the [*, H, I, J] bias tensor added to attention logits.
        Capturing it separately shows whether the pair bias itself has extreme
        values that could dominate (or saturate) the FP8 attention logit range.
        """
        def hook(module: nn.Module, inputs: tuple, output: Tensor) -> None:
            try:
                with torch.no_grad():
                    rec = self._build_record(
                        output, layer_index, msa_layer_index,
                        operator_name, "tri_bias_output"
                    )
                    if rec is not None:
                        self._write_record(rec)
            except Exception:
                log.warning("Tri-bias hook error: %s", traceback.format_exc())
        return hook

    # ------------------------------------------------------------------
    # Monkey-patching helpers for pre-softmax captures
    # ------------------------------------------------------------------

    def _patch_attention_pair_bias_v2(
        self,
        module,          # AttentionPairBiasV2 instance
        layer_index: int,
        msa_layer_index: int,
    ) -> None:
        """Monkey-patch AttentionPairBias (v2) forward to capture pre-softmax `attn`."""
        orig_forward = module.forward
        manager = self
        op_name = type(module).__name__

        def patched_forward(s, z, mask, k_in, multiplicity=1):
            # Replicate the v2 forward, injecting the capture
            B = s.shape[0]
            # ---- CAPTURE INPUT (s tensor) ----
            # The outer forward hook cannot see this because PairformerLayer calls
            # attention(...) with all keyword args, leaving inputs=() in the hook.
            with torch.no_grad():
                rec = manager._build_record(
                    s, layer_index, msa_layer_index, op_name, "input"
                )
                if rec is not None:
                    manager._write_record(rec)
            q = module.proj_q(s).view(B, -1, module.num_heads, module.head_dim)
            k = module.proj_k(k_in).view(B, -1, module.num_heads, module.head_dim)
            v = module.proj_v(k_in).view(B, -1, module.num_heads, module.head_dim)

            bias = module.proj_z(z)
            bias = bias.repeat_interleave(multiplicity, 0)
            g = module.proj_g(s).sigmoid()

            with torch.autocast("cuda", enabled=False):
                attn = torch.einsum("bihd,bjhd->bhij", q.float(), k.float())
                attn = attn / (module.head_dim ** 0.5) + bias.float()
                attn = attn + (1 - mask[:, None, None].float()) * -module.inf
                # ---- CAPTURE PRE-SOFTMAX ----
                with torch.no_grad():
                    rec = manager._build_record(
                        attn, layer_index, msa_layer_index, op_name, "pre_softmax_score"
                    )
                    if rec is not None:
                        manager._write_record(rec)
                # ---- END CAPTURE ----
                attn = attn.softmax(dim=-1)
                o = torch.einsum("bhij,bjhd->bihd", attn, v.float()).to(v.dtype)
            o = o.reshape(B, -1, module.c_s)
            o = module.proj_o(g * o)
            return o

        module.forward = patched_forward
        self._patched.append((module, "forward", orig_forward))

    def _patch_triangle_attention_mha(
        self,
        mha_module,       # Attention instance (primitives.py)
        layer_index: int,
        msa_layer_index: int,
        operator_name: str,
    ) -> None:
        """Monkey-patch the inner Attention.forward (primitives.py) to capture
        the pre-softmax score tensor `a` (shape [*, H, Q, K]).

        This is inserted at the point where biases are summed but before softmax.
        We replicate the logic of _attention() and Attention.forward() here.
        """
        import math as _math
        from boltz.model.layers.triangular_attention.primitives import softmax_no_cast
        from boltz.model.layers.triangular_attention.utils import (
            flatten_final_dims,
            permute_final_dims,
        )
        orig_forward = mha_module.forward
        manager = self
        mha = mha_module  # reference to avoid closure issues

        def patched_forward(q_x, kv_x, tri_bias, mask_bias, mask, use_kernels=False):
            if use_kernels:
                # Can't intercept kernel path cleanly; fall through to original
                return orig_forward(q_x, kv_x, tri_bias, mask_bias, mask, use_kernels)

            # Replicate Attention.forward + _attention with capture
            q, k, v = mha._prep_qkv(q_x, kv_x, apply_scale=True)

            # _attention inline
            key_T = permute_final_dims(k, (1, 0))
            a = torch.matmul(q, key_T)
            a = a + mask_bias
            a = a + tri_bias
            # ---- CAPTURE PRE-SOFTMAX ----
            with torch.no_grad():
                rec = manager._build_record(
                    a, layer_index, msa_layer_index, operator_name, "pre_softmax_score"
                )
                if rec is not None:
                    manager._write_record(rec)
            # ---- END CAPTURE ----
            a = softmax_no_cast(a, -1)
            a = torch.matmul(a, v)
            o = a.transpose(-2, -3)
            o = mha._wrap_up(o, q_x)
            return o

        mha_module.forward = patched_forward
        self._patched.append((mha_module, "forward", orig_forward))

    # ------------------------------------------------------------------
    # Registration entry points
    # ------------------------------------------------------------------

    def register_pairformer_layer(
        self,
        layer: nn.Module,
        layer_index: int,
    ) -> None:
        """Register hooks on all sub-modules of a single PairformerLayer."""
        from boltz.model.layers.triangular_mult import (
            TriangleMultiplicationIncoming,
            TriangleMultiplicationOutgoing,
        )
        from boltz.model.layers.triangular_attention.attention import TriangleAttention
        from boltz.model.layers.attentionv2 import AttentionPairBias as AttentionPairBiasV2

        if layer_index not in self.layers_to_capture:
            return

        li = layer_index
        msa_li = -1

        # Triangle multiplication operators
        for attr, cls in [
            ("tri_mul_out", TriangleMultiplicationOutgoing),
            ("tri_mul_in", TriangleMultiplicationIncoming),
        ]:
            submod = getattr(layer, attr, None)
            if submod is not None and isinstance(submod, cls):
                h = submod.register_forward_hook(
                    self._make_forward_hook(li, msa_li, type(submod).__name__)
                )
                self._handles.append(h)

        # Triangle attention operators + inner mha + triangle bias linear
        for attr in ("tri_att_start", "tri_att_end"):
            ta = getattr(layer, attr, None)
            if ta is not None and isinstance(ta, TriangleAttention):
                op_name = type(ta).__name__
                # Outer hook (input/output of the full TriangleAttention)
                h = ta.register_forward_hook(
                    self._make_forward_hook(li, msa_li, op_name)
                )
                self._handles.append(h)

                # Triangle-bias linear hook (captures [*, H, I, J] bias values)
                bias_linear = getattr(ta, "linear", None)
                if bias_linear is not None:
                    h2 = bias_linear.register_forward_hook(
                        self._make_tri_bias_hook(li, msa_li, op_name)
                    )
                    self._handles.append(h2)

                # Inner mha pre-softmax patch
                mha = getattr(ta, "mha", None)
                if mha is not None:
                    self._patch_triangle_attention_mha(mha, li, msa_li, op_name)

        # AttentionPairBias v2 (s-track)
        attn = getattr(layer, "attention", None)
        if attn is not None and isinstance(attn, AttentionPairBiasV2):
            # Outer hook
            h = attn.register_forward_hook(
                self._make_forward_hook(li, msa_li, type(attn).__name__)
            )
            self._handles.append(h)
            # Pre-softmax patch
            self._patch_attention_pair_bias_v2(attn, li, msa_li)

    def register_msa_layer(
        self,
        layer: nn.Module,
        msa_layer_index: int,
    ) -> None:
        """Register hooks for OuterProductMean and PairWeightedAveraging inside MSALayer."""
        from boltz.model.layers.outer_product_mean import OuterProductMean
        from boltz.model.layers.pair_averaging import PairWeightedAveraging

        li = -1
        msa_li = msa_layer_index

        for attr, cls in [
            ("outer_product_mean", OuterProductMean),
            ("pair_weighted_averaging", PairWeightedAveraging),
        ]:
            submod = getattr(layer, attr, None)
            if submod is not None and isinstance(submod, cls):
                h = submod.register_forward_hook(
                    self._make_forward_hook(li, msa_li, type(submod).__name__)
                )
                self._handles.append(h)

    def remove_all(self) -> None:
        """Remove all registered hooks and undo monkey-patches."""
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()

        for mod, attr, orig_fn in self._patched:
            try:
                setattr(mod, attr, orig_fn)
            except Exception:
                pass
        self._patched.clear()


# ---------------------------------------------------------------------------
# Weight analysis
# ---------------------------------------------------------------------------

def analyse_weights(
    model: nn.Module,
    output_file: Path,
    checkpoint_name: str,
) -> None:
    """Iterate over all Pairformer operator sub-modules and log weight statistics.

    This is a one-time analysis (weights are fixed between inference runs).
    Results are saved to {output_dir}/{checkpoint_name}_weights.jsonl.

    Per-output-channel stats identify whether MXFP8 column-wise quantisation
    would encounter problematic channel outliers (the SmoothQuant scenario).
    """
    log.info("Analysing weights → %s", output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    from boltz.model.layers.pairformer import PairformerModule, PairformerNoSeqLayer
    from boltz.model.layers.triangular_mult import (
        TriangleMultiplicationIncoming,
        TriangleMultiplicationOutgoing,
    )
    from boltz.model.layers.triangular_attention.attention import TriangleAttention
    from boltz.model.layers.attentionv2 import AttentionPairBias as AttentionPairBiasV2
    from boltz.model.layers.outer_product_mean import OuterProductMean
    from boltz.model.layers.pair_averaging import PairWeightedAveraging
    from boltz.model.modules.trunkv2 import MSALayer

    OPERATOR_CLASSES = (
        TriangleMultiplicationOutgoing,
        TriangleMultiplicationIncoming,
        TriangleAttention,
        AttentionPairBiasV2,
        OuterProductMean,
        PairWeightedAveraging,
    )

    with output_file.open("w") as fh:
        # Walk named modules
        for full_name, submod in model.named_modules():
            if not isinstance(submod, OPERATOR_CLASSES):
                continue

            # Determine layer indices from the module path
            # e.g. "pairformer_module.layers.7.tri_mul_out" → layer_index=7
            # e.g. "msa_module.layers.2.outer_product_mean" → msa_layer_index=2
            layer_index = -1
            msa_layer_index = -1
            parts = full_name.split(".")
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts):
                    try:
                        idx = int(parts[i + 1])
                        # Distinguish pairformer vs msa by context
                        if "msa_module" in full_name:
                            msa_layer_index = idx
                        else:
                            layer_index = idx
                    except ValueError:
                        pass

            op_name = type(submod).__name__

            for param_name, param in submod.named_parameters(recurse=True):
                if param.ndim < 1:
                    continue
                try:
                    stats = _compute_weight_stats(param.data)
                    rec = WeightRecord(
                        layer_index=layer_index,
                        msa_layer_index=msa_layer_index,
                        operator_name=op_name,
                        parameter_name=param_name,
                        shape=list(param.shape),
                        dtype=str(param.dtype),
                        **stats,
                    )
                    fh.write(json.dumps(asdict(rec)) + "\n")
                except Exception:
                    log.warning(
                        "Weight stats failed %s/%s: %s",
                        full_name, param_name, traceback.format_exc(),
                    )

    log.info("Weight analysis complete.")


# ---------------------------------------------------------------------------
# Model loading (mirrors boltz predict exactly)
# ---------------------------------------------------------------------------

def load_boltz2_model(
    cache: Path,
    device: str,
    recycling_steps: int = 1,
    diffusion_samples: int = 1,
) -> "Boltz2":
    """Load Boltz-2 checkpoint the same way `boltz predict` does.

    Uses PairformerArgsV2 (num_blocks=64, v2=True) to match the deployed weights.
    """
    from dataclasses import asdict

    from boltz.model.models.boltz2 import Boltz2

    # These mirror the dataclasses in main.py exactly
    pairformer_args = dict(
        num_blocks=64,
        num_heads=16,
        dropout=0.0,
        activation_checkpointing=False,
        offload_to_cpu=False,
        v2=True,
    )
    msa_args = dict(
        msa_s=64,
        msa_blocks=4,
        msa_dropout=0.0,
        z_dropout=0.0,
        use_paired_feature=True,
        pairwise_head_width=32,
        pairwise_num_heads=4,
        activation_checkpointing=False,
        offload_to_cpu=False,
        subsample_msa=False,
        num_subsampled_msa=1024,
    )
    diffusion_params = dict(
        gamma_0=0.8, gamma_min=1.0, noise_scale=1.003, rho=7, step_scale=1.5,
        sigma_min=0.0001, sigma_max=160.0, sigma_data=16.0,
        P_mean=-1.2, P_std=1.5,
        coordinate_augmentation=True,
        alignment_reverse_diff=True,
        synchronize_sigmas=True,
    )
    predict_args = dict(
        recycling_steps=recycling_steps,
        sampling_steps=20,
        diffusion_samples=diffusion_samples,
        max_parallel_samples=1,
        write_confidence_summary=False,
        write_full_pae=False,
        write_full_pde=False,
    )
    steering_args = dict(
        fk_steering=False,
        num_particles=3,
        fk_lambda=4.0,
        fk_resampling_interval=3,
        physical_guidance_update=False,
        contact_guidance_update=True,
        num_gd_steps=20,
    )

    checkpoint = cache / "boltz2_conf.ckpt"
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}\n"
            "Run `boltz predict` once to auto-download, or set --cache."
        )

    log.info("Loading Boltz-2 from %s", checkpoint)
    model = Boltz2.load_from_checkpoint(
        str(checkpoint),
        strict=True,
        predict_args=predict_args,
        map_location="cpu",
        diffusion_process_args=diffusion_params,
        ema=False,
        use_kernels=False,    # kernels bypass hook-able Python paths
        pairformer_args=pairformer_args,
        msa_args=msa_args,
        steering_args=steering_args,
    )
    model.eval()
    model.to(device)
    log.info("Model loaded (%.1f GB VRAM)", _vram_gb(device))
    return model


def _vram_gb(device: str) -> float:
    try:
        return torch.cuda.memory_allocated(device) / 1e9
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Data pipeline helpers (mirrors boltz predict)
# ---------------------------------------------------------------------------

def prepare_input(
    input_path: Path,
    out_dir: Path,
    cache: Path,
    use_msa_server: bool,
    max_seqlen: int,
) -> Optional[Tuple[Any, Any]]:
    """Run the full boltz preprocessing pipeline.

    Returns (data_module, manifest) or None if seqlen exceeds max_seqlen.
    """
    import warnings as _w
    _w.filterwarnings("ignore")

    from boltz.data import const
    from boltz.data.mol import load_canonicals
    from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
    from boltz.data.types import Manifest, Record
    from boltz.main import (
        check_inputs,
        process_inputs,
        filter_inputs_structure,
        BoltzProcessedInput,
    )

    mol_dir = cache / "mols"

    data_paths = check_inputs(input_path)
    process_inputs(
        data=data_paths,
        out_dir=out_dir,
        ccd_path=cache / "ccd.pkl",
        mol_dir=mol_dir,
        use_msa_server=use_msa_server,
        msa_server_url="https://api.colabfold.com",
        msa_pairing_strategy="greedy",
        msa_server_username=None,
        msa_server_password=None,
        api_key_header=None,
        api_key_value=None,
        boltz2=True,
        preprocessing_threads=1,
        max_msa_seqs=8192,
    )

    manifest = Manifest.load(out_dir / "processed" / "manifest.json")

    # Check sequence length
    for rec in manifest.records:
        # Approximate seqlen from number of chains / tokens
        # We can't know the exact token count before featurisation; use chain lengths
        total_len = sum(
            len(getattr(c, "sequence", "") or "")
            for c in rec.chains
        )
        if total_len > max_seqlen:
            log.warning(
                "Skipping %s: approximate seqlen %d > %d",
                rec.id, total_len, max_seqlen,
            )
            return None

    processed_dir = out_dir / "processed"
    processed = BoltzProcessedInput(
        manifest=manifest,
        targets_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa",
        constraints_dir=(
            (processed_dir / "constraints")
            if (processed_dir / "constraints").exists() else None
        ),
        template_dir=(
            (processed_dir / "templates")
            if (processed_dir / "templates").exists() else None
        ),
        extra_mols_dir=(
            (processed_dir / "mols")
            if (processed_dir / "mols").exists() else None
        ),
    )

    data_module = Boltz2InferenceDataModule(
        manifest=processed.manifest,
        target_dir=processed.targets_dir,
        msa_dir=processed.msa_dir,
        mol_dir=mol_dir,
        num_workers=0,
        constraints_dir=processed.constraints_dir,
        template_dir=processed.template_dir,
        extra_mols_dir=processed.extra_mols_dir,
    )

    return data_module, manifest


# ---------------------------------------------------------------------------
# Main capture routine
# ---------------------------------------------------------------------------

def run_capture(
    model: "Boltz2",
    data_module: Any,
    activation_file: Path,
    layers_to_capture: List[int],
    top_k: int,
    recycling_steps: int,
    diffusion_samples: int,
    device: str,
) -> None:
    """Run inference with hooks active; write activation JSONL records."""

    from boltz.model.modules.trunkv2 import MSALayer

    # We need to know seqlen at hook-fire time.
    # We resolve it lazily from the first tensor shape seen.
    hook_manager_holder: List[Optional[HookManager]] = [None]

    dataloader = data_module.predict_dataloader()
    batches = list(dataloader)
    if not batches:
        log.warning("No batches found in dataloader.")
        return

    for batch_idx, batch in enumerate(batches):
        # Move batch to device
        batch = _move_to_device(batch, device)

        # Determine seqlen from token_pad_mask
        seqlen = int(batch.get("token_pad_mask", torch.zeros(1, 1)).shape[-1])
        log.info("Batch %d: seqlen=%d", batch_idx, seqlen)

        # Build HookManager now that we know seqlen
        hm = HookManager(
            output_file=activation_file,
            layers_to_capture=layers_to_capture,
            top_k=top_k,
            seqlen=seqlen,
        )

        # Register hooks on main Pairformer module
        pf_module = getattr(model, "pairformer_module", None)
        if pf_module is not None:
            layers_attr = getattr(pf_module, "_orig_mod", pf_module)
            pf_layers = getattr(layers_attr, "layers", [])
            num_blocks = len(pf_layers)
            log.info("Found %d Pairformer blocks", num_blocks)
            for li, layer in enumerate(pf_layers):
                hm.register_pairformer_layer(layer, li)

        # Register hooks on MSA module layers
        msa_mod = getattr(model, "msa_module", None)
        if msa_mod is not None:
            msa_mod_actual = getattr(msa_mod, "_orig_mod", msa_mod)
            msa_layers = getattr(msa_mod_actual, "layers", [])
            log.info("Found %d MSA layers", len(msa_layers))
            for msa_li, msa_layer in enumerate(msa_layers):
                if isinstance(msa_layer, MSALayer):
                    hm.register_msa_layer(msa_layer, msa_li)

        registered_hook_count = len(hm._handles) + len(hm._patched)
        log.info(
            "Registered %d hooks / patches for batch %d",
            registered_hook_count, batch_idx,
        )

        try:
            with torch.no_grad():
                _ = model(
                    batch,
                    recycling_steps=recycling_steps,
                    num_sampling_steps=20,
                    diffusion_samples=diffusion_samples,
                    max_parallel_samples=1,
                    run_confidence_sequentially=True,
                )
        except Exception:
            log.error("Inference failed: %s", traceback.format_exc())
        finally:
            hm.remove_all()
            torch.cuda.empty_cache()
            gc.collect()

        log.info(
            "Batch %d complete. VRAM: %.1f GB", batch_idx, _vram_gb(device)
        )


def _move_to_device(obj: Any, device: str) -> Any:
    if isinstance(obj, Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        converted = [_move_to_device(v, device) for v in obj]
        return type(obj)(converted)
    return obj


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------

def print_summary(activation_file: Path, summary_file: Path) -> None:
    """Read the JSONL file and print the three summary tables.

    Table 1 — Outlier hotspots
    Table 2 — Symmetry breaks
    Table 3 — Channel outlier severity
    """
    if not activation_file.exists():
        log.warning("Activation file not found: %s", activation_file)
        return

    records = []
    with activation_file.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if not records:
        log.warning("No activation records found.")
        return

    def _fmt(r: dict) -> str:
        li   = r.get("layer_index", -1)
        msa  = r.get("msa_layer_index", -1)
        loc  = f"pf_layer={li}" if li >= 0 else f"msa_layer={msa}"
        return f"{r['operator_name']:40s}  {loc:20s}  role={r['tensor_role']}"

    lines = []

    # ── Table 1 ──────────────────────────────────────────────────────────────
    lines.append("\n" + "="*80)
    lines.append("TABLE 1 — Outlier Hotspots (MXFP8 range concerns)")
    lines.append("="*80)

    top_absmax = sorted(records, key=lambda r: r.get("abs_max", 0), reverse=True)[:5]
    lines.append("\nTop-5 by abs_max:")
    for r in top_absmax:
        lines.append(f"  abs_max={r['abs_max']:12.4f}   {_fmt(r)}")

    top_kurt = sorted(records, key=lambda r: r.get("kurtosis", 0), reverse=True)[:5]
    lines.append("\nTop-5 by kurtosis (Fisher):")
    for r in top_kurt:
        lines.append(f"  kurtosis={r['kurtosis']:10.2f}   {_fmt(r)}")

    top_frac = sorted(records, key=lambda r: r.get("fraction_gt6", 0), reverse=True)[:5]
    lines.append("\nTop-5 by fraction |x|>6 (E4M3 overflow risk):")
    for r in top_frac:
        lines.append(f"  frac_gt6={r['fraction_gt6']:.6f}   {_fmt(r)}")

    # ── Table 2 ──────────────────────────────────────────────────────────────
    lines.append("\n" + "="*80)
    lines.append("TABLE 2 — Symmetry Breaks (pair matrix asymmetry)")
    lines.append("="*80)

    sym_recs = [r for r in records if r.get("symmetry_score", -1) >= 0]
    if sym_recs:
        worst_corr = sorted(
            sym_recs, key=lambda r: r.get("symmetric_outlier_correlation", 1.0)
        )[:5]
        lines.append("\nLowest symmetric_outlier_correlation (outlier pairs not mirrored):")
        for r in worst_corr:
            lines.append(
                f"  corr={r['symmetric_outlier_correlation']:.4f}   {_fmt(r)}"
            )

        worst_sym_max = sorted(
            sym_recs, key=lambda r: r.get("symmetry_max", 0), reverse=True
        )[:5]
        lines.append("\nHighest symmetry_max (largest individual asymmetry):")
        for r in worst_sym_max:
            lines.append(
                f"  sym_max={r['symmetry_max']:.6f}   {_fmt(r)}"
            )
    else:
        lines.append("  (no pair-tensor records found)")

    # ── Table 3 ──────────────────────────────────────────────────────────────
    lines.append("\n" + "="*80)
    lines.append("TABLE 3 — Channel Outlier Severity (SmoothQuant indicator)")
    lines.append("="*80)

    top_chan = sorted(
        records, key=lambda r: r.get("channel_max_to_median_ratio", 0), reverse=True
    )[:5]
    lines.append("\nTop-5 by channel_max_to_median_ratio:")
    for r in top_chan:
        lines.append(
            f"  ratio={r['channel_max_to_median_ratio']:10.2f}   {_fmt(r)}"
        )

    lines.append("\n" + "="*80)

    summary_text = "\n".join(lines)
    print(summary_text)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with summary_file.open("w") as fh:
        fh.write(summary_text + "\n")
    log.info("Summary saved → %s", summary_file)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Capture Pairformer activation/weight statistics for MXFP8 analysis."
    )
    p.add_argument("--input", required=True, type=Path,
                   help=".yaml or .fasta input file (same format as boltz predict).")
    p.add_argument("--output_dir", default=Path("./boltz_results/activation_stats"), type=Path,
                   help="Directory for JSONL output files. Default: ./boltz_results/activation_stats")
    p.add_argument("--cache", default=None, type=Path,
                   help="Boltz cache directory (default: ~/.boltz or $BOLTZ_CACHE).")
    p.add_argument("--use_msa_server", action="store_true",
                   help="Use MMSeqs2 server for MSA generation.")
    p.add_argument("--device", default="cuda:0", type=str,
                   help="Torch device string (default: cuda:0).")
    p.add_argument("--max_seqlen", default=800, type=int,
                   help="Skip inputs longer than this (approximate token count).")
    p.add_argument(
        "--layers_to_capture",
        default="0,8,16,24,32,40,48,56,63",
        type=str,
        help=(
            '"all" or comma-separated layer indices, e.g. "0,8,16,24,32,40,48,56,63". '
            'Default: "0,8,16,24,32,40,48,56,63"'
        ),
    )
    p.add_argument("--capture_weights", action="store_true",
                   help="Run one-time weight analysis before inference.")
    p.add_argument("--capture_top_k", default=100, type=int,
                   help="Number of top absolute values to record per tensor (default: 100).")
    p.add_argument("--recycling_steps", default=1, type=int,
                   help="Recycling steps (default: 1; fewer is faster for profiling).")
    p.add_argument("--diffusion_samples", default=1, type=int,
                   help="Diffusion samples (default: 1).")
    return p.parse_args()


def resolve_cache(cache_arg: Optional[Path]) -> Path:
    if cache_arg is not None:
        return cache_arg.expanduser().resolve()
    env = os.environ.get("BOLTZ_CACHE")
    if env:
        return Path(env).expanduser().resolve()
    return Path("~/.boltz").expanduser()


def resolve_layers(spec: str, num_blocks: int = 64) -> List[int]:
    if spec.strip().lower() == "all":
        return list(range(num_blocks))
    try:
        return [int(x.strip()) for x in spec.split(",") if x.strip()]
    except ValueError:
        raise ValueError(f"Invalid --layers_to_capture spec: {spec!r}")


def main() -> None:
    args = parse_args()

    # Suppress noisy warnings from Lightning / other libs
    warnings.filterwarnings("ignore")
    os.environ.setdefault("CUEQ_DEFAULT_CONFIG", "1")
    os.environ.setdefault("CUEQ_DISABLE_AOT_TUNING", "1")

    # Resolve cache
    cache = resolve_cache(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    # Output dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve layers
    layers_to_capture = resolve_layers(args.layers_to_capture, num_blocks=64)
    log.info("Layers to capture: %s", layers_to_capture)

    # ── Load model ──────────────────────────────────────────────────────────
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("highest")

    from rdkit import Chem
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

    model = load_boltz2_model(
        cache=cache,
        device=args.device,
        recycling_steps=args.recycling_steps,
        diffusion_samples=args.diffusion_samples,
    )

    checkpoint_name = "boltz2_conf"

    # ── Weight analysis ──────────────────────────────────────────────────────
    if args.capture_weights:
        weight_file = args.output_dir / f"{checkpoint_name}_weights.jsonl"
        analyse_weights(model, weight_file, checkpoint_name)

    # ── Prepare input ────────────────────────────────────────────────────────
    input_name = args.input.stem
    proc_out_dir = args.output_dir / f"boltz_results_{input_name}"

    result = prepare_input(
        input_path=args.input,
        out_dir=proc_out_dir,
        cache=cache,
        use_msa_server=args.use_msa_server,
        max_seqlen=args.max_seqlen,
    )
    if result is None:
        log.error("Input preparation failed or seqlen exceeded limit. Exiting.")
        return

    data_module, manifest = result

    # ── Activation capture ───────────────────────────────────────────────────
    activation_file = args.output_dir / f"{input_name}_activations.jsonl"
    log.info("Writing activations → %s", activation_file)

    run_capture(
        model=model,
        data_module=data_module,
        activation_file=activation_file,
        layers_to_capture=layers_to_capture,
        top_k=args.capture_top_k,
        recycling_steps=args.recycling_steps,
        diffusion_samples=args.diffusion_samples,
        device=args.device,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    summary_file = args.output_dir / f"{input_name}_summary.txt"
    print_summary(activation_file, summary_file)


if __name__ == "__main__":
    main()
