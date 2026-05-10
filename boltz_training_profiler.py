#!/workspace/boltz/.venv/bin/python
"""
boltz_training_profiler.py
──────────────────────────
Profiles Boltz-2 (AlphaFold3-class) training-mode activations AND gradients
for the pair representation (z) across Pairformer blocks.

Purpose: characterise activation / gradient distributions for MX/NV FP8/FP4
mixed-precision training research.

Usage
─────
# Step 1: print forward() signature and required feature keys, then exit
.venv/bin/python boltz_training_profiler.py \\
    --boltz_src /workspace/boltz/src \\
    --output_dir ./training_results \\
    --discover

# Step 2: full profiling run
.venv/bin/python boltz_training_profiler.py \\
    --boltz_src /workspace/boltz/src \\
    --output_dir ./training_results \\
    --device cuda \\
    --target_Ns 64 128 256 512 768 1024 1536 2048
"""

import argparse
import gc
import inspect
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for HPC / headless
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kurtosis as scipy_kurtosis, pearsonr


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — DISCOVER BATCH FORMAT
# ─────────────────────────────────────────────────────────────────────────────

def discover_batch_format(model, boltz_src: str) -> None:
    """Print model forward() / training_step() signatures and key source."""
    print("\n" + "=" * 70)
    print("STEP 1: DISCOVERING BATCH FORMAT")
    print("=" * 70)

    print("\n=== forward() signature ===")
    print(inspect.signature(model.forward))

    print("\n=== training_step() signature ===")
    print(inspect.signature(model.training_step))

    # Read boltz2.py source and show the forward method
    boltz2_path = os.path.join(boltz_src, "boltz", "model", "models", "boltz2.py")
    try:
        with open(boltz2_path) as fh:
            src = fh.read()
        start = src.find("def forward")
        print("\n=== boltz2.py: def forward (first 3000 chars) ===")
        print(src[start : start + 3000])
    except FileNotFoundError:
        print(f"  [warn] Could not open {boltz2_path}")

    # Discover pairformer block container
    print("\n=== Pairformer block container ===")
    pf_mod = model.pairformer_module
    if hasattr(pf_mod, "_orig_mod"):          # compiled version
        pf_mod = pf_mod._orig_mod
    if hasattr(pf_mod, "layers"):
        print(f"  model.pairformer_module.layers  — {len(pf_mod.layers)} blocks")
    elif hasattr(pf_mod, "blocks"):
        print(f"  model.pairformer_module.blocks  — {len(pf_mod.blocks)} blocks")
    else:
        print("  Could not find block list; inspect model.pairformer_module manually")

    print("\n=== model.hparams (constructor args) ===")
    for k, v in model.hparams.items():
        if not isinstance(v, dict):
            print(f"  {k}: {v}")

    print("\n=== Required feats keys (inferred from source) ===")
    required = [
        # Token-level
        "token_index", "residue_index", "asym_id", "entity_id", "sym_id",
        "mol_type", "res_type [N, num_tokens=33]", "disto_center [N, 3]",
        "token_bonds [N, N, 1]", "token_pad_mask [N]",
        "token_resolved_mask [N]", "token_disto_mask [N]",
        "pocket_feature [N, 4]", "cyclic_period [N]",
        # Boltz-2 extras
        "contact_conditioning [N, N, 5]  ← one-hot over 5 classes (UNSPECIFIED=0)",
        "contact_threshold [N, N]        ← float distance threshold",
        "type_bonds [N, N]               ← long, bond type index (bond_type_feature=True)",
        "method_feature [N]              ← long, method class index",
        "modified [N]                    ← long, 0/1 modified flag",
        # MSA-level
        "msa [n_msa, N, 33]", "msa_paired [n_msa, N]",
        "deletion_value [n_msa, N]", "has_deletion [n_msa, N]",
        "deletion_mean [N]", "profile [N, 33]", "msa_mask [n_msa, N]",
        # Atom-level
        "ref_pos [n_atoms, 3]", "atom_pad_mask [n_atoms]",
        "atom_resolved_mask [n_atoms]",
        "ref_element [n_atoms, 128]", "ref_charge [n_atoms]",
        "ref_atom_name_chars [n_atoms, 4, 64]", "ref_space_uid [n_atoms]",
        "atom_to_token [n_atoms, N] (one-hot float)",
    ]
    for r in required:
        print(f"  • {r}")

    print("\nRun without --discover to start profiling.\n")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — BUILD SYNTHETIC BATCH
# ─────────────────────────────────────────────────────────────────────────────

# Real constants from boltz.data.const (replicated here to avoid import order issues)
_NUM_TOKENS   = 33    # len(const.tokens): pad + "-" + 21 AA + 5 RNA + 5 DNA
_NUM_ELEMENTS = 128   # const.num_elements
_POCKET_DIMS  = 4     # len(const.pocket_contact_info)
_ATOM_NAME_BINS = 64  # num_bins for ref_atom_name_chars one-hot
_ATOM_NAME_CHARS = 4  # 4 character positions per atom name
_ATOMS_PER_WINDOW = 32  # atoms_per_window_queries default


def make_synthetic_batch(N: int, device: str, model) -> dict:
    """
    Build a minimal synthetic feature dict for a Boltz-2 forward pass
    through the trunk (InputEmbedder → MSAModule → PairformerModule).

    Parameters
    ----------
    N : int
        Number of tokens (sequence length).  Controls pair repr N×N×token_z.
    device : str
        'cuda' or 'cpu'.
    model :
        Boltz2 instance (used to read hparams for optional validation).

    Returns
    -------
    dict[str, torch.Tensor]
        All features required by the trunk.  Every tensor is on `device`.
    """
    # n_atoms must be a multiple of atoms_per_window_queries (32)
    # Use one atom per token, rounded up to the nearest multiple of 32.
    n_atoms_raw = N
    n_atoms = (
        (n_atoms_raw + _ATOMS_PER_WINDOW - 1) // _ATOMS_PER_WINDOW
    ) * _ATOMS_PER_WINDOW

    n_msa = 4   # minimal MSA: a few sequences

    d = device

    # ── Token-level ──────────────────────────────────────────────────────────
    # res_type: one-hot over num_tokens classes, all "ALA" (index 2)
    res_type_idx = torch.full((1, N), 2, dtype=torch.long, device=d)
    res_type = torch.nn.functional.one_hot(res_type_idx, num_classes=_NUM_TOKENS).float()

    # pocket_feature: one-hot over 4 classes, all "UNSPECIFIED" (index 0)
    pocket_feat_idx = torch.zeros(1, N, dtype=torch.long, device=d)
    pocket_feature = torch.nn.functional.one_hot(
        pocket_feat_idx, num_classes=_POCKET_DIMS
    ).float()

    # token_bonds: shape [1, N, N, 1], mostly 0
    token_bonds = torch.zeros(1, N, N, 1, device=d)

    batch = {
        # indices
        "token_index":      torch.arange(N, device=d).unsqueeze(0),            # [1, N]
        "residue_index":    torch.arange(N, device=d).unsqueeze(0),            # [1, N]
        "asym_id":          torch.zeros(1, N, dtype=torch.long, device=d),     # [1, N]
        "entity_id":        torch.zeros(1, N, dtype=torch.long, device=d),     # [1, N]
        "sym_id":           torch.zeros(1, N, dtype=torch.long, device=d),     # [1, N]
        "mol_type":         torch.zeros(1, N, dtype=torch.long, device=d),     # [1, N]
        "cyclic_period":    torch.zeros(1, N, dtype=torch.long, device=d),     # [1, N]
        # token representations
        "res_type":         res_type,                                           # [1, N, 33]
        "profile":          res_type.clone(),                                   # [1, N, 33]
        "deletion_mean":    torch.zeros(1, N, device=d),                       # [1, N]
        "pocket_feature":   pocket_feature,                                     # [1, N, 4]
        "disto_center":     torch.randn(1, N, 3, device=d),                    # [1, N, 3]
        "token_bonds":      token_bonds,                                        # [1, N, N, 1]
        # masks
        "token_pad_mask":       torch.ones(1, N, device=d),                    # [1, N]
        "token_resolved_mask":  torch.ones(1, N, device=d),                    # [1, N]
        "token_disto_mask":     torch.ones(1, N, device=d),                    # [1, N]
        # frames (required by featurizer but not used in trunk; zeros are fine)
        "frames_idx":           torch.zeros(1, N, 3, dtype=torch.long, device=d),  # [1, N, 3]
        "frame_resolved_mask":  torch.zeros(1, N, device=d),                   # [1, N]
    }

    # ── MSA-level ─────────────────────────────────────────────────────────────
    # msa: integer indices (MSAModuleV2 calls one_hot internally)
    batch.update({
        "msa":            torch.full((1, n_msa, N), 2, dtype=torch.long, device=d),   # [1, n_msa, N]
        "msa_paired":     torch.zeros(1, n_msa, N, device=d),                         # [1, n_msa, N]
        "deletion_value": torch.zeros(1, n_msa, N, device=d),                         # [1, n_msa, N]
        "has_deletion":   torch.zeros(1, n_msa, N, device=d),                         # [1, n_msa, N] float
        "msa_mask":       torch.ones(1, n_msa, N, device=d),                          # [1, n_msa, N]
    })

    # ── Atom-level ────────────────────────────────────────────────────────────
    # ref_element: one-hot over 128 elements (carbon = element index 6)
    ref_elem_idx = torch.full((1, n_atoms), 6, dtype=torch.long, device=d)
    ref_element = torch.nn.functional.one_hot(
        ref_elem_idx, num_classes=_NUM_ELEMENTS
    ).float()

    # ref_atom_name_chars: one-hot over 64 classes for each of 4 char positions
    # shape: [1, n_atoms, 4, 64]
    atom_char_idx = torch.zeros(1, n_atoms, _ATOM_NAME_CHARS, dtype=torch.long, device=d)
    ref_atom_name_chars = torch.nn.functional.one_hot(
        atom_char_idx, num_classes=_ATOM_NAME_BINS
    ).float()

    # atom_to_token: one-hot [1, n_atoms, N]
    # Map each atom → its corresponding token (1-to-1 for first N atoms, 0 for padding)
    atom_to_token_idx = torch.clamp(
        torch.arange(n_atoms, device=d), max=N - 1
    ).unsqueeze(0)  # [1, n_atoms]
    atom_to_token = torch.nn.functional.one_hot(
        atom_to_token_idx, num_classes=N
    ).float()  # [1, n_atoms, N]

    batch.update({
        "ref_pos":              torch.randn(1, n_atoms, 3, device=d),           # [1, n_atoms, 3]
        "atom_pad_mask":        torch.ones(1, n_atoms, device=d),               # [1, n_atoms]
        "atom_resolved_mask":   torch.ones(1, n_atoms, device=d),               # [1, n_atoms]
        "ref_element":          ref_element,                                     # [1, n_atoms, 128]
        "ref_charge":           torch.zeros(1, n_atoms, device=d),              # [1, n_atoms]
        "ref_atom_name_chars":  ref_atom_name_chars,                            # [1, n_atoms, 4, 64]
        "ref_space_uid":        torch.zeros(1, n_atoms, dtype=torch.long, device=d),  # [1, n_atoms]
        "atom_to_token":        atom_to_token,                                   # [1, n_atoms, N]
    })

    # ── Boltz-2-specific feats ────────────────────────────────────────────────
    # contact_conditioning: one-hot over 5 classes (UNSPECIFIED=0 → all tokens
    # unspecified → only index 0 is 1, rest 0)
    contact_cond = torch.zeros(1, N, N, 5, device=d)
    contact_cond[:, :, :, 0] = 1.0        # all UNSPECIFIED

    # type_bonds: long tensor [1, N, N], 0 = no bond (used by bond_type_feature)
    type_bonds = torch.zeros(1, N, N, dtype=torch.long, device=d)

    # method_feature: int index per token (0 = first method class)
    method_feature = torch.zeros(1, N, dtype=torch.long, device=d)

    # modified: int flag per token (0 = unmodified residue)
    modified = torch.zeros(1, N, dtype=torch.long, device=d)

    batch.update({
        "contact_conditioning": contact_cond,                                    # [1, N, N, 5]
        "contact_threshold":    torch.zeros(1, N, N, device=d),                 # [1, N, N]
        "type_bonds":           type_bonds,                                      # [1, N, N]
        "method_feature":       method_feature,                                  # [1, N]
        "modified":             modified,                                        # [1, N]
    })

    return batch


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — PROXY LOSS (trunk only, no diffusion module)
# ─────────────────────────────────────────────────────────────────────────────

def compute_proxy_loss(model, feats: dict, device: str) -> torch.Tensor:
    """
    Run the Boltz-2 trunk (InputEmbedder → MSA → Pairformer → Distogram)
    and return a scalar proxy loss that flows gradients through all
    Pairformer blocks.

    We deliberately bypass model.forward() to avoid:
      • the diffusion structure module (needs complex atom features)
      • gradient-blocking when structure_prediction_training=False

    The proxy loss is:
        loss = mean(|z|) + mean(|pdistogram|)
    which forces non-trivial gradients through all 48 Pairformer blocks.
    """
    # ── InputEmbedder ──
    s_inputs = model.input_embedder(feats)

    # ── Initialise s and z ──
    s_init = model.s_init(s_inputs)
    z_init = (
        model.z_init_1(s_inputs)[:, :, None]
        + model.z_init_2(s_inputs)[:, None, :]
    )
    rel_pos_enc = model.rel_pos(feats)
    z_init = z_init + rel_pos_enc
    z_init = z_init + model.token_bonds(feats["token_bonds"].float())

    # ── Boltz-2 extras ──
    if model.bond_type_feature:
        z_init = z_init + model.token_bonds_type(feats["type_bonds"].long())
    z_init = z_init + model.contact_conditioning(feats)

    # ── Recycling (0 extra recycles — just the final round) ──
    s = torch.zeros_like(s_init)
    z = torch.zeros_like(z_init)

    s = s_init + model.s_recycle(model.s_norm(s))
    z = z_init + model.z_recycle(model.z_norm(z))

    # ── Token mask ──
    mask = feats["token_pad_mask"].float()
    pair_mask = mask[:, :, None] * mask[:, None, :]

    # ── MSA module (updates z) ──
    msa_module = model.msa_module
    if hasattr(msa_module, "_orig_mod") and not model.training:
        msa_module = msa_module._orig_mod
    z = z + msa_module(z, s_inputs, feats, use_kernels=False)

    # ── Pairformer ──
    pairformer = model.pairformer_module
    if hasattr(pairformer, "_orig_mod"):      # compiled
        pairformer = pairformer._orig_mod
    s, z = pairformer(s, z, mask=mask, pair_mask=pair_mask, use_kernels=False)

    # ── Proxy loss ──
    pdistogram = model.distogram_module(z)
    loss = z.float().abs().mean() + pdistogram.float().abs().mean()
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — HOOK REGISTRATION
# ─────────────────────────────────────────────────────────────────────────────

def register_training_hooks(model):
    """
    Register forward and full-backward hooks on each PairformerLayer block.

    Forward hook captures z (pair repr) output  → shape [N, N, token_z].
    Backward hook captures dL/dz gradient output → shape [N, N, token_z].

    Returns
    -------
    fwd_acts   : dict[int → Tensor]   block_idx → z  [N, N, token_z]
    bwd_grads  : dict[int → Tensor]   block_idx → dL/dz
    handles    : list of hook handles (call h.remove() to clean up)
    """
    fwd_acts   = {}
    bwd_grads  = {}
    handles    = []

    pf_mod = model.pairformer_module
    if hasattr(pf_mod, "_orig_mod"):
        pf_mod = pf_mod._orig_mod
    blocks = pf_mod.layers

    print(f"  Registering hooks on {len(blocks)} PairformerLayer blocks")

    def make_fwd_hook(idx: int):
        def hook(module, inp, output):
            # output = (s, z); z is the pair representation
            z_out = output[1] if isinstance(output, tuple) else output
            # z_out shape: [1, N, N, token_z]  →  squeeze batch → [N, N, token_z]
            fwd_acts[idx] = z_out.float().squeeze(0).detach().clone().cpu()
        return hook

    def make_bwd_hook(idx: int):
        def hook(module, grad_inp, grad_out):
            # grad_out[0] corresponds to gradient w.r.t. the first output (s)
            # grad_out[1] corresponds to gradient w.r.t. the second output (z)
            # PairformerLayer returns (s, z), so grad_out = (grad_s, grad_z)
            if isinstance(grad_out, tuple) and len(grad_out) >= 2:
                gz = grad_out[1]
            else:
                gz = grad_out[0]
            if gz is not None:
                bwd_grads[idx] = gz.float().squeeze(0).detach().clone().cpu()
        return hook

    for idx, block in enumerate(blocks):
        h1 = block.register_forward_hook(make_fwd_hook(idx))
        h2 = block.register_full_backward_hook(make_bwd_hook(idx))
        handles.extend([h1, h2])

    return fwd_acts, bwd_grads, handles


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — FULL PROFILING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def profile_one_N(model, N: int, device: str, blocks_to_profile: list) -> dict:
    """
    Profile one sequence length N.

    Returns a dict with keys:
        N, oom, memory, blocks
    or on OOM:
        N, oom=True, phase, memory
    """
    sep = "=" * 52
    print(f"\n{sep}")
    print(f"  Profiling N = {N}")
    n_atoms = ((N + _ATOMS_PER_WINDOW - 1) // _ATOMS_PER_WINDOW) * _ATOMS_PER_WINDOW
    token_z = _get_token_z(model)
    pair_mb  = N * N * token_z * 2 / 1e6    # BF16
    atom_mb  = n_atoms * (3 + _NUM_ELEMENTS + 1 + _ATOM_NAME_CHARS * _ATOM_NAME_BINS) * 4 / 1e6
    print(f"  Pair repr: {N}×{N}×{token_z} = {pair_mb:.1f} MB (BF16)")
    print(f"  n_atoms: {n_atoms}  (padded to mult of {_ATOMS_PER_WINDOW})")
    print(sep)

    # Reset memory stats
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

    mem_before = _cuda_mem(device)

    batch = make_synthetic_batch(N, device, model)
    fwd_acts, bwd_grads, handles = register_training_hooks(model)

    try:
        model.zero_grad()
        loss = compute_proxy_loss(model, batch, device)

        mem_after_fwd = _cuda_mem(device)
        mem_peak_fwd  = _cuda_peak(device)
        print(f"  Forward OK.  loss={loss.item():.4f}  peak={mem_peak_fwd:.1f} MB")

        loss.backward()

        mem_after_bwd = _cuda_mem(device)
        mem_peak_bwd  = _cuda_peak(device)
        print(f"  Backward OK.  peak={mem_peak_bwd:.1f} MB")
        print(f"  Activations captured: {sorted(fwd_acts.keys())}")
        print(f"  Gradients  captured:  {sorted(bwd_grads.keys())}")

        # Analyse each requested block
        block_results = {}
        for idx in blocks_to_profile:
            has_z = idx in fwd_acts
            has_g = idx in bwd_grads
            if has_z and has_g:
                z = fwd_acts[idx]    # [N, N, token_z]
                g = bwd_grads[idx]   # [N, N, token_z]
                print(f"  Analysing block {idx:2d}: z{tuple(z.shape)}  g{tuple(g.shape)}")
                block_results[idx] = analyze_block(z, g)
            elif has_z:
                print(f"  Block {idx}: activation OK but NO gradient captured")
                block_results[idx] = {"note": "no_gradient"}
            else:
                print(f"  Block {idx}: NOT captured")

        return {
            "N": N,
            "oom": False,
            "n_atoms": n_atoms,
            "token_z": token_z,
            "memory": {
                "before_MB":             mem_before,
                "peak_forward_MB":       mem_peak_fwd,
                "peak_backward_MB":      mem_peak_bwd,
                "after_fwd_MB":          mem_after_fwd,
                "after_bwd_MB":          mem_after_bwd,
                "theoretical_BF16_pair_MB":  N * N * token_z * 2 / 1e6,
                "theoretical_FP8_pair_MB":   N * N * token_z * 1 / 1e6,
                "theoretical_FP4_pair_MB":   N * N * token_z * 0.5 / 1e6,
            },
            "blocks": block_results,
        }

    except torch.cuda.OutOfMemoryError:
        phase = "backward" if len(bwd_grads) > 0 else "forward"
        print(f"  *** OOM during {phase} at N={N} ***")
        return {
            "N": N,
            "oom": True,
            "phase": phase,
            "memory": {
                "before_MB":            mem_before,
                "peak_forward_MB":      _cuda_peak(device),
                "peak_backward_MB":     None,
                "theoretical_BF16_pair_MB": N * N * _get_token_z(model) * 2 / 1e6,
            },
        }

    except Exception:
        traceback.print_exc()
        return {"N": N, "oom": False, "error": traceback.format_exc()}

    finally:
        for h in handles:
            h.remove()
        model.zero_grad()
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(x: torch.Tensor, label: str) -> dict:
    """
    Compute M1–M4 metrics for one tensor x of shape [N, N, token_z].

    Parameters
    ----------
    x : torch.Tensor  float32, [N, N, C]
    label : str       'activation' or 'gradient'
    """
    N = x.shape[0]
    C = x.shape[-1]

    # ── M1: Basic statistics ──────────────────────────────────────────────
    x_abs = x.abs()
    nonzero_mask = x_abs > 0
    if nonzero_mask.any():
        absmin = x_abs[nonzero_mask].amin().item()
        dyn_range = x_abs.amax().item() / max(absmin, 1e-10)
    else:
        absmin    = 0.0
        dyn_range = 0.0

    stats = {
        "mean":             x.mean().item(),
        "std":              x.std().item(),
        "absmax":           x_abs.amax().item(),
        "absmin_nonzero":   absmin,
        "dynamic_range":    dyn_range,
        "outlier_fraction": (x_abs > 3 * x.std()).float().mean().item(),
        "l2_norm":          x.norm(p=2).item(),
        "l1_norm":          x.norm(p=1).item(),
    }

    # ── M2: Kurtosis ──────────────────────────────────────────────────────
    flat = x.numpy().flatten().astype(np.float32)
    kurtosis = {
        "global": float(scipy_kurtosis(flat, fisher=True)),
        "per_hz_block": [
            float(
                scipy_kurtosis(
                    x[..., b * (C // 4) : (b + 1) * (C // 4)].numpy().flatten(),
                    fisher=True,
                )
            )
            for b in range(4)
        ],
    }

    # ── M3: Diagonal correlation ──────────────────────────────────────────
    dist = (
        torch.abs(
            torch.arange(N).unsqueeze(1) - torch.arange(N).unsqueeze(0)
        )
        .float()
        .numpy()
    )
    proximity = 1.0 / (1.0 + dist)
    scale_map  = x.abs().amax(dim=-1).numpy()          # [N, N]

    r_val, p_val = pearsonr(scale_map.flatten(), proximity.flatten())
    diagonal = {"pearson_r": float(r_val), "p_value": float(p_val)}

    # ── M4: FP8 format suitability ────────────────────────────────────────
    def pot_scale(t: torch.Tensor) -> torch.Tensor:
        """Power-of-two scale factor matching absmax."""
        return 2.0 ** torch.round(torch.log2(t.clamp(min=1e-10)))

    def quant_e4m3(t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return torch.round((t / s).clamp(-448, 448) * 8) / 8 * s

    def quant_e5m2(t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return torch.round((t / s).clamp(-57344, 57344) * 4) / 4 * s

    def relative_mse(orig: torch.Tensor, quant: torch.Tensor) -> float:
        return (
            ((orig - quant) ** 2).mean()
            / (orig ** 2).mean().clamp(min=1e-8)
        ).item()

    # Per-token (per-[i,j] pair) scale: [1, N, N, 1]
    scale_pt  = pot_scale(x.abs().amax(dim=-1, keepdim=True))
    # Per-tensor scale: scalar broadcast
    scale_ts  = pot_scale(x.abs().amax().view(1, 1, 1))

    e4m3_pt = relative_mse(x, quant_e4m3(x, scale_pt))
    e5m2_pt = relative_mse(x, quant_e5m2(x, scale_pt))
    e4m3_ts = relative_mse(x, quant_e4m3(x, scale_ts))
    e5m2_ts = relative_mse(x, quant_e5m2(x, scale_ts))

    format_suit = {
        "per_token_e4m3_rmse":  e4m3_pt,
        "per_token_e5m2_rmse":  e5m2_pt,
        "per_tensor_e4m3_rmse": e4m3_ts,
        "per_tensor_e5m2_rmse": e5m2_ts,
        "better_format":        "e4m3" if e4m3_pt < e5m2_pt else "e5m2",
        "per_tensor_viable":    bool(e4m3_ts < 0.05),
    }

    return {
        "stats":    stats,
        "kurtosis": kurtosis,
        "diagonal": diagonal,
        "format":   format_suit,
    }


def compute_comparison_metrics(z: torch.Tensor, g: torch.Tensor) -> dict:
    """
    Compute relative metrics comparing z (activation) to g (gradient).

    Parameters
    ----------
    z, g : torch.Tensor float32, [N, N, C]
    """
    z_scale = z.abs().amax(dim=-1)                 # [N, N]
    g_scale = g.abs().amax(dim=-1)                 # [N, N]
    ratio   = g_scale / z_scale.clamp(min=1e-10)  # [N, N]

    kurt_z = scipy_kurtosis(z.numpy().flatten(), fisher=True)
    kurt_g = scipy_kurtosis(g.numpy().flatten(), fisher=True)

    return {
        "scale_ratio_mean":   ratio.mean().item(),
        "scale_ratio_median": ratio.median().item(),
        "scale_ratio_std":    ratio.std().item(),
        # <<1 → gradient underflow risk in FP8
        # ~1  → same scale, same quantisation format works for both
        "kurtosis_ratio": float(kurt_g / max(abs(kurt_z), 0.1)),
        # >1 → gradients are harder to quantise than activations
    }


def analyze_block(z: torch.Tensor, g: torch.Tensor) -> dict:
    """Run all metrics for one Pairformer block."""
    act_metrics  = compute_all_metrics(z, "activation")
    grad_metrics = compute_all_metrics(g, "gradient")
    cmp_metrics  = compute_comparison_metrics(z, g)

    # Fill diagonal_r_diff
    cmp_metrics["diagonal_r_diff"] = (
        act_metrics["diagonal"]["pearson_r"]
        - grad_metrics["diagonal"]["pearson_r"]
    )

    return {
        "activation": act_metrics,
        "gradient":   grad_metrics,
        "comparison": cmp_metrics,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def _N_color_map(all_results: dict):
    """Return a dict N → colour from the viridis colormap."""
    Ns   = sorted(k for k, v in all_results.items() if not v.get("oom"))
    cmap = plt.cm.viridis
    return {N: cmap(i / max(len(Ns) - 1, 1)) for i, N in enumerate(Ns)}


def _setup_ax(ax, title: str, xlabel: str, ylabel: str):
    sns.despine(ax=ax)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)


def plot_kurtosis_act_vs_grad(all_results: dict, outdir: Path) -> None:
    """
    Line plot: X = block index, Y = kurtosis.
    Solid lines = activation, dashed = gradient.  One colour per N.
    Reference lines at y=0 (Gaussian), y=3 (Laplace), y=8 (LLM typical).
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = _N_color_map(all_results)

    for N, res in sorted(all_results.items()):
        if res.get("oom") or "blocks" not in res:
            continue
        blocks_data = res["blocks"]
        c = cmap[N]

        idxs_act, kurt_act = [], []
        idxs_grd, kurt_grd = [], []

        for idx in sorted(blocks_data.keys()):
            bd = blocks_data[idx]
            if "activation" in bd:
                idxs_act.append(idx)
                kurt_act.append(bd["activation"]["kurtosis"]["global"])
            if "gradient" in bd:
                idxs_grd.append(idx)
                kurt_grd.append(bd["gradient"]["kurtosis"]["global"])

        if idxs_act:
            ax.plot(idxs_act, kurt_act, color=c, lw=1.8,
                    label=f"N={N} act")
        if idxs_grd:
            ax.plot(idxs_grd, kurt_grd, color=c, lw=1.8, ls="--",
                    label=f"N={N} grad")

    for y, lbl, col in [
        (0,  "Gaussian (κ=0)",     "steelblue"),
        (3,  "Laplace (κ=3)",      "darkorange"),
        (8,  "Typical LLM (κ=8)", "crimson"),
    ]:
        ax.axhline(y, ls=":", lw=1, color=col, label=lbl)

    _setup_ax(ax, "Kurtosis: Activation vs Gradient Across Pairformer Blocks",
              "Block index", "Fisher kurtosis (excess)")
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(outdir / "kurtosis_act_vs_grad.png", dpi=300)
    plt.close(fig)
    print("  Saved kurtosis_act_vs_grad.png")


def plot_gradient_norm(all_results: dict, outdir: Path) -> None:
    """
    Line plot: X = block index, Y = L2 norm (log scale).
    Solid = activation norm, dashed = gradient norm.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = _N_color_map(all_results)

    for N, res in sorted(all_results.items()):
        if res.get("oom") or "blocks" not in res:
            continue
        c = cmap[N]
        blocks_data = res["blocks"]

        ia, na, ig, ng = [], [], [], []
        for idx in sorted(blocks_data.keys()):
            bd = blocks_data[idx]
            if "activation" in bd:
                ia.append(idx)
                na.append(bd["activation"]["stats"]["l2_norm"])
            if "gradient" in bd:
                ig.append(idx)
                ng.append(bd["gradient"]["stats"]["l2_norm"])

        if ia:
            ax.semilogy(ia, na, color=c, lw=1.8, label=f"N={N} act")
        if ig:
            ax.semilogy(ig, ng, color=c, lw=1.8, ls="--", label=f"N={N} grad")

    _setup_ax(ax, "L2 Norm Across Blocks: Activation vs Gradient",
              "Block index", "L2 norm (log scale)")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(outdir / "gradient_norm.png", dpi=300)
    plt.close(fig)
    print("  Saved gradient_norm.png")


def plot_dynamic_range(all_results: dict, outdir: Path) -> None:
    """
    Side-by-side bar chart: activation DR vs gradient DR.
    One group of bars per N value.
    """
    valid = {N: r for N, r in all_results.items()
             if not r.get("oom") and "blocks" in r}
    if not valid:
        return

    Ns    = sorted(valid.keys())
    act_dr, grd_dr = [], []

    for N in Ns:
        blocks_data = valid[N]["blocks"]
        a_vals = [
            v["activation"]["stats"]["dynamic_range"]
            for v in blocks_data.values()
            if "activation" in v
        ]
        g_vals = [
            v["gradient"]["stats"]["dynamic_range"]
            for v in blocks_data.values()
            if "gradient" in v
        ]
        act_dr.append(np.mean(a_vals) if a_vals else 0.0)
        grd_dr.append(np.mean(g_vals) if g_vals else 0.0)

    x = np.arange(len(Ns))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w / 2, act_dr, w, label="Activation DR", color="steelblue")
    ax.bar(x + w / 2, grd_dr, w, label="Gradient DR",   color="darkorange")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in Ns], rotation=30)
    _setup_ax(ax, "Dynamic Range: Activation vs Gradient (mean over profiled blocks)",
              "N (sequence length)", "Dynamic range (log scale)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "dynamic_range.png", dpi=300)
    plt.close(fig)
    print("  Saved dynamic_range.png")


def plot_diagonal_corr(all_results: dict, outdir: Path) -> None:
    """
    Line plot: X = block index, Y = Pearson r (diagonal structure).
    Solid = activation, dashed = gradient.
    Reference line at r=0.3 (meaningful threshold).
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = _N_color_map(all_results)

    for N, res in sorted(all_results.items()):
        if res.get("oom") or "blocks" not in res:
            continue
        c = cmap[N]
        blocks_data = res["blocks"]

        ia, ra, ig, rg = [], [], [], []
        for idx in sorted(blocks_data.keys()):
            bd = blocks_data[idx]
            if "activation" in bd:
                ia.append(idx)
                ra.append(bd["activation"]["diagonal"]["pearson_r"])
            if "gradient" in bd:
                ig.append(idx)
                rg.append(bd["gradient"]["diagonal"]["pearson_r"])

        if ia:
            ax.plot(ia, ra, color=c, lw=1.8, label=f"N={N} act")
        if ig:
            ax.plot(ig, rg, color=c, lw=1.8, ls="--", label=f"N={N} grad")

    ax.axhline(0.3, ls=":", lw=1, color="crimson", label="r=0.3 (meaningful)")
    ax.axhline(0.0, ls=":", lw=0.8, color="gray")

    _setup_ax(ax, "Diagonal Structure: Pearson r (Activation vs Gradient)",
              "Block index", "Pearson r (absmax vs proximity)")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(outdir / "diagonal_corr.png", dpi=300)
    plt.close(fig)
    print("  Saved diagonal_corr.png")


def plot_scale_ratio(all_results: dict, outdir: Path) -> None:
    """
    Line plot: X = block index, Y = mean(grad_scale / act_scale).
    Reference lines at y=1 (same scale) and y=0.01 (typical LLM).
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = _N_color_map(all_results)

    for N, res in sorted(all_results.items()):
        if res.get("oom") or "blocks" not in res:
            continue
        c = cmap[N]
        blocks_data = res["blocks"]

        idxs, ratios = [], []
        for idx in sorted(blocks_data.keys()):
            bd = blocks_data[idx]
            if "comparison" in bd:
                idxs.append(idx)
                ratios.append(bd["comparison"]["scale_ratio_mean"])

        if idxs:
            ax.semilogy(idxs, ratios, color=c, lw=1.8, marker="o",
                        ms=4, label=f"N={N}")

    ax.axhline(1.0,  ls=":",  lw=1.2, color="steelblue",   label="ratio=1 (same scale)")
    ax.axhline(0.01, ls="--", lw=1.0, color="darkorange",  label="ratio=0.01 (LLM typical)")

    _setup_ax(ax, "Gradient / Activation Scale Ratio Across Blocks",
              "Block index", "mean(grad_scale / act_scale)  [log scale]")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "scale_ratio.png", dpi=300)
    plt.close(fig)
    print("  Saved scale_ratio.png")


def plot_format_suitability(all_results: dict, outdir: Path) -> None:
    """
    2×2 heatmap: rows = (Activation, Gradient), cols = (E4M3, E5M2).
    Colour = mean relative MSE across all profiled blocks and N values.
    """
    # Aggregate
    matrix = np.zeros((2, 2))   # [act/grad, e4m3/e5m2]
    counts = np.zeros((2, 2))

    for res in all_results.values():
        if res.get("oom") or "blocks" not in res:
            continue
        for bd in res["blocks"].values():
            for row, tensor_key in enumerate(["activation", "gradient"]):
                if tensor_key not in bd:
                    continue
                fmt = bd[tensor_key]["format"]
                matrix[row, 0] += fmt["per_token_e4m3_rmse"]
                matrix[row, 1] += fmt["per_token_e5m2_rmse"]
                counts[row, :] += 1

    counts = np.maximum(counts, 1)
    matrix /= counts

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, vmin=0, vmax=0.1, cmap="RdYlGn_r")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["E4M3", "E5M2"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Activation", "Gradient"])
    plt.colorbar(im, ax=ax, label="Mean relative MSE")

    # Annotate cells
    for r in range(2):
        for c in range(2):
            val = matrix[r, c]
            color = "white" if val > 0.05 else "black"
            ax.text(c, r, f"{val:.4f}", ha="center", va="center",
                    fontsize=10, color=color)

    ax.set_title("FP8 Format Suitability Matrix\n"
                 "(green < 0.01 good, yellow 0.01–0.05 ok, red > 0.05 bad)",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outdir / "format_suitability.png", dpi=300)
    plt.close(fig)
    print("  Saved format_suitability.png")


def plot_memory_vs_seqlen(all_results: dict, outdir: Path,
                          gpu_mem_limit_gb: float = 80.0) -> None:
    """
    KEY PAPER FIGURE
    ────────────────
    X: sequence length N
    Y: peak memory in MB (log scale)

    Lines:
      • Blue  — measured BF16 training peak
      • Green — theoretical if pair repr stored as FP8 (×2 savings on NxN)
      • Red   — theoretical if pair repr stored as FP4 (×4 savings on NxN)

    Red X markers show OOM events and which phase (forward / backward).
    Horizontal dashed line = GPU memory limit.
    """
    Ns_ok   = sorted(k for k, v in all_results.items() if not v.get("oom") and "memory" in v)
    Ns_oom  = sorted(k for k, v in all_results.items() if v.get("oom"))

    if not Ns_ok:
        print("  [warn] No successful runs to plot memory vs seqlen")
        return

    peak_bf16 = []
    fp8_line  = []
    fp4_line  = []

    for N in Ns_ok:
        res = all_results[N]
        bf16_peak = res["memory"].get("peak_backward_MB") or res["memory"].get("peak_forward_MB", 0)
        token_z   = res.get("token_z", 128)

        pair_bf16_mb = N * N * token_z * 2 / 1e6
        pair_fp8_mb  = N * N * token_z * 1 / 1e6
        pair_fp4_mb  = N * N * token_z * 0.5 / 1e6

        savings_fp8 = max(pair_bf16_mb - pair_fp8_mb, 0.0)
        savings_fp4 = max(pair_bf16_mb - pair_fp4_mb, 0.0)

        peak_bf16.append(bf16_peak)
        fp8_line.append(max(bf16_peak - savings_fp8, 0.1))
        fp4_line.append(max(bf16_peak - savings_fp4, 0.1))

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.semilogy(Ns_ok, peak_bf16, "o-", color="royalblue",  lw=2, ms=6,
                label="Measured BF16 training peak")
    ax.semilogy(Ns_ok, fp8_line,  "s--", color="forestgreen", lw=2, ms=6,
                label="Theoretical: FP8 pair repr (×2 memory saving)")
    ax.semilogy(Ns_ok, fp4_line,  "^-.", color="darkorange",  lw=2, ms=6,
                label="Theoretical: FP4 pair repr (×4 memory saving)")

    # OOM markers
    for N in Ns_oom:
        phase = all_results[N].get("phase", "forward")
        ax.axvline(N, color="crimson", ls=":", lw=0.8)
        ax.scatter([N], [gpu_mem_limit_gb * 1e3 * 0.8], marker="X",
                   color="crimson", s=120, zorder=5)
        ax.text(N, gpu_mem_limit_gb * 1e3 * 0.85, f"OOM\n({phase})",
                ha="center", va="bottom", fontsize=7, color="crimson")

    # GPU memory limit
    ax.axhline(gpu_mem_limit_gb * 1e3, ls="--", lw=1.5, color="gray",
               label=f"GPU limit ({gpu_mem_limit_gb:.0f} GB)")

    # Annotation: MegaFold reference (1.23x)
    if Ns_ok and peak_bf16:
        ann_N = Ns_ok[-1]
        ann_y = peak_bf16[-1]
        ax.annotate("MegaFold: 1.23× reduction", xy=(ann_N, ann_y),
                    xytext=(ann_N, ann_y * 0.6),
                    arrowprops=dict(arrowstyle="->", color="gray"),
                    fontsize=8, ha="right", color="gray")

    _setup_ax(ax, "Training Peak Memory vs Sequence Length\n"
              "X marks OOM.  FP8 / FP4 pair representation extends trainable N",
              "N (sequence length)", "Peak memory (MB, log scale)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, which="both", axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(outdir / "memory_vs_seqlen.png", dpi=300)
    plt.close(fig)
    print("  Saved memory_vs_seqlen.png  ← KEY PAPER FIGURE")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _cuda_mem(device: str) -> float:
    return torch.cuda.memory_allocated(device) / 1e6 if device == "cuda" else 0.0


def _cuda_peak(device: str) -> float:
    return torch.cuda.max_memory_allocated(device) / 1e6 if device == "cuda" else 0.0


def _get_token_z(model) -> int:
    """Read token_z from model hparams."""
    return model.hparams.get("token_z", 128)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile Boltz-2 training activations and gradients "
                    "for MX/NV FP8/FP4 mixed-precision research."
    )
    parser.add_argument(
        "--boltz_src", required=True,
        help="Path to boltz/src directory (contains boltz/ package)",
    )
    parser.add_argument(
        "--ckpt_path", default="/root/.boltz/boltz2_aff.ckpt",
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Directory where results and plots will be saved",
    )
    parser.add_argument(
        "--device", default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run on (default: cuda)",
    )
    parser.add_argument(
        "--target_Ns", nargs="+", type=int,
        default=[64, 128, 256, 512, 768, 1024, 1536, 2048],
        help="Sequence lengths to profile",
    )
    parser.add_argument(
        "--gpu_mem_gb", type=float, default=80.0,
        help="GPU memory limit in GB (shown as dashed line in memory plot)",
    )
    parser.add_argument(
        "--discover", action="store_true",
        help="Print forward() signature and required feature keys, then exit",
    )
    args = parser.parse_args()

    # ── Load model ────────────────────────────────────────────────────────────
    sys.path.insert(0, args.boltz_src)
    from boltz.model.models.boltz2 import Boltz2  # noqa: PLC0415

    ckpt_path = args.ckpt_path
    print(f"\nLoading checkpoint: {ckpt_path}")

    # The checkpoint may have been saved with a slightly newer codebase.
    # We patch hparams in-memory to handle two known mismatches:
    #   1. pairformer_args missing 'v2': the checkpoint uses AttentionPairBiasV2
    #      (no norm_s) but the default v2=False would create the old AttentionPairBias.
    #   2. diffusion_process_args contains 'mse_rotational_alignment' not in
    #      the current AtomDiffusion signature (handled by **kwargs above).
    import tempfile  # noqa: PLC0415
    raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pa = raw_ckpt["hyper_parameters"].setdefault("pairformer_args", {})
    pa["v2"] = True   # checkpoint uses AttentionPairBiasV2 layers

    with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as tmp:
        torch.save(raw_ckpt, tmp.name)
        tmp_ckpt_path = tmp.name

    try:
        model = Boltz2.load_from_checkpoint(tmp_ckpt_path, map_location="cpu")
    finally:
        os.unlink(tmp_ckpt_path)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA not available — falling back to CPU")
        device = "cpu"

    model = model.to(device)
    # The affinity checkpoint has most trunk parameters frozen (requires_grad=False).
    # For gradient profiling we need gradients through all Pairformer layers.
    model.requires_grad_(True)
    model.train()   # TRAINING MODE — critical for gradient profiling
    print(f"Model on {device}, in training mode (all parameters unfrozen).")

    # ── Step 1: discover (optional) ───────────────────────────────────────────
    if args.discover:
        discover_batch_format(model, args.boltz_src)
        return

    # ── Setup output dir ──────────────────────────────────────────────────────
    outdir = Path(args.output_dir) / "training_profile"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {outdir}")

    # ── Determine blocks to profile ───────────────────────────────────────────
    pf_mod = model.pairformer_module
    if hasattr(pf_mod, "_orig_mod"):
        pf_mod = pf_mod._orig_mod
    n_blocks = len(pf_mod.layers)

    # Profile ~12 evenly spaced blocks (first, last, and every ~4 in between)
    blocks_to_profile = sorted(set(
        [0, n_blocks - 1]
        + list(range(0, n_blocks, max(1, n_blocks // 10)))
    ))
    print(f"\nPairformer has {n_blocks} blocks.")
    print(f"Profiling blocks: {blocks_to_profile}\n")

    # ── Main loop ─────────────────────────────────────────────────────────────
    sns.set_style("whitegrid")

    all_results: dict = {}
    json_path = outdir / "summary_training.json"

    for N in args.target_Ns:
        result = profile_one_N(model, N, device, blocks_to_profile)
        all_results[N] = result

        # Serialise — convert int keys to str for JSON
        _save_results(all_results, json_path)

        if result.get("oom") and result.get("phase") == "forward":
            print(f"\nStopping: OOM on forward pass at N={N}. "
                  "Larger N values will also OOM.")
            break

    print(f"\n=== Generating plots (output: {outdir}) ===")
    plot_kurtosis_act_vs_grad(all_results, outdir)
    plot_gradient_norm(all_results, outdir)
    plot_dynamic_range(all_results, outdir)
    plot_diagonal_corr(all_results, outdir)
    plot_scale_ratio(all_results, outdir)
    plot_format_suitability(all_results, outdir)
    plot_memory_vs_seqlen(all_results, outdir, gpu_mem_limit_gb=args.gpu_mem_gb)

    print(f"\nDone.  Results saved to {outdir}")


def _save_results(all_results: dict, path: Path) -> None:
    """Serialise all_results to JSON (converts int keys → str, handles non-serialisable values)."""
    def _to_json(obj):
        if isinstance(obj, dict):
            return {str(k): _to_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_json(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return str(obj)
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        return obj

    with open(path, "w") as fh:
        json.dump(_to_json(all_results), fh, indent=2)


if __name__ == "__main__":
    main()
