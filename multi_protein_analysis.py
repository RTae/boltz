#!/usr/bin/env python3
"""
multi_protein_analysis.py – Cross-protein Pairformer activation study.

Runs profile_pairformer.py's profiling pipeline across up to 5 small example
proteins, then aggregates the results to answer:

  "Are the dominant Pairformer pair-tensor channels consistent across proteins?"

Usage:
    python multi_protein_analysis.py [--cache ~/.boltz] [--out_dir ./multi_protein_results]
                                     [--cpu] [--no_bf16] [--use_msa_server]

Outputs (written to --out_dir):
    <name>/pairformer_stats.json          per-protein hook statistics
    cross_protein_per_block_max.png       abs_max curves for all proteins, one per layer
    cross_protein_top_channels.png        top-5 dominant channels per protein (bar chart)
"""

import argparse
import importlib.util
import json
import multiprocessing
import os
import sys
import warnings
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Import profile_pairformer from the same directory
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
_pp = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_pp)  # type: ignore[union-attr]

# Re-export symbols used here
_tensor_stats = _pp._tensor_stats  # noqa: SLF001
attach_hooks = _pp.attach_hooks
remove_hooks = _pp.remove_hooks
run_trunk_forward = _pp.run_trunk_forward
generate_plots = _pp.generate_plots
print_summary = _pp.print_summary
_json_safe = _pp._json_safe  # noqa: SLF001
SUBMODULE_NAMES = _pp.SUBMODULE_NAMES

# Boltz internals (already imported by profile_pairformer, pull through sys.modules)
from boltz.main import (  # noqa: E402
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
# Input catalogue
# ---------------------------------------------------------------------------

#  Each entry describes one profiling target.
#  needs_msa_server=True  → requires --use_msa_server to be fetched from ColabFold.
#  needs_msa_server=False → has either msa:empty or a local .a3m file; works offline.

INPUTS: List[Dict[str, Any]] = [
    {
        "name": "prot_no_msa",
        "yaml": "examples/prot_no_msa.yaml",
        "needs_msa_server": False,
        "description": "113-residue protein (empty MSA – offline)",
    },
    {
        "name": "prot_custom_msa",
        "yaml": "examples/prot_custom_msa.yaml",
        "needs_msa_server": False,
        "description": "113-residue protein (local .a3m MSA file)",
    },
    {
        "name": "cyclic_prot",
        "yaml": "examples/cyclic_prot.yaml",
        "needs_msa_server": True,
        "description": "13-residue cyclic peptide",
    },
    {
        "name": "prot",
        "yaml": "examples/prot.yaml",
        "needs_msa_server": True,
        "description": "113-residue protein (ColabFold MSA)",
    },
    {
        "name": "multimer",
        "yaml": "examples/multimer.yaml",
        "needs_msa_server": True,
        "description": "Two-chain multimer ~113+113 residues",
    },
]

# ---------------------------------------------------------------------------
# Channel-hit analysis
# ---------------------------------------------------------------------------

TOP_K_PER_LAYER = 5    # how many channels to call "top" per layer
REPORT_N_CHANNELS = 3  # how many top channels to report per protein


def analyse_channels(
    storage: List[Dict],
    top_k_per_layer: int = TOP_K_PER_LAYER,
    report_n: int = REPORT_N_CHANNELS,
) -> Dict[str, Any]:
    """Find which z-pair channels consistently land in the top-k per-layer std.

    Returns
    -------
    dict with keys:
        num_layers     : int
        total_channels : int
        top_channels   : list of (channel_idx, hit_pct) sorted descending
        all_hit_pct    : dict channel_idx -> hit_pct
    """
    layer_z_recs = [r for r in storage if r.get("op") == "layer_output_z"]
    if not layer_z_recs:
        return {"num_layers": 0, "total_channels": 0, "top_channels": [], "all_hit_pct": {}}

    num_layers = len(layer_z_recs)
    channel_hits: Counter = Counter()
    total_channels = 0

    for rec in layer_z_recs:
        ch_std: Optional[List[float]] = rec.get("per_channel_std")
        if not ch_std:
            continue
        total_channels = max(total_channels, len(ch_std))
        ranked = sorted(range(len(ch_std)), key=lambda i: ch_std[i], reverse=True)
        for ch_idx in ranked[:top_k_per_layer]:
            channel_hits[ch_idx] += 1

    all_hit_pct: Dict[int, float] = {
        ch: count / num_layers * 100.0
        for ch, count in channel_hits.items()
    }

    top_channels = sorted(all_hit_pct.items(), key=lambda x: x[1], reverse=True)[:report_n]

    return {
        "num_layers": num_layers,
        "total_channels": total_channels,
        "top_channels": list(top_channels),
        "all_hit_pct": {str(k): v for k, v in all_hit_pct.items()},
    }


# ---------------------------------------------------------------------------
# Device / batch helpers
# ---------------------------------------------------------------------------

def _to_device(obj, dev: torch.device):
    if isinstance(obj, torch.Tensor):
        return obj.to(dev)
    if isinstance(obj, dict):
        return {k: _to_device(v, dev) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_device(v, dev) for v in obj]
    return obj


def _load_model(cache: Path, checkpoint: Optional[str]) -> Boltz1:
    ckpt = checkpoint or str(cache / "boltz1_conf.ckpt")
    print(f"Loading Boltz-1 checkpoint: {ckpt} …")

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
        ckpt,
        strict=True,
        predict_args=predict_args,
        map_location="cpu",
        diffusion_process_args=asdict(BoltzDiffusionParams()),
        ema=False,
        use_kernels=False,
        pairformer_args=asdict(PairformerArgs(
            num_blocks=48,
            num_heads=16,
            dropout=0.0,
            activation_checkpointing=False,
        )),
        msa_args=asdict(MSAModuleArgs(
            subsample_msa=False,
            use_paired_feature=False,
        )),
        steering_args=asdict(BoltzSteeringParams()),
    )
    model.eval()
    model.use_kernels = False
    return model


def _process_one(
    yaml_path: Path,
    work_dir: Path,
    cache: Path,
    use_msa_server: bool,
) -> Optional[Path]:
    """Process a YAML input into boltz1-format processed data under *work_dir*.

    Returns the path to the ``processed/`` subdirectory on success, else None.
    """
    ccd_path = cache / "ccd.pkl"
    mol_dir = cache / "mols"

    processed_dir = work_dir / "processed"

    # Re-use existing processed data when present
    if (processed_dir / "manifest.json").exists():
        print(f"  [cache hit] Reusing processed data at {processed_dir}")
        return processed_dir

    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        data_list = check_inputs(yaml_path)
    except Exception as exc:
        print(f"  ERROR during check_inputs: {exc}", file=sys.stderr)
        return None

    manifest = process_inputs(
        data=data_list,
        out_dir=work_dir,
        ccd_path=ccd_path,
        mol_dir=mol_dir,
        msa_server_url="https://api.colabfold.com",
        msa_pairing_strategy="greedy",
        use_msa_server=use_msa_server,
        boltz2=False,
        preprocessing_threads=min(4, multiprocessing.cpu_count()),
    )

    if not manifest or not manifest.records:
        manifest_path = processed_dir / "manifest.json"
        if manifest_path.exists():
            manifest = Manifest.load(manifest_path)

    if not manifest or not manifest.records:
        print(
            f"  ERROR: manifest has no records for {yaml_path.name}. "
            "Check that the input is valid and MSA is available.",
            file=sys.stderr,
        )
        return None

    return processed_dir


def _profile_one(
    name: str,
    processed_dir: Path,
    model: Boltz1,
    device: torch.device,
    use_bf16: bool,
    out_dir: Path,
) -> Optional[List[Dict]]:
    """Run a trunk forward pass with hooks, save JSON, return storage list."""
    manifest_path = processed_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"  ERROR: manifest.json missing in {processed_dir}", file=sys.stderr)
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

    data_module = BoltzInferenceDataModule(
        manifest=processed.manifest,
        target_dir=processed.targets_dir,
        msa_dir=processed.msa_dir,
        num_workers=0,
        constraints_dir=processed.constraints_dir,
    )

    try:
        batch = next(iter(data_module.predict_dataloader()))
    except StopIteration:
        print(f"  ERROR: DataLoader is empty for {name}. The processed data may be invalid.",
              file=sys.stderr)
        return None

    feats = _to_device(batch, device)
    token_count = int(feats["token_pad_mask"].sum().item())
    print(f"  Token count: {token_count}")

    # Locate the pairformer module
    if getattr(model, "is_pairformer_compiled", False):
        pairformer_module = model.pairformer_module._orig_mod  # noqa: SLF001
    else:
        pairformer_module = model.pairformer_module

    handles, storage = attach_hooks(pairformer_module)

    try:
        run_trunk_forward(model, feats, device, use_bf16)
    except Exception as exc:
        print(f"  ERROR during forward pass: {exc}", file=sys.stderr)
        remove_hooks(handles)
        return None

    remove_hooks(handles)
    print(f"  Collected {len(storage)} hook records.")

    # Save stats
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "pairformer_stats.json"
    out_records: Dict[str, Any] = {
        "protein": name,
        "token_count": token_count,
        "num_layers": len([r for r in storage if r.get("op") == "layer_output_z"]),
        "device": str(device),
        "bf16_autocast": use_bf16,
        "records": _json_safe(storage),
    }
    with open(json_path, "w") as f:
        json.dump(out_records, f, indent=2)
    print(f"  Saved {json_path}")

    # Per-protein plots
    generate_plots(storage, out_dir)

    return storage


# ---------------------------------------------------------------------------
# Cross-protein aggregation plots
# ---------------------------------------------------------------------------

def _plot_cross_protein_per_block_max(
    all_results: Dict[str, List[Dict]],
    out_path: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available – skipping cross-protein per-block plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    for name, storage in all_results.items():
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

    ax.set_xlabel("Layer index")
    ax.set_ylabel("abs_max of pair tensor z")
    ax.set_title("Per-block abs_max of pair tensor z — all proteins")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    import matplotlib.pyplot as plt2
    plt2.close(fig)
    print(f"Saved {out_path}")


def _plot_cross_protein_top_channels(
    channel_analyses: Dict[str, Dict],
    out_path: Path,
    top_n: int = 5,
) -> None:
    """Bar chart showing top-N dominant channels (by hit%) for each protein."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available – skipping cross-protein channel plot.")
        return

    proteins = list(channel_analyses.keys())
    n_proteins = len(proteins)
    if n_proteins == 0:
        return

    # Collect per-protein top-N (channel_idx, hit_pct)
    protein_top: Dict[str, List[Tuple[int, float]]] = {}
    all_top_channels = set()

    for protein, analysis in channel_analyses.items():
        all_hit = {int(k): v for k, v in analysis.get("all_hit_pct", {}).items()}
        top = sorted(all_hit.items(), key=lambda x: x[1], reverse=True)[:top_n]
        protein_top[protein] = top
        for ch, _ in top:
            all_top_channels.add(ch)

    # Union of top channels, sorted
    channels = sorted(all_top_channels)
    n_channels = len(channels)
    if n_channels == 0:
        return

    ch_to_idx = {ch: i for i, ch in enumerate(channels)}
    bar_width = 0.8 / n_proteins
    x = np.arange(n_channels)
    colors = plt.cm.tab10(np.linspace(0, 0.9, n_proteins))

    fig, ax = plt.subplots(figsize=(max(12, n_channels * 0.8), 6))

    for p_i, (protein, analysis) in enumerate(channel_analyses.items()):
        all_hit = {int(k): v for k, v in analysis.get("all_hit_pct", {}).items()}
        heights = [all_hit.get(ch, 0.0) for ch in channels]
        offsets = x + p_i * bar_width - (n_proteins - 1) * bar_width / 2
        ax.bar(offsets, heights, width=bar_width * 0.9, label=protein, color=colors[p_i], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"ch {ch}" for ch in channels], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Channel index")
    ax.set_ylabel(f"Hit% (layers where channel is in top-{TOP_K_PER_LAYER})")
    ax.set_title(f"Top-{top_n} dominant pair-tensor channels per protein")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    import matplotlib.pyplot as plt2
    plt2.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Verdict printer
# ---------------------------------------------------------------------------

def _print_verdict(channel_analyses: Dict[str, Dict]) -> None:
    if not channel_analyses:
        print("No results to compare.")
        return

    print("\n" + "=" * 70)
    print("CROSS-PROTEIN CHANNEL DOMINANCE ANALYSIS")
    print("=" * 70)

    # Table header
    header_proteins = list(channel_analyses.keys())
    print(f"\n{'Protein':<22}  {'Top-3 channels (hit%)'}")
    print("-" * 70)

    per_protein_top1: Dict[str, int] = {}

    for protein, analysis in channel_analyses.items():
        top = analysis.get("top_channels", [])
        if not top:
            print(f"  {protein:<20}  (no data)")
            continue
        parts = [f"ch{ch}={pct:.0f}%" for ch, pct in top]
        print(f"  {protein:<20}  {', '.join(parts)}")
        if top:
            per_protein_top1[protein] = top[0][0]

    print()

    # Verdict
    if not per_protein_top1:
        print("VERDICT: insufficient data.")
        return

    top1_channels = list(per_protein_top1.values())
    unique_top1 = set(top1_channels)

    if len(unique_top1) == 1:
        print(f"VERDICT: CONSISTENT — channel {list(unique_top1)[0]} "
              f"is the #1 dominant channel across ALL proteins.")
    else:
        # Check if there's a majority
        ctr = Counter(top1_channels)
        majority_ch, majority_cnt = ctr.most_common(1)[0]
        if majority_cnt / len(top1_channels) >= 0.6:
            minority = [p for p, ch in per_protein_top1.items() if ch != majority_ch]
            print(
                f"VERDICT: MOSTLY CONSISTENT — channel {majority_ch} dominates "
                f"in {majority_cnt}/{len(top1_channels)} proteins.\n"
                f"  Exceptions: {', '.join(minority)}"
            )
        else:
            per_str = ", ".join(f"{p}→ch{ch}" for p, ch in per_protein_top1.items())
            print(f"VERDICT: VARIES — dominant channels differ per protein: {per_str}")

    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross-protein Pairformer activation study."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./multi_protein_results",
        help="Root output directory (default: ./multi_protein_results).",
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
        help="Optional path to a Boltz-1 checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--use_msa_server",
        action="store_true",
        help=(
            "Use the MMSeqs2 ColabFold server for MSA generation. "
            "Required for prot.yaml, cyclic_prot.yaml, multimer.yaml."
        ),
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference.",
    )
    parser.add_argument(
        "--no_bf16",
        action="store_true",
        help="Disable bfloat16 autocast (run in float32).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    warnings.filterwarnings("ignore", ".*Tensor Cores.*")
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("highest")

    cache = Path(args.cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    repo_root = _SCRIPT_DIR  # script lives at boltz repo root

    # ------------------------------------------------------------------
    # 1. Download weights (once)
    # ------------------------------------------------------------------
    print("Checking / downloading Boltz-1 model weights …")
    download_boltz1(cache)

    # ------------------------------------------------------------------
    # 2. Decide which inputs to run
    # ------------------------------------------------------------------
    targets = []
    skipped = []
    for inp in INPUTS:
        yaml_path = (repo_root / inp["yaml"]).resolve()
        if not yaml_path.exists():
            print(f"[SKIP] {inp['name']}: YAML not found at {yaml_path}")
            skipped.append(inp["name"])
            continue
        if inp["needs_msa_server"] and not args.use_msa_server:
            print(
                f"[SKIP] {inp['name']}: needs MSA server; pass --use_msa_server to include it."
            )
            skipped.append(inp["name"])
            continue
        targets.append({**inp, "yaml_path": yaml_path})

    if not targets:
        print(
            "\nNo targets to run. "
            "Pass --use_msa_server to also profile proteins that require MSA fetching.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\nWill profile {len(targets)} protein(s): "
          f"{', '.join(t['name'] for t in targets)}")
    if skipped:
        print(f"Skipped: {', '.join(skipped)}")
    print()

    # ------------------------------------------------------------------
    # 3. Load model (once, shared across all proteins)
    # ------------------------------------------------------------------
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    use_bf16 = (not args.no_bf16) and (device.type == "cuda")
    print(f"Device: {device}  |  bfloat16 autocast: {use_bf16}\n")

    model = _load_model(cache, args.checkpoint)
    model = model.to(device)

    # ------------------------------------------------------------------
    # 4. Process + profile each protein
    # ------------------------------------------------------------------
    all_storage: Dict[str, List[Dict]] = {}
    channel_analyses: Dict[str, Dict] = {}

    for t in targets:
        name = t["name"]
        print(f"\n{'─'*60}")
        print(f"  [{name}]  {t['description']}")
        print(f"{'─'*60}")

        work_dir = out_root / name / "processed_input"
        protein_out_dir = out_root / name

        # Step A: process input
        print(f"  Processing {t['yaml_path'].name} …")
        processed_dir = _process_one(
            yaml_path=t["yaml_path"],
            work_dir=work_dir,
            cache=cache,
            use_msa_server=t["needs_msa_server"] and args.use_msa_server,
        )
        if processed_dir is None:
            print(f"  [FAILED] Skipping {name}.")
            continue

        # Step B: run hooks
        print(f"  Running trunk forward pass …")
        storage = _profile_one(
            name=name,
            processed_dir=processed_dir,
            model=model,
            device=device,
            use_bf16=use_bf16,
            out_dir=protein_out_dir,
        )
        if storage is None:
            print(f"  [FAILED] Skipping {name}.")
            continue

        all_storage[name] = storage

        # Step C: channel analysis
        analysis = analyse_channels(storage)
        channel_analyses[name] = analysis

        # Print per-protein summary
        print_summary(storage)

    # ------------------------------------------------------------------
    # 5. Cross-protein aggregation
    # ------------------------------------------------------------------
    if len(all_storage) < 2:
        print("\nFewer than 2 proteins profiled successfully — cross-protein plots skipped.")
        _print_verdict(channel_analyses)
        return

    print("\n\nGenerating cross-protein comparison plots …")

    _plot_cross_protein_per_block_max(
        all_results=all_storage,
        out_path=out_root / "cross_protein_per_block_max.png",
    )

    _plot_cross_protein_top_channels(
        channel_analyses=channel_analyses,
        out_path=out_root / "cross_protein_top_channels.png",
        top_n=5,
    )

    # ------------------------------------------------------------------
    # 6. Print verdict
    # ------------------------------------------------------------------
    _print_verdict(channel_analyses)

    print(f"All outputs written to: {out_root.resolve()}")


if __name__ == "__main__":
    main()
