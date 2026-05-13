"""
analyze_and_plot.py
===================
Cross-input analysis and visualization for Pairformer MXFP8 profiling.

Reads activations.jsonl + meta.json from one or more boltz_results_stats/{input}/
directories (produced by capture_pairformer_stats.py) and generates:

  boltz_results_plots/
    abs_max_by_layer.png
    kurtosis_by_layer.png
    frac_gt6_by_layer.png
    symmetry_breaks.png
    channel_outlier_severity.png
    opm_distribution_{input_name}.png   (one per input with OPM histogram data)
    pairformer_layer0_distributions.png
    cross_input_summary.txt
    cross_input_summary.csv

Usage:
    python analyze_and_plot.py \\
        --input_dirs boltz_results_stats/prot boltz_results_stats/homodimer \\
        --output_dir boltz_results_plots

    python analyze_and_plot.py \\
        --input_glob "boltz_results_stats/*" \\
        --output_dir boltz_results_plots
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger("analyze_plots")

# ---------------------------------------------------------------------------
# Color palette — consistent assignment by input name (sorted alphabetically)
# ---------------------------------------------------------------------------

_TAB10 = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def assign_colors(input_names: List[str]) -> Dict[str, str]:
    sorted_names = sorted(input_names)
    return {name: _TAB10[i % len(_TAB10)] for i, name in enumerate(sorted_names)}


# Operator display order and grouping
_PAIRFORMER_OPS = [
    "TriangleMultiplicationOutgoing",
    "TriangleMultiplicationIncoming",
    "TriangleAttentionStartingNode",
    "TriangleAttentionEndingNode",
    "AttentionPairBias",
]
_MSA_OPS = [
    "OuterProductMean",
    "PairWeightedAveraging",
]
_OP_SHORT = {
    "TriangleMultiplicationOutgoing": "TriMulOut",
    "TriangleMultiplicationIncoming": "TriMulIn",
    "TriangleAttentionStartingNode": "TriAttStart",
    "TriangleAttentionEndingNode": "TriAttEnd",
    "AttentionPairBias": "AttnPairBias",
    "OuterProductMean": "OPM",
    "PairWeightedAveraging": "PairWtAvg",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_input_dir(input_dir: Path) -> Tuple[List[dict], dict]:
    """Load activations.jsonl and meta.json from an input directory."""
    records: List[dict] = []
    act_file = input_dir / "activations.jsonl"
    if act_file.exists():
        with act_file.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    else:
        log.warning("No activations.jsonl in %s", input_dir)

    meta: dict = {}
    meta_file = input_dir / "meta.json"
    if meta_file.exists():
        with meta_file.open() as fh:
            try:
                meta = json.load(fh)
            except json.JSONDecodeError:
                pass

    return records, meta


def collect_all_inputs(
    input_dirs: List[Path],
) -> Dict[str, Tuple[List[dict], dict]]:
    """Return {input_name: (records, meta)} for each directory."""
    result: Dict[str, Tuple[List[dict], dict]] = {}
    for d in input_dirs:
        d = Path(d)
        if not d.is_dir():
            log.warning("Skipping non-directory: %s", d)
            continue
        name = d.name
        records, meta = load_input_dir(d)
        if not records:
            log.warning("No records loaded from %s — skipping", d)
            continue
        result[name] = (records, meta)
        log.info("Loaded %d records from %s (seqlen=%s)", len(records), name, meta.get("seqlen", "?"))
    return result


def flat_records(
    collected: Dict[str, Tuple[List[dict], dict]],
) -> List[dict]:
    """Return a flat list of records, each annotated with input_name and seqlen."""
    out: List[dict] = []
    for input_name, (records, meta) in collected.items():
        seqlen = meta.get("seqlen", 0)
        for r in records:
            r = dict(r)
            r["input_name"] = input_name
            r["seqlen_from_meta"] = seqlen
            out.append(r)
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(val: Any, default: float = float("nan")) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _layer_key(r: dict) -> int:
    """Return the relevant layer index for a record."""
    li = r.get("layer_index", -1)
    msa = r.get("msa_layer_index", -1)
    return li if li >= 0 else msa


def _op_display(op: str) -> str:
    return _OP_SHORT.get(op, op)


# ---------------------------------------------------------------------------
# Plot 1-3: Metric by layer (line plots)
# ---------------------------------------------------------------------------

def _plot_metric_by_layer(
    collected: Dict[str, Tuple[List[dict], dict]],
    metric_key: str,
    ylabel: str,
    title: str,
    output_file: Path,
    log_scale: bool = False,
    role_filter: Optional[str] = None,          # None = all roles
) -> None:
    """
    Line plots: X = layer index, Y = metric_key.
    Layout: subplots per operator type (Pairformer and MSA).
    One colored line per input. Separate linestyles per role if role_filter is None.
    """
    colors = assign_colors(list(collected.keys()))

    all_ops = _PAIRFORMER_OPS + _MSA_OPS
    n_ops = len(all_ops)
    ncols = 3
    nrows = math.ceil(n_ops / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6 * ncols, 4 * nrows),
        squeeze=False,
    )
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    # Pre-group data: {op_name: {input_name: {role: {layer: [values]}}}}
    grouped: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for input_name, (records, meta) in collected.items():
        for r in records:
            op = r.get("operator_name", "")
            if op not in all_ops:
                continue
            role = r.get("tensor_role", "")
            if role_filter is not None and role != role_filter:
                continue
            li = _layer_key(r)
            val = _safe_float(r.get(metric_key))
            if not math.isnan(val) and val >= 0:
                grouped[op][input_name][role][li].append(val)

    _ROLE_STYLES = {
        "input": "-",
        "output": "--",
        "pre_softmax_score": ":",
        "tri_bias_output": "-.",
    }

    for idx, op in enumerate(all_ops):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        ax.set_title(_op_display(op), fontsize=10)
        ax.set_xlabel("Layer index")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if log_scale:
            ax.set_yscale("log")

        legend_handles = []
        for input_name, (records, meta) in collected.items():
            color = colors[input_name]
            op_data = grouped[op][input_name]
            for role, layer_vals in sorted(op_data.items()):
                ls = _ROLE_STYLES.get(role, "-")
                xs = sorted(layer_vals.keys())
                ys = [float(np.mean(layer_vals[x])) for x in xs]
                if not xs:
                    continue
                label = f"{input_name} [{role}]" if role_filter is None else input_name
                line, = ax.plot(xs, ys, color=color, linestyle=ls,
                                marker="o", markersize=3, linewidth=1.4,
                                label=label)
                legend_handles.append(line)

        if legend_handles:
            ax.legend(fontsize=7, ncol=1, loc="upper right",
                      framealpha=0.7)

    # Hide unused axes
    for idx in range(n_ops, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", output_file)


# ---------------------------------------------------------------------------
# Plot 4: Symmetry breaks
# ---------------------------------------------------------------------------

def plot_symmetry_breaks(
    collected: Dict[str, Tuple[List[dict], dict]],
    output_file: Path,
) -> None:
    """Bar plot of symmetric_outlier_correlation per (op, layer), one panel per input."""
    names = list(collected.keys())
    if not names:
        return

    ncols = min(len(names), 3)
    nrows = math.ceil(len(names) / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(7 * ncols, 5 * nrows),
                             squeeze=False)
    fig.suptitle(
        "Symmetry Breaks: symmetric_outlier_correlation\n"
        "(low = outlier pairs not mirrored across pair diagonal)",
        fontsize=12, fontweight="bold",
    )
    colors = assign_colors(names)

    for panel_idx, input_name in enumerate(names):
        row, col = divmod(panel_idx, ncols)
        ax = axes[row][col]
        ax.set_title(input_name, fontsize=10)
        ax.set_xlabel("(operator, layer)")
        ax.set_ylabel("symmetric_outlier_correlation")
        ax.grid(True, alpha=0.3, axis="y")

        records, _ = collected[input_name]
        sym_recs = [
            r for r in records
            if _safe_float(r.get("symmetric_outlier_correlation", -1)) >= 0
        ]
        if not sym_recs:
            ax.text(0.5, 0.5, "No pair-tensor records", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        # Sort by ascending correlation (worst first)
        sym_recs = sorted(sym_recs, key=lambda r: _safe_float(r.get("symmetric_outlier_correlation", 1.0)))
        sym_recs = sym_recs[:20]  # top-20 worst

        labels = [
            f"{_op_display(r['operator_name'])}\nl{_layer_key(r)}"
            for r in sym_recs
        ]
        values = [_safe_float(r.get("symmetric_outlier_correlation", 0)) for r in sym_recs]
        bar_colors = [colors[input_name]] * len(labels)
        bars = ax.bar(range(len(labels)), values, color=bar_colors, edgecolor="white")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=6, rotation=70, ha="right")
        ax.set_ylim(0, 1)
        # Add value labels on bars
        for bar, val in zip(bars, values):
            if val < 0.5:
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=6)

    for idx in range(len(names), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", output_file)


# ---------------------------------------------------------------------------
# Plot 5: Channel outlier severity
# ---------------------------------------------------------------------------

def plot_channel_outlier_severity(
    collected: Dict[str, Tuple[List[dict], dict]],
    output_file: Path,
) -> None:
    """Bar plot of channel_max_to_median_ratio per (op, layer), one panel per input."""
    names = list(collected.keys())
    if not names:
        return

    ncols = min(len(names), 3)
    nrows = math.ceil(len(names) / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(7 * ncols, 5 * nrows),
                             squeeze=False)
    fig.suptitle(
        "Channel Outlier Severity: channel_max_to_median_ratio\n"
        "(SmoothQuant indicator — ratio >> 1 requires per-channel scaling)",
        fontsize=12, fontweight="bold",
    )
    colors = assign_colors(names)

    for panel_idx, input_name in enumerate(names):
        row, col = divmod(panel_idx, ncols)
        ax = axes[row][col]
        ax.set_title(input_name, fontsize=10)
        ax.set_xlabel("(operator, layer)")
        ax.set_ylabel("channel_max_to_median_ratio")
        ax.grid(True, alpha=0.3, axis="y")

        records, _ = collected[input_name]
        recs = sorted(records,
                      key=lambda r: _safe_float(r.get("channel_max_to_median_ratio", 0)),
                      reverse=True)
        recs = recs[:20]  # top-20 worst

        if not recs:
            ax.text(0.5, 0.5, "No records", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        labels = [
            f"{_op_display(r['operator_name'])}\nl{_layer_key(r)}\n[{r.get('tensor_role','?')}]"
            for r in recs
        ]
        values = [_safe_float(r.get("channel_max_to_median_ratio", 0)) for r in recs]
        ax.bar(range(len(labels)), values, color=colors[input_name], edgecolor="white")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=6, rotation=70, ha="right")

    for idx in range(len(names), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", output_file)


# ---------------------------------------------------------------------------
# Plot 6: OPM distribution histograms
# ---------------------------------------------------------------------------

def plot_opm_distributions(
    collected: Dict[str, Tuple[List[dict], dict]],
    output_dir: Path,
) -> None:
    """
    For each input that has OPM histogram data, produce a histogram plot.
    Log-scale Y. Vertical lines at E4M3 max (448) and 6.
    """
    for input_name, (records, meta) in collected.items():
        # Find OPM input records with histogram data
        opm_recs = [
            r for r in records
            if r.get("operator_name") == "OuterProductMean"
            and r.get("tensor_role") == "input"
            and r.get("histogram_bins") is not None
        ]
        if not opm_recs:
            log.info("No OPM histogram data for %s — skipping opm_distribution plot", input_name)
            continue

        ncols = min(len(opm_recs), 3)
        nrows = math.ceil(len(opm_recs) / ncols)
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(6 * ncols, 4 * nrows),
                                 squeeze=False)
        fig.suptitle(
            f"OuterProductMean input distribution — {input_name}",
            fontsize=12, fontweight="bold",
        )

        for panel_idx, r in enumerate(opm_recs):
            row, col = divmod(panel_idx, ncols)
            ax = axes[row][col]

            bins = r["histogram_bins"]     # list of N+1 bin edges
            counts = r["histogram_counts"] # list of N bin counts

            # Bin centres
            centres = [(bins[i] + bins[i + 1]) / 2.0 for i in range(len(counts))]
            widths  = [bins[i + 1] - bins[i] for i in range(len(counts))]

            ax.bar(centres, counts, width=widths, color="#1f77b4",
                   edgecolor="none", alpha=0.75)
            ax.set_yscale("log")
            ax.set_xlabel("Activation value")
            ax.set_ylabel("Count (log scale)")
            msa_li = r.get("msa_layer_index", -1)
            li = r.get("layer_index", -1)
            loc_str = f"msa_layer={msa_li}" if msa_li >= 0 else f"layer={li}"
            ax.set_title(f"OPM input  {loc_str}\nabs_max={r.get('abs_max', 0):.1f}",
                         fontsize=9)
            ax.grid(True, alpha=0.3)

            # Reference lines
            ax.axvline(448,  color="red",    linestyle="--", linewidth=1.2,
                       label="E4M3 max (448)")
            ax.axvline(-448, color="red",    linestyle="--", linewidth=1.2)
            ax.axvline(6,    color="orange", linestyle=":",  linewidth=1.2,
                       label="|x|=6 threshold")
            ax.axvline(-6,   color="orange", linestyle=":",  linewidth=1.2)
            ax.legend(fontsize=7)

        for idx in range(len(opm_recs), nrows * ncols):
            row2, col2 = divmod(idx, ncols)
            axes[row2][col2].set_visible(False)

        fig.tight_layout()
        out_file = output_dir / f"opm_distribution_{input_name}.png"
        fig.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved %s", out_file)


# ---------------------------------------------------------------------------
# Plot 7: Layer-0 distributions (TriMul output + AttentionPairBias input)
# ---------------------------------------------------------------------------

def plot_layer0_distributions(
    collected: Dict[str, Tuple[List[dict], dict]],
    output_file: Path,
) -> None:
    """
    Histograms of TriMulOut output and AttentionPairBias input at layer 0,
    one colored line per input (using histogram_bins data if available,
    else falling back to bar-style from top_k values).
    """
    target_ops_roles = [
        ("TriangleMultiplicationOutgoing", "output"),
        ("AttentionPairBias", "input"),
    ]
    colors = assign_colors(list(collected.keys()))

    ncols = len(target_ops_roles)
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))
    fig.suptitle(
        "Pairformer Layer-0 Distributions\n"
        "(tests whether layer-0 kurtosis pattern is universal)",
        fontsize=12, fontweight="bold",
    )

    for col, (op_name, role) in enumerate(target_ops_roles):
        ax = axes[col]
        ax.set_title(f"{_op_display(op_name)} [{role}]  layer=0", fontsize=10)
        ax.set_xlabel("Activation value")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.3)

        for input_name, (records, meta) in collected.items():
            # Find layer-0 record for this op/role
            candidates = [
                r for r in records
                if r.get("operator_name") == op_name
                and r.get("tensor_role") == role
                and r.get("layer_index", -1) == 0
            ]
            if not candidates:
                continue
            r = candidates[0]
            color = colors[input_name]
            seqlen = meta.get("seqlen", "?")

            bins = r.get("histogram_bins")
            counts = r.get("histogram_counts")
            if bins is not None and counts is not None:
                centres = np.array([(bins[i] + bins[i + 1]) / 2.0 for i in range(len(counts))])
                cnt_arr = np.array(counts, dtype=float)
                total = cnt_arr.sum()
                if total > 0:
                    density = cnt_arr / total / (bins[1] - bins[0])
                    ax.plot(centres, density, color=color, linewidth=1.4,
                            label=f"{input_name} (N={seqlen})")
            else:
                # Fallback: scatter top-k abs values
                top_vals = r.get("top_k_abs_values", [])
                if top_vals:
                    ax.scatter(top_vals, [1e-4] * len(top_vals),
                               color=color, s=4, alpha=0.5,
                               label=f"{input_name} (N={seqlen}, top-k only)")

        # Reference lines
        ax.axvline(448,  color="red",    linestyle="--", linewidth=1, label="E4M3 max")
        ax.axvline(-448, color="red",    linestyle="--", linewidth=1)
        ax.axvline(6,    color="orange", linestyle=":",  linewidth=1, label="|x|=6")
        ax.axvline(-6,   color="orange", linestyle=":",  linewidth=1)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", output_file)


# ---------------------------------------------------------------------------
# Output 8+9: Cross-input summary (text + CSV)
# ---------------------------------------------------------------------------

def write_cross_input_summary(
    collected: Dict[str, Tuple[List[dict], dict]],
    output_dir: Path,
) -> None:
    """Write cross_input_summary.txt and cross_input_summary.csv."""
    all_recs = flat_records(collected)
    if not all_recs:
        log.warning("No records to summarise.")
        return

    metrics = [
        "abs_max", "kurtosis", "fraction_gt6", "channel_max_to_median_ratio",
        "symmetric_outlier_correlation", "p999_to_p50_ratio",
    ]

    # ── cross_input_summary.csv ───────────────────────────────────────────
    csv_file = output_dir / "cross_input_summary.csv"
    with csv_file.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "input_name", "seqlen", "operator", "layer", "msa_layer",
            "role", "metric_name", "metric_value",
        ])
        for r in all_recs:
            base = [
                r.get("input_name", ""),
                r.get("seqlen_from_meta", ""),
                r.get("operator_name", ""),
                r.get("layer_index", -1),
                r.get("msa_layer_index", -1),
                r.get("tensor_role", ""),
            ]
            for m in metrics:
                val = r.get(m)
                if val is not None:
                    writer.writerow(base + [m, val])
    log.info("Saved %s", csv_file)

    # ── cross_input_summary.txt ───────────────────────────────────────────
    txt_file = output_dir / "cross_input_summary.txt"
    lines: List[str] = []
    lines.append("=" * 80)
    lines.append("CROSS-INPUT PAIRFORMER STATISTICS SUMMARY")
    lines.append(f"Inputs: {', '.join(sorted(collected.keys()))}")
    lines.append("=" * 80)

    # --- Per-metric global distribution ---
    lines.append("\n── Global distribution of each metric across all (input, op, layer) tuples ──")
    for m in metrics:
        vals = [_safe_float(r.get(m)) for r in all_recs]
        vals = [v for v in vals if not math.isnan(v) and v >= -1e-6]
        if not vals:
            continue
        arr = sorted(vals)
        n = len(arr)
        p50 = arr[n // 2]
        p90 = arr[int(n * 0.90)]
        p99 = arr[int(n * 0.99)]
        lines.append(
            f"  {m:40s}  n={n:6d}  median={p50:12.4f}  p90={p90:12.4f}  p99={p99:12.4f}"
        )

    # --- Top-10 worst (op, layer) averaged across inputs for each metric ---
    for m in metrics:
        lines.append(f"\n── Top-10 worst (operator, layer, role) by {m} [mean across inputs] ──")

        # Group by (op, layer, msa_layer, role)
        key_vals: dict = defaultdict(list)
        for r in all_recs:
            val = _safe_float(r.get(m))
            if math.isnan(val):
                continue
            key = (
                r.get("operator_name", ""),
                r.get("layer_index", -1),
                r.get("msa_layer_index", -1),
                r.get("tensor_role", ""),
            )
            key_vals[key].append(val)

        # Compute mean per key
        key_mean = {k: float(np.mean(v)) for k, v in key_vals.items() if v}

        # Sort: descending for most metrics, ascending for symmetric_outlier_correlation
        reverse = (m != "symmetric_outlier_correlation")
        top10 = sorted(key_mean.items(), key=lambda x: x[1], reverse=reverse)[:10]

        for (op, li, msa_li, role), mean_val in top10:
            loc = f"pf_layer={li}" if li >= 0 else f"msa_layer={msa_li}"
            lines.append(
                f"  {_op_display(op):18s}  {loc:16s}  role={role:22s}  mean={mean_val:12.4f}"
            )

    # --- Per-operator type: which layers consistently appear in top-10 ---
    lines.append("\n── Consistent hot-spot layers per operator (across all inputs) ──")
    all_ops = _PAIRFORMER_OPS + _MSA_OPS
    for op in all_ops:
        op_recs = [r for r in all_recs if r.get("operator_name") == op]
        if not op_recs:
            continue

        # For abs_max, find which layers appear in top-3 for EACH input
        per_input_top3: dict = defaultdict(set)
        for inp_name in collected:
            inp_op_recs = [
                r for r in op_recs
                if r.get("input_name") == inp_name and r.get("tensor_role") == "output"
            ]
            if not inp_op_recs:
                continue
            sorted_recs = sorted(inp_op_recs,
                                  key=lambda r: _safe_float(r.get("abs_max", 0)),
                                  reverse=True)
            for r in sorted_recs[:3]:
                per_input_top3[inp_name].add(_layer_key(r))

        if not per_input_top3:
            continue

        # Layers that appear in multiple inputs
        from collections import Counter
        counter: Counter = Counter()
        for layers in per_input_top3.values():
            for lay in layers:
                counter[lay] += 1

        consistent = [(lay, cnt) for lay, cnt in counter.most_common()
                      if cnt >= max(1, len(per_input_top3) // 2)]

        if consistent:
            consistent_str = ", ".join(
                f"layer={lay}(in {cnt}/{len(per_input_top3)} inputs)"
                for lay, cnt in consistent[:5]
            )
            lines.append(f"  {_op_display(op):18s}: {consistent_str}")

    # --- Variance across inputs ---
    lines.append("\n── Input-dependence: abs_max variance across inputs per (op, layer, role) ──")
    lines.append("   (high CV = input-dependent, low CV = structural property of the model)")
    key_per_input: dict = defaultdict(dict)
    for r in all_recs:
        key = (
            r.get("operator_name", ""),
            r.get("layer_index", -1),
            r.get("msa_layer_index", -1),
            r.get("tensor_role", ""),
        )
        inp = r.get("input_name", "")
        val = _safe_float(r.get("abs_max", float("nan")))
        if not math.isnan(val):
            key_per_input[key][inp] = val

    cv_entries = []
    for key, inp_vals in key_per_input.items():
        if len(inp_vals) < 2:
            continue
        vals = list(inp_vals.values())
        mean_v = float(np.mean(vals))
        std_v  = float(np.std(vals))
        cv = std_v / (mean_v + 1e-8)
        cv_entries.append((key, cv, mean_v, std_v))

    cv_entries.sort(key=lambda x: x[1], reverse=True)
    lines.append("\n  Top-10 most input-dependent (highest CV):")
    for (op, li, msa_li, role), cv, mean_v, std_v in cv_entries[:10]:
        loc = f"pf_layer={li}" if li >= 0 else f"msa_layer={msa_li}"
        lines.append(
            f"    {_op_display(op):18s}  {loc:14s}  role={role:20s}  "
            f"CV={cv:.3f}  mean={mean_v:.2f}  std={std_v:.2f}"
        )

    lines.append("\n  Top-10 most structurally consistent (lowest CV, mean abs_max > 1):")
    low_cv = [(k, cv, m, s) for k, cv, m, s in cv_entries if m > 1.0]
    low_cv.sort(key=lambda x: x[1])
    for (op, li, msa_li, role), cv, mean_v, std_v in low_cv[:10]:
        loc = f"pf_layer={li}" if li >= 0 else f"msa_layer={msa_li}"
        lines.append(
            f"    {_op_display(op):18s}  {loc:14s}  role={role:20s}  "
            f"CV={cv:.3f}  mean={mean_v:.2f}  std={std_v:.2f}"
        )

    lines.append("\n" + "=" * 80)

    txt_content = "\n".join(lines)
    with txt_file.open("w") as fh:
        fh.write(txt_content + "\n")
    print(txt_content)
    log.info("Saved %s", txt_file)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cross-input Pairformer activation analysis and visualization.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input_dirs", nargs="+", type=Path, metavar="DIR",
        help="One or more boltz_results_stats/{input_name}/ directories.",
    )
    group.add_argument(
        "--input_glob", type=str, metavar="PATTERN",
        help='Glob pattern, e.g. "boltz_results_stats/*".',
    )
    p.add_argument(
        "--output_dir", default=Path("boltz_results_plots"), type=Path,
        help="Output directory for plots and summaries. Default: boltz_results_plots",
    )
    p.add_argument(
        "--role", default=None, type=str,
        help="Filter records by tensor_role for line plots (default: all roles).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Collect input directories
    if args.input_dirs:
        input_dirs = [Path(d) for d in args.input_dirs]
    else:
        input_dirs = [Path(p) for p in glob.glob(args.input_glob)]
        input_dirs = sorted(d for d in input_dirs if d.is_dir())
        if not input_dirs:
            log.error("No directories matched glob pattern: %s", args.input_glob)
            sys.exit(1)

    log.info("Input directories: %s", [str(d) for d in input_dirs])

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load all data
    collected = collect_all_inputs(input_dirs)
    if not collected:
        log.error("No valid input data loaded.")
        sys.exit(1)

    log.info("Loaded data for inputs: %s", list(collected.keys()))

    # ── Plot 1: abs_max by layer ──────────────────────────────────────────
    _plot_metric_by_layer(
        collected,
        metric_key="abs_max",
        ylabel="abs_max (log scale)",
        title="Abs-Max by Layer and Operator (MXFP8 range overflow risk)",
        output_file=args.output_dir / "abs_max_by_layer.png",
        log_scale=True,
        role_filter=args.role,
    )

    # ── Plot 2: kurtosis by layer ─────────────────────────────────────────
    _plot_metric_by_layer(
        collected,
        metric_key="kurtosis",
        ylabel="kurtosis (Fisher)",
        title="Kurtosis by Layer and Operator (heavy tail → FP8 clipping error)",
        output_file=args.output_dir / "kurtosis_by_layer.png",
        log_scale=False,
        role_filter=args.role,
    )

    # ── Plot 3: frac_gt6 by layer ─────────────────────────────────────────
    _plot_metric_by_layer(
        collected,
        metric_key="fraction_gt6",
        ylabel="fraction |x| > 6",
        title="E4M3 Overflow Risk (fraction_gt6) by Layer",
        output_file=args.output_dir / "frac_gt6_by_layer.png",
        log_scale=False,
        role_filter=args.role,
    )

    # ── Plot 4: symmetry breaks ───────────────────────────────────────────
    plot_symmetry_breaks(collected, args.output_dir / "symmetry_breaks.png")

    # ── Plot 5: channel outlier severity ─────────────────────────────────
    plot_channel_outlier_severity(
        collected, args.output_dir / "channel_outlier_severity.png"
    )

    # ── Plot 6: OPM distributions ─────────────────────────────────────────
    plot_opm_distributions(collected, args.output_dir)

    # ── Plot 7: layer-0 distributions ────────────────────────────────────
    plot_layer0_distributions(
        collected, args.output_dir / "pairformer_layer0_distributions.png"
    )

    # ── Output 8+9: cross-input text + CSV summary ────────────────────────
    write_cross_input_summary(collected, args.output_dir)

    log.info("All outputs written to %s", args.output_dir)


if __name__ == "__main__":
    main()
