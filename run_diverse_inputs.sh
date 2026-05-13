#!/usr/bin/env bash
# run_diverse_inputs.sh
# =====================
# Run capture_pairformer_stats.py on a diverse set of inputs, then produce
# cross-input plots with analyze_and_plot.py.
#
# Prerequisites:
#   .venv/bin/python  (or set PYTHON= to your Python path)
#   boltz cache already populated (run `boltz predict` once to download weights)
#
# Usage:
#   bash run_diverse_inputs.sh
#
# Outputs go to:
#   boltz_results_stats/{input_name}/   — activations, weights, summary, meta
#   boltz_results_plots/                — cross-input plots and summaries
#
# All output directories match the boltz_results_* gitignore pattern.

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration — override with environment variables as needed
# ---------------------------------------------------------------------------

PYTHON="${PYTHON:-.venv/bin/python}"
SCRIPT="capture_pairformer_stats.py"
PLOT_SCRIPT="analyze_and_plot.py"
YAML_GEN="generate_example_yamls.py"
STATS_DIR="boltz_results_stats"
PLOTS_DIR="boltz_results_plots"
LOG_FILE="${STATS_DIR}/run.log"
DEVICE="${DEVICE:-cuda:0}"
SAMPLING_PRESET="${SAMPLING_PRESET:-medium}"
MAX_SEQLEN="${MAX_SEQLEN:-1100}"   # cover large_complex (~874 aa tokens)
RECYCLING="${RECYCLING:-1}"
MSA_FLAG="${MSA_FLAG:---use_msa_server}"   # set MSA_FLAG="" to skip MSA server

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "${LOG_FILE}"; }
err() { echo "[$(date '+%H:%M:%S')] ERROR: $*" | tee -a "${LOG_FILE}" >&2; }

mkdir -p "${STATS_DIR}"
: > "${LOG_FILE}"  # truncate log

log "=== run_diverse_inputs.sh started ==="
log "Python:  ${PYTHON}"
log "Device:  ${DEVICE}"
log "Preset:  ${SAMPLING_PRESET}"
log "MaxSeq:  ${MAX_SEQLEN}"

# ---------------------------------------------------------------------------
# Step 1 — Generate example YAML files from RCSB (if not already present)
# ---------------------------------------------------------------------------

log ""
log "─── Step 1: Generate example YAML files ───────────────────────────────"

NEED_GEN=0
for yaml in \
    examples/protein_medium.yaml \
    examples/protein_large.yaml \
    examples/protein_rna.yaml \
    examples/protein_ligand.yaml \
    examples/antibody.yaml \
    examples/homodimer.yaml \
    examples/large_complex.yaml
do
    if [[ ! -f "${yaml}" ]]; then
        NEED_GEN=1
        break
    fi
done

if [[ ${NEED_GEN} -eq 1 ]]; then
    log "Running generate_example_yamls.py ..."
    "${PYTHON}" "${YAML_GEN}" 2>&1 | tee -a "${LOG_FILE}"
else
    log "All YAML files already present. Run: python ${YAML_GEN} --force  to regenerate."
fi

# ---------------------------------------------------------------------------
# Step 2 — Define inputs to process
# ---------------------------------------------------------------------------

# Format: "yaml_file  extra_flags"
# --capture_weights only on the first run (model weights are the same for all)
declare -a INPUTS=(
    "examples/prot.yaml            --capture_weights"
    "examples/protein_medium.yaml  "
    "examples/protein_large.yaml   "
    "examples/protein_rna.yaml     "
    "examples/protein_ligand.yaml  "
    "examples/antibody.yaml        "
    "examples/homodimer.yaml       "
    "examples/large_complex.yaml   "
)

# ---------------------------------------------------------------------------
# Step 3 — Run capture for each input
# ---------------------------------------------------------------------------

log ""
log "─── Step 3: Run capture_pairformer_stats.py ────────────────────────────"

FAILED_INPUTS=()
SUCCEEDED_INPUTS=()

for entry in "${INPUTS[@]}"; do
    # Split into yaml path and extra flags
    yaml_file=$(echo "${entry}" | awk '{print $1}')
    extra_flags=$(echo "${entry}" | cut -d' ' -f2-)

    input_name=$(basename "${yaml_file}" .yaml)

    if [[ ! -f "${yaml_file}" ]]; then
        err "YAML not found: ${yaml_file} — skipping"
        FAILED_INPUTS+=("${input_name} (yaml missing)")
        continue
    fi

    log ""
    log ">>> Processing: ${input_name}  (${yaml_file})"

    # Check if already completed (activations.jsonl exists and is non-empty)
    out_dir="${STATS_DIR}/${input_name}"
    if [[ -f "${out_dir}/activations.jsonl" ]] && \
       [[ -s "${out_dir}/activations.jsonl" ]]; then
        log "    Already captured — skipping (delete ${out_dir}/activations.jsonl to re-run)"
        SUCCEEDED_INPUTS+=("${input_name}")
        continue
    fi

    # Build command
    cmd=(
        "${PYTHON}" "${SCRIPT}"
        --input "${yaml_file}"
        --output_dir "${STATS_DIR}"
        --device "${DEVICE}"
        --sampling_preset "${SAMPLING_PRESET}"
        --max_seqlen "${MAX_SEQLEN}"
        --recycling_steps "${RECYCLING}"
        --diffusion_samples 1
    )
    [[ -n "${MSA_FLAG}" ]] && cmd+=("${MSA_FLAG}")
    # Append per-input extra flags (if any)
    for flag in ${extra_flags}; do
        [[ -n "${flag}" ]] && cmd+=("${flag}")
    done

    log "    Command: ${cmd[*]}"

    if "${cmd[@]}" 2>&1 | tee -a "${LOG_FILE}"; then
        log "    SUCCESS: ${input_name}"
        SUCCEEDED_INPUTS+=("${input_name}")
    else
        err "FAILED: ${input_name} — continuing with next input"
        FAILED_INPUTS+=("${input_name}")
    fi
done

# ---------------------------------------------------------------------------
# Step 4 — Cross-input analysis and plots
# ---------------------------------------------------------------------------

log ""
log "─── Step 4: analyze_and_plot.py ────────────────────────────────────────"

if [[ ${#SUCCEEDED_INPUTS[@]} -eq 0 ]]; then
    err "No successful captures — skipping analysis."
else
    # Build list of input dirs that have data
    INPUT_DIRS=()
    for name in "${SUCCEEDED_INPUTS[@]}"; do
        dir="${STATS_DIR}/${name}"
        if [[ -f "${dir}/activations.jsonl" ]]; then
            INPUT_DIRS+=("${dir}")
        fi
    done

    if [[ ${#INPUT_DIRS[@]} -gt 0 ]]; then
        log "Running analyze_and_plot.py on: ${INPUT_DIRS[*]}"
        "${PYTHON}" "${PLOT_SCRIPT}" \
            --input_dirs "${INPUT_DIRS[@]}" \
            --output_dir "${PLOTS_DIR}" \
            2>&1 | tee -a "${LOG_FILE}"
        log "Plots written to: ${PLOTS_DIR}/"
    else
        err "No activations.jsonl files found in succeeded inputs."
    fi
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

log ""
log "=== run_diverse_inputs.sh complete ==="
log "Succeeded (${#SUCCEEDED_INPUTS[@]}): ${SUCCEEDED_INPUTS[*]:-none}"
if [[ ${#FAILED_INPUTS[@]} -gt 0 ]]; then
    log "Failed    (${#FAILED_INPUTS[@]}): ${FAILED_INPUTS[*]}"
fi
log "Log: ${LOG_FILE}"
log "Stats: ${STATS_DIR}/"
log "Plots: ${PLOTS_DIR}/"
