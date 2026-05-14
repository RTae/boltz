#!/usr/bin/env bash
set -euo pipefail

readonly BOLTZ_DATA_BASE_URL="https://boltz1.s3.us-east-2.amazonaws.com"

usage() {
	cat <<'EOF'
Usage: scripts/train/download_data.sh [--output-dir PATH] [DATASET ...]

Download Boltz pre-processed training data.

Datasets:
	all                Download every dataset listed below (default)
	rcsb_targets       Pre-processed RCSB structures
	rcsb_msa           Pre-processed RCSB MSAs
	openfold_targets   Pre-processed OpenFold structures
	openfold_msa       Pre-processed OpenFold MSAs
	symmetry           Pre-computed ligand symmetry file

Examples:
	scripts/train/download_data.sh
	scripts/train/download_data.sh --output-dir data rcsb_targets rcsb_msa symmetry
EOF
}

download_file() {
	local url="$1"
	local destination="$2"

	mkdir -p "$OUTPUT_DIR"

	if command -v wget >/dev/null 2>&1; then
		wget -c -O "$destination" "$url"
		return
	fi

	if command -v curl >/dev/null 2>&1; then
		curl --fail -L -C - -o "$destination" "$url"
		return
	fi

	echo "error: wget or curl is required to download training data" >&2
	exit 1
}

download_archive() {
	local archive_name="$1"
	local extracted_name="$2"
	local archive_path="$OUTPUT_DIR/$archive_name"
	local extracted_path="$OUTPUT_DIR/$extracted_name"
	local url="$BOLTZ_DATA_BASE_URL/$archive_name"

	if [[ -e "$extracted_path" ]]; then
		echo "Skipping $archive_name because $extracted_path already exists"
		return
	fi

	if [[ -f "$archive_path" ]]; then
		echo "Using existing $archive_name"
	else
		echo "Downloading $archive_name"
		download_file "$url" "$archive_path"
	fi

	echo "Extracting $archive_name"
	tar -xf "$archive_path" -C "$OUTPUT_DIR"
	rm -f "$archive_path"
}

download_plain_file() {
	local file_name="$1"
	local file_path="$OUTPUT_DIR/$file_name"
	local url="$BOLTZ_DATA_BASE_URL/$file_name"

	if [[ -f "$file_path" ]]; then
		echo "Skipping $file_name because $file_path already exists"
		return
	fi

	echo "Downloading $file_name"
	download_file "$url" "$file_path"
}

download_dataset() {
	local dataset="$1"

	case "$dataset" in
		rcsb_targets)
			download_archive "rcsb_processed_targets.tar" "rcsb_processed_targets"
			;;
		rcsb_msa)
			download_archive "rcsb_processed_msa.tar" "rcsb_processed_msa"
			;;
		openfold_targets)
			download_archive "openfold_processed_targets.tar" "openfold_processed_targets"
			;;
		openfold_msa)
			download_archive "openfold_processed_msa.tar" "openfold_processed_msa"
			;;
		symmetry)
			download_plain_file "symmetry.pkl"
			;;
		*)
			echo "error: unknown dataset '$dataset'" >&2
			exit 1
			;;
	esac
}

OUTPUT_DIR="$PWD"
selected_datasets=()

while [[ $# -gt 0 ]]; do
	case "$1" in
		--help|-h)
			usage
			exit 0
			;;
		--output-dir)
			if [[ $# -lt 2 ]]; then
				echo "error: --output-dir requires a path" >&2
				exit 1
			fi
			OUTPUT_DIR="$2"
			shift 2
			;;
		all|rcsb_targets|rcsb_msa|openfold_targets|openfold_msa|symmetry)
			selected_datasets+=("$1")
			shift
			;;
		*)
			echo "error: unknown dataset or option '$1'" >&2
			usage >&2
			exit 1
			;;
	esac
done

if [[ ${#selected_datasets[@]} -eq 0 ]]; then
	selected_datasets=(all)
fi

if [[ " ${selected_datasets[*]} " == *" all "* ]]; then
	selected_datasets=(
		rcsb_targets
		rcsb_msa
		openfold_targets
		openfold_msa
		symmetry
	)
fi

for dataset in "${selected_datasets[@]}"; do
	download_dataset "$dataset"
done

echo "Training data downloaded to $OUTPUT_DIR"
