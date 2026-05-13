#!/usr/bin/env bash
# Run the full IRAP-Vietnam preparation pipeline.
#
# Usage:
#   prepare_dataset.sh <data_dir> [--skip-download] [--skip-extract]
#                                 [--ratios "TRAIN VAL TEST"] [--seed N]
#
# <data_dir>  IRAP_Vietnam dataset root. Must contain <data_dir>/_raw/ with
#             coding-tables.zip and attribute_metadata.json before running.
#
# Stage 3c (splits) here uses the automatic K-Means allocator; for manual
# adjustment re-run later with split_editor.py.
set -euo pipefail

usage() {
  sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'
}

if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

DATA_DIR="$1"; shift
RATIOS="0.8 0.1 0.1"
SEED=0
SKIP_DOWNLOAD=0
SKIP_EXTRACT=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-download) SKIP_DOWNLOAD=1; shift ;;
    --skip-extract)  SKIP_EXTRACT=1;  shift ;;
    --ratios) RATIOS="$2"; shift 2 ;;
    --seed)   SEED="$2";   shift 2 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $SKIP_DOWNLOAD -eq 0 ]]; then
  python "$SCRIPT_DIR/download_images.py" "$DATA_DIR"
fi
if [[ $SKIP_EXTRACT -eq 0 ]]; then
  python "$SCRIPT_DIR/extract_images.py" "$DATA_DIR" --ignore-duplicates
fi
python "$SCRIPT_DIR/parse_coding_tables.py" "$DATA_DIR"
python "$SCRIPT_DIR/build_metadata.py"      "$DATA_DIR"
python "$SCRIPT_DIR/make_splits.py"         "$DATA_DIR" --ratios $RATIOS --seed "$SEED"
