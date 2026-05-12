#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT/.deps:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PROJECT_ROOT/.cache/mpl}"
export OPENML_DATA_HOME="${OPENML_DATA_HOME:-$PROJECT_ROOT/data/openml_cache}"
mkdir -p "$MPLCONFIGDIR" "$OPENML_DATA_HOME"
exec "${GRS_ANFIS_PYTHON:-/home/harp3133t/anaconda3/envs/tabi/bin/python}" "$@"
