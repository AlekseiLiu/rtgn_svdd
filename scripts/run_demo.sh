#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/demo.json}
PYTHON_BIN=${PYTHON_BIN:-python}
RTGN_SVDD_BIN=${RTGN_SVDD_BIN:-rtgn-svdd}
export PYTHONUNBUFFERED=1

mapfile -t RUN_IDENTITY < <(
    "$PYTHON_BIN" -c \
        'import json,sys; c=json.load(open(sys.argv[1])); print(c["experiment"]["output_root"]); print(c["experiment"]["name"])' \
        "$CONFIG"
)
if [[ ${#RUN_IDENTITY[@]} -ne 2 ]]; then
    echo "could not resolve experiment identity from $CONFIG" >&2
    exit 2
fi

OUTPUT_ROOT=${RUN_IDENTITY[0]}
EXPERIMENT_NAME=${RUN_IDENTITY[1]}
COMPLETION_MARKER="$OUTPUT_ROOT/$EXPERIMENT_NAME.completed"
if [[ -e "$COMPLETION_MARKER" ]]; then
    echo "refusing to overwrite completed demo: $COMPLETION_MARKER" >&2
    echo "copy the config and change experiment.name or experiment.output_root" >&2
    exit 2
fi

run_stage() {
    echo "[rTGN-SVDD demo] $*"
    "$RTGN_SVDD_BIN" "$@"
}

run_stage validate --config "$CONFIG"
run_stage train --config "$CONFIG" --variant baseline
run_stage train --config "$CONFIG" --variant robust
run_stage evaluate --config "$CONFIG" --variant baseline
run_stage evaluate --config "$CONFIG" --variant robust
run_stage diagnostic --config "$CONFIG"

mkdir -p "$OUTPUT_ROOT"
touch "$COMPLETION_MARKER"
echo "[rTGN-SVDD demo] complete: $COMPLETION_MARKER"
