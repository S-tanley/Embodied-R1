#!/bin/bash
# Usage:
#   ./run_all.sh [OPTIONS] [EVAL...]
#
# Options:
#   -g GPUS           Comma-separated GPU IDs (default: 0)
#   -n NUM_PROCESSES  Number of processes for accelerate (default: 1)
#   -p PORT           Main process port base (default: 54320; increments per eval)
#   -h                Show this help
#
# EVAL can be one or more of:
#   3d  affordance  blink  crpe  cvbench  embspatial
#   roborefit  sat  vabench_point  vabench_trace  where2place
#
# Examples:
#   ./run_all.sh -g 3,4,5 -n 3 sat roborefit
#   ./run_all.sh -g 2 -n 1 cvbench embspatial roborefit
#   ./run_all.sh -g 0,1 -n 2          # run ALL evals

set -e
cd "$(dirname "$0")"

# ---------- defaults ----------
GPUS="0"
NUM_PROCESSES=1
PORT_BASE=54320

declare -A SCRIPT_MAP=(
    [3d]=hf_inference_3d.py
    [affordance]=hf_inference_affordance.py
    [blink]=hf_inference_blink.py
    [crpe]=hf_inference_crpe.py
    [cvbench]=hf_inference_cvbench.py
    [embspatial]=hf_inference_embspatial.py
    [roborefit]=hf_inference_roborefit.py
    [sat]=hf_inference_sat.py
    [vabench_point]=hf_inference_vabench_point.py
    [vabench_trace]=hf_inference_vabench_visual_trace.py
    [where2place]=hf_inference_where2place.py
)

ALL_EVALS="3d affordance blink crpe cvbench embspatial roborefit sat vabench_point vabench_trace where2place"

# ---------- parse options ----------
while getopts "g:n:p:h" opt; do
    case $opt in
        g) GPUS="$OPTARG" ;;
        n) NUM_PROCESSES="$OPTARG" ;;
        p) PORT_BASE="$OPTARG" ;;
        h)
            head -20 "$0" | grep "^#" | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown option. Use -h for help."; exit 1 ;;
    esac
done
shift $((OPTIND - 1))

# ---------- select evals ----------
if [ $# -eq 0 ]; then
    SELECTED="$ALL_EVALS"
else
    SELECTED="$*"
fi

# ---------- validate ----------
for name in $SELECTED; do
    if [ -z "${SCRIPT_MAP[$name]}" ]; then
        echo "ERROR: Unknown eval '$name'. Valid names: $ALL_EVALS"
        exit 1
    fi
done

# ---------- run ----------
PORT=$PORT_BASE
mkdir -p ../logs

echo "GPUs: $GPUS  |  num_processes: $NUM_PROCESSES"
echo "Evals: $SELECTED"
echo ""

for name in $SELECTED; do
    script="${SCRIPT_MAP[$name]}"
    logfile="../logs/run_${name}.log"

    echo "=========================================="
    echo "Running: $script  (port $PORT)"
    echo "Log: $logfile"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES=$GPUS accelerate launch \
        --num_processes=$NUM_PROCESSES \
        --main_process_port=$PORT \
        "$script" 2>&1 | tee "$logfile"

    PORT=$((PORT + 1))
    echo ""
done

echo "All evaluations completed!"
