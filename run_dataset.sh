#!/bin/bash
set -euo pipefail

DATA_ROOT=/home/yyin5/mack/datasets_mack_raw
OUTPUT_PATH=/home/yyin5/mack/yyin5/HUGSIM/outputs/benchmark_drivor_mpc

usage() {
    echo "Usage: $0 <agent_name> <dataset> <sim_cuda> <ad_cuda> [scenario_order] [split]"
    echo "  scenario_order: forward (default) or reverse -- run two workers from opposite"
    echo "                  ends of the list; finished scenes are skipped via their eval.json."
    echo "  split:          public (default) or private"
}

if [ -z "${1-}" ]; then
    echo "Error: Agent name must be provided as the first argument."
    usage; exit 1
fi
AGENT=${1}

if [ -z "${2-}" ]; then
    echo "Error: Dataset must be provided as the second argument."
    usage; exit 1
fi
DATASET=${2}

if [ -z "${3-}" ] || [ -z "${4-}" ]; then
    echo "Error: sim_cuda and ad_cuda must be provided as the third and fourth arguments."
    usage; exit 1
fi
sim_cuda=${3}
ad_cuda=${4}
scenario_order=${5:-forward}
split=${6:-public}

if [ "${scenario_order}" != "forward" ] && [ "${scenario_order}" != "reverse" ]; then
    echo "Error: scenario_order must be either 'forward' or 'reverse'."
    usage; exit 1
fi

# The public/private split differs in exactly three things: where the scenarios live, which
# base config carries the output_dir, and how the per-agent output group is named.
case "${split}" in
    public)
        scenario_dir="${DATA_ROOT}/HUGSIM-public/ss/scenarios/${DATASET}"
        base_cfg="./configs/sim/${DATASET}_base.yaml"
        out_group="${DATASET}_${AGENT}"
        ;;
    private)
        scenario_dir="${DATA_ROOT}/HUGSIM-private/ss/scenarios/${DATASET}"
        base_cfg="./configs/sim/${DATASET}_private.yaml"
        out_group="${DATASET}_private_${AGENT}"
        ;;
    *)
        echo "Error: split must be either 'public' or 'private'."
        usage; exit 1
        ;;
esac

echo "--- Processing dataset: ${DATASET} (${split}, ${scenario_order}) ---"

# Check if scenario files exist to avoid errors with glob
if ! ls "${scenario_dir}"/*.yaml 1> /dev/null 2>&1; then
    echo "Warning: No .yaml scenario files found in ${scenario_dir}. Skipping."
    exit 0
fi

if [ "${scenario_order}" = "reverse" ]; then
    mapfile -t scenario_files < <(printf '%s\n' "${scenario_dir}"/*.yaml | sort -r)
else
    mapfile -t scenario_files < <(printf '%s\n' "${scenario_dir}"/*.yaml | sort)
fi

for cfg in "${scenario_files[@]}"; do
    echo "Running scenario: ${cfg}"

    # --- extract scene_name and mode from yaml to build output dir path ---
    scene_name=$(grep "scene_name:" "$cfg" | sed 's/scene_name:[ ]*//' | sed 's/ //g' | sed "s/'//g")
    mode=$(grep "mode:" "$cfg" | sed 's/mode:[ ]*//')

    # --- build output dir path ---
    out_dir="${OUTPUT_PATH}/${out_group}/${scene_name}_${mode}"

    # --- skip only if eval output already exists ---
    if [ -f "$out_dir/eval.json" ]; then
        echo "⚠️  eval.json already exists, skipping: $out_dir"
        continue
    fi

    echo "→ Running simulation, output will be saved to: $out_dir"

    CUDA_VISIBLE_DEVICES=${sim_cuda} \
    python closed_loop.py --scenario_path "${cfg}" \
                        --base_path "${base_cfg}" \
                        --camera_path "./configs/sim/${DATASET}_camera.yaml" \
                        --kinematic_path "./configs/sim/kinematic.yaml" \
                        --ad "${AGENT}" \
                        --ad_cuda ${ad_cuda}
done
