#!/bin/bash
set -euo pipefail

SCENARIO_PATH=/home/yyin5/mack/datasets_mack_raw/HUGSIM/ss/scenarios

OUTPUT_PATH=/home/yyin5/mack/yyin5/HUGSIM/outputs/benchmark

if [ -z "${1-}" ]; then
    echo "Error: Agent name must be provided as the first argument."
    echo "Usage: $0 <agent_name>"
    exit 1
fi
AGENT=${1}

DATASET=${2}

sim_cuda=${3}
ad_cuda=${4}

# change this variable as the scenario path on your machine

echo "--- Processing dataset: ${DATASET} ---"
scenario_dir=${SCENARIO_PATH}/${DATASET}

# Check if scenario files exist to avoid errors with glob
if ! ls "${scenario_dir}"/*.yaml 1> /dev/null 2>&1; then
    echo "Warning: No .yaml scenario files found in ${scenario_dir}. Skipping."
    continue
fi

for cfg in "${scenario_dir}"/*.yaml; do
    echo "Running scenario: ${cfg}"

    # --- extract scene_name and mode from yaml to build output dir path ---
    scene_name=$(grep "scene_name:" "$cfg" | sed 's/scene_name:[ ]*//' | sed 's/ //g' | sed "s/'//g")
    mode=$(grep "mode:" "$cfg" | sed 's/mode:[ ]*//')

    # --- build output dir path ---
    out_dir="${OUTPUT_PATH}/${DATASET}${AGENT}/${scene_name}_${mode}"

    # --- skip if already exists ---
    if [ -d "$out_dir" ]; then
        echo "⚠️  Output already exists, skipping: $out_dir"
        continue
    fi

    echo "→ Running simulation, output will be saved to: $out_dir"

    CUDA_VISIBLE_DEVICES=${sim_cuda} \
    python closed_loop.py --scenario_path "${cfg}" \
                        --base_path "./configs/sim/${DATASET}_base.yaml" \
                        --camera_path "./configs/sim/${DATASET}_camera.yaml" \
                        --kinematic_path "./configs/sim/kinematic.yaml" \
                        --ad "${AGENT}" \
                        --ad_cuda ${ad_cuda}
done
