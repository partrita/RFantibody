#!/bin/bash

# Define common variables
NAME="9gox"
PDB_ANTIBODY="CCR8_subset.pdb"
PDB_ANTIGEN="9gox_CLC.pdb"
HOTSPOT="A175, S176, G179, V180, L181"
DESIGN_LOOPS="H1:7,H2:5,H3:5-8"
NUM_DESIGNS=2
SEQS_PER_STRUCT=2

# Define directory structure
BASE_DIR="/home/scripts"
INPUT_DIR="${BASE_DIR}/examples"
OUTPUT_DIR="${INPUT_DIR}"
RF_DIFFUSION_OUTPUT="${OUTPUT_DIR}/rfdiffusion/example_outputs/${NAME}"
PROTEINMPNN_OUTPUT="${OUTPUT_DIR}/proteinmpnn/example_outputs/${NAME}"
RF2_OUTPUT="${OUTPUT_DIR}/rf2/example_outputs/${NAME}"

# Function: Execute and verify each step
run_step() {
    local step=$1
    local command=$2

    echo "Step ${step}: Executing..."
    eval $command

    if [ $? -eq 0 ]; then
        echo "Step ${step} completed successfully."
        return 0
    else
        echo "Error occurred during step ${step}."
        return 1
    fi
}

# Step 1: RFdiffusion
step1_command="poetry run python ${BASE_DIR}/rfdiffusion_inference.py \
    --config-path /home/src/rfantibody/rfdiffusion/config/inference \
    --config-name antibody \
    antibody.target_pdb=${INPUT_DIR}/rfdiffusion/example_inputs/${PDB_ANTIBODY} \
    antibody.framework_pdb=${INPUT_DIR}/rfdiffusion/example_inputs/${PDB_ANTIGEN} \
    inference.ckpt_override_path=/home/weights/RFdiffusion_Ab.pt \
    'ppi.hotspot_res=[${HOTSPOT}]' \
    'antibody.design_loops=[${DESIGN_LOOPS}]' \
    inference.num_designs=${NUM_DESIGNS} \
    inference.output_prefix=${RF_DIFFUSION_OUTPUT}/ab"

# Step 2: ProteinMPNN
step2_command="poetry run python ${BASE_DIR}/proteinmpnn_interface_design.py \
    -pdbdir ${RF_DIFFUSION_OUTPUT} \
    -outpdbdir ${PROTEINMPNN_OUTPUT} \
    -seqs_per_struct ${SEQS_PER_STRUCT} -debug"

# Step 3: RF2 prediction
step3_command="poetry run python ${BASE_DIR}/rf2_predict.py \
    input.pdb_dir=${PROTEINMPNN_OUTPUT} \
    output.pdb_dir=${RF2_OUTPUT}"

# Execute steps sequentially
run_step 1 "$step1_command" && \
run_step 2 "$step2_command" && \
run_step 3 "$step3_command"

# Check final result
if [ $? -eq 0 ]; then
    echo "All processes completed successfully."
else
    echo "An error occurred during the process."
fi
