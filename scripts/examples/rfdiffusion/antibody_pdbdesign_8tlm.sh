#!/bin/bash

poetry run python  /home/scripts/rfdiffusion_inference.py \
    --config-path  /home/src/rfantibody/rfdiffusion/config/inference \
    --config-name antibody \
    antibody.target_pdb=/home/scripts/examples/rfdiffusion/example_inputs/CCR8_subset.pdb \
    antibody.framework_pdb=/home/scripts/examples/rfdiffusion/example_inputs/hu-4D5-8_Fv.pdb \
    inference.ckpt_override_path=/home/weights/RFdiffusion_Ab.pt \
    'ppi.hotspot_res=[A175, V175, S176, E177, D178, G179, V180]' \
    'antibody.design_loops=[H1:7,H2:6,H3:5-13]' \
    inference.num_designs=2 \
    inference.output_prefix=/home/scripts/examples/rfdiffusion/example_outputs/8tlm/ab_8tlm
