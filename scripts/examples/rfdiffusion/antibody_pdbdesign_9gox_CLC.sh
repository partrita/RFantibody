#!/bin/bash

poetry run python  /home/scripts/rfdiffusion_inference.py \
    --config-path  /home/src/rfantibody/rfdiffusion/config/inference \
    --config-name antibody \
    antibody.target_pdb=/home/scripts/examples/rfdiffusion/example_inputs/CCR8_subset.pdb \
    antibody.framework_pdb=/home/scripts/examples/rfdiffusion/example_inputs/9gox_CLC.pdb \
    inference.ckpt_override_path=/home/weights/RFdiffusion_Ab.pt \
    'ppi.hotspot_res=[A175, S176, G179, V180, L181]' \
    'antibody.design_loops=[H1:7,H2:5,H3:5-8]' \
    inference.num_designs=2 \
    inference.output_prefix=/home/scripts/examples/rfdiffusion/example_outputs/9gox/ab_9gox
