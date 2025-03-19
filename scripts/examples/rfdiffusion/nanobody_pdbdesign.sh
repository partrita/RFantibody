#!/bin/bash

poetry run python  /home/scripts/rfdiffusion_inference.py \
    --config-path /home/src/rfantibody/rfdiffusion/config/inference \
    --config-name antibody \
    antibody.target_pdb=/home/scripts/examples/rfdiffusion/example_inputs/rsv_site3.pdb \
    antibody.framework_pdb=/home/scripts/examples/rfdiffusion/example_inputs/h-NbBCII10.pdb \
    inference.ckpt_override_path=/home/weights/RFdiffusion_Ab.pt \
    'ppi.hotspot_res=[T305,T456]' \
    'antibody.design_loops=[H1:7,H2:6,H3:5-13]' \
    inference.num_designs=2 \
    inference.final_step=48 \
    inference.deterministic=True \
    diffuser.T=50 \
    inference.output_prefix=/home/scripts/examples/rfdiffusion/example_outputs/nanobody/nb
