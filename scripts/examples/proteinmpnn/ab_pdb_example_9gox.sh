#!/bin/bash

poetry run python /home/scripts/proteinmpnn_interface_design.py \
    -pdbdir /home/scripts/examples/rfdiffusion/example_outputs/9gox \
    -outpdbdir /home/scripts/examples/proteinmpnn/example_outputs/9gox \
    -loop_string H1,H2,H3 -seqs_per_struct 10 -debug
