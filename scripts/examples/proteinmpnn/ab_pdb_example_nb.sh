#!/bin/bash

poetry run python /home/scripts/proteinmpnn_interface_design.py \
    -pdbdir /home/scripts/examples/rfdiffusion/example_outputs/nanobody \
    -outpdbdir /home/scripts/examples/proteinmpnn/example_outputs/nanobody \
    -debug
