#!/bin/bash

poetry run python /home/scripts/util/chothia2HLT.py  \
    -i /home/scripts/examples/rfdiffusion/example_inputs/8tlm_chothia.pdb \
    -o /home/scripts/examples/rfdiffusion/example_inputs/8tlm \
    -H A -L B -T C
    # H stands for heavy chain, L stands for light chain, T standas for target protein.
