#!/bin/bash

poetry run python /home/scripts/util/chothia2HLT.py  \
    -i /home/scripts/examples/rfdiffusion/example_inputs/9gox_chothia.pdb \
    -o /home/scripts/examples/rfdiffusion/example_inputs/9gox \
    -H H -L L -T A 
    # H stands for heavy chain, L stands for light chain, T standas for target protein.
