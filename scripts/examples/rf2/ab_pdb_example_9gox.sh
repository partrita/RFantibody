#!/bin/bash

poetry run python /home/scripts/rf2_predict.py \
    input.pdb_dir=/home/scripts/examples/proteinmpnn/example_outputs/9gox \
    output.pdb_dir=/home/scripts/examples/rf2/example_outputs/9gox
