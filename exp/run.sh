#!/bin/bash
# NERSC environment setup
module load conda
eval "$(conda shell.bash hook)"
conda activate /pscratch/sd/s/sgkim/tchoe_home/envs/tchoe_env

# Run python experiment script
python "$(dirname "$0")/run_exp.py"
