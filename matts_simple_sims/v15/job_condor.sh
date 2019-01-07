#!/bin/bash

source activate bnl

echo `which python`

# about 1 to 1.6 hours per job
# args are nsims, seed, odir
python run_sim_condor.py $1 $2 $3
