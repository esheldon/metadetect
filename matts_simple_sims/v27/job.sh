#!/bin/bash
#SBATCH -J run-tiles
#SBATCH -A metashear
#SBATCH -p bdwall
#SBATCH -N 100
#SBATCH --ntasks-per-node=1
#SBATCH -o myjob.oe
#SBATCH -t 15:00:00

source activate lcrc

echo `which python`

export I_MPI_FABRICS=shm:tmi
srun python run_sim.py 100000
