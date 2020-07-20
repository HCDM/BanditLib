#!/bin/bash

#SBATCH --job-name=LinUCB
#SBATCH --output=partial_day/%j.out
#SBATCH --output=batch_results/yahoo_4layer/MLP_%j.out
#SBATCH --output=batch_results/yahoo_2layer/MLPMany_%j.out
#SBATCH --ntasks=1

srun python Simulation.py --config config.yaml 
