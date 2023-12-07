#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --job-name=LinProb
#SBATCH --nodes=1
#SBATCH --time=11:59:00
#SBATCH --exclusive
#SBATCH --mem=220G

module load cuda
module load conda
conda activate deeplearning3

srun python prova.py
