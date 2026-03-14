#!/bin/bash
#SBATCH --job-name=chebi_task1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --gpus=1
#SBATCH --partition=tutorial
#SBATCH --time=0-12:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

module load GCCcore/13.2.0
module load Python/3.11.5

uv run main.py
