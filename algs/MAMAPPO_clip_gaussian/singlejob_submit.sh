#!/bin/bash
# Reserve 4 cores (not threads), 1 exclusive node, 1 machine, 16G of total RAM, on the short partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --partition=short
#SBATCH --hint=nomultithread
#SBATCH --mail-user=e.marchesini@northeastern.edu
#SBATCH --mail-type=END,FAIL

# Run our python script (1 task with 24 cores)
#python main.py
srun python run_sweeps_from_cmd_file.py
