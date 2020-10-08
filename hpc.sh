#!/bin/bash
#SBATCH -J DelSwitch
#SBATCH -p high
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o /homedtic/gjimenez/DADES/DADES/Delineator/Logs/Config_%A_%a.out
#SBATCH -e /homedtic/gjimenez/DADES/DADES/Delineator/Logs/Config_%A_%a.err
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load Python/3.6.4-foss-2017a
source VirtEnv/DeepLearning/bin/activate

