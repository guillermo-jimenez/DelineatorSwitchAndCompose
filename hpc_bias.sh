#!/bin/bash
#SBATCH -J DelBias
#SBATCH -p short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --array=0000-1000
#SBATCH -o /homedtic/gjimenez/DADES/DADES/DelineationResults/BIAS/LOGS/%J_%A_%a.out
#SBATCH -e /homedtic/gjimenez/DADES/DADES/DelineationResults/BIAS/LOGS/%J_%A_%a.err

ADDER=0000

SLURM_ARRAY_TASK_ID=$(expr $SLURM_ARRAY_TASK_ID + $ADDER);

module load Python/3.6.4-foss-2017a;
module load libGLU/9.0.0-foss-2017a;
source ~/VirtEnv/DeepLearning3/bin/activate;

cd ~/GitHub/DelineatorSwitchAndCompose;

python3 compute_bias.py --basedir /homedtic/gjimenez/DADES/DADES/PhysioNet/QTDB/manual0_bias --outdir /homedtic/gjimenez/DADES/DADES/DelineationResults/BIAS --signal_id ${SLURM_ARRAY_TASK_ID}
