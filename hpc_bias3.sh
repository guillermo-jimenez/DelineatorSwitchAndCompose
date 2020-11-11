#!/bin/bash
#SBATCH -J DelBias
#SBATCH -p medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --array=000-999
#SBATCH -o /homedtic/gjimenez/DADES/DADES/DelineationResults/BIAS_0_SEPARATE/LOGS/%A_%a.out
#SBATCH -e /homedtic/gjimenez/DADES/DADES/DelineationResults/BIAS_0_SEPARATE/LOGS/%A_%a.err

ORIGINAL_SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID;
WINDOW_SIZE=0;

module load Python/3.6.4-foss-2017a;
module load libGLU/9.0.0-foss-2017a;
source ~/VirtEnv/DeepLearning3/bin/activate;

cd ~/GitHub/DelineatorSwitchAndCompose;

for i in `seq 0 1000 10000`; 
do 
    SLURM_ARRAY_TASK_ID=$(expr $ORIGINAL_SLURM_ARRAY_TASK_ID + $i);
    python3 compute_bias.py --basedir /homedtic/gjimenez/DADES/DADES/PhysioNet/QTDB/manual0_bias --outdir /homedtic/gjimenez/DADES/DADES/DelineationResults/BIAS_0_SEPARATE --signal_id ${SLURM_ARRAY_TASK_ID} --win_size ${WINDOW_SIZE};
done
