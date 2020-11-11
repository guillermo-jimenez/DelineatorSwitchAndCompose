#!/bin/bash
#SBATCH -J DelSwitch
#SBATCH -p high
#SBATCH --exclude=node0[19-21,25]
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --threads-per-core=4
#SBATCH --mem=32G
#SBATCH --array=0-6
#SBATCH -o /homedtic/gjimenez/DADES/DADES/Delineator/Logs/%A-%a.out
#SBATCH -e /homedtic/gjimenez/DADES/DADES/Delineator/Logs/%A-%a.err

module load Python/3.6.4-foss-2017a;
source ~/VirtEnv/DeepLearning3/bin/activate;

cd ~/GitHub/DelineatorSwitchAndCompose;

python3 train_multi.py --config_file ./configurations/HPC/${SLURM_ARRAY_TASK_ID}.json --input_files ./pickle/ --model_name TESTF1Loss_${SLURM_ARRAY_TASK_ID};

