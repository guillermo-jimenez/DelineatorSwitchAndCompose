#!/bin/bash
#SBATCH -J DelSwitch
#SBATCH -p high
#SBATCH --exclude=node0[19-21,25]
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-20
#SBATCH -o /homedtic/gjimenez/DADES/DADES/Delineator/Logs/%A-%a.out
#SBATCH -e /homedtic/gjimenez/DADES/DADES/Delineator/Logs/%A-%a.err

# Get a list of all possible models
list_all_models=(UNet5LevelsDiceOnly WNet5LevelsDiceOnly WNet5LevelsSelfAttentionDiceOnly WNet5LevelsSelfAttentionConvDiceOnly WNet5LevelsConvDiceOnly UNet5LevelsConvDiceOnly
UNet6LevelsDiceOnly WNet6LevelsDiceOnly WNet6LevelsSelfAttentionDiceOnly WNet6LevelsSelfAttentionConvDiceOnly WNet6LevelsConvDiceOnly UNet6LevelsConvDiceOnly
UNet7LevelsDiceOnly WNet7LevelsDiceOnly WNet7LevelsSelfAttentionDiceOnly WNet7LevelsSelfAttentionConvDiceOnly WNet7LevelsConvDiceOnly UNet7LevelsConvDiceOnly)

# Exit if array ID larger than array length
if [ $SLURM_ARRAY_TASK_ID -gt ${#list_all_models[@]} ];
then 
    exit 1
fi

# Get specific model
model=${list_all_models[$SLURM_ARRAY_TASK_ID]}

module load Python/3.6.4-foss-2017a;
module load PyTorch/1.6.0-foss-2017a-Python-3.6.4-CUDA-10.1.105;
module load OpenBLAS/0.2.19-foss-2017a-LAPACK-3.7.0;
module load OpenMPI/2.0.2-GCC-6.3.0-2.27;

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK;

source ~/VirtEnv/DeepLearning3/bin/activate;
cd ~/GitHub/DelineatorSwitchAndCompose;

python3 train_real.py --config_file ./configurations/${model}.json --input_files ./pickle/ --model_name ${model}_real_$(date '+%Y%m%d%H%M%S') --hpc 1;

