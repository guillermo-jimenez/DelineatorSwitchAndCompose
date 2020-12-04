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
list_all_models=(UNet5Levels WNet5Levels WNet5LevelsSelfAttention WNet5LevelsSelfAttentionConv WNet5LevelsConv UNet5LevelsConv
UNet6Levels WNet6Levels WNet6LevelsSelfAttention WNet6LevelsSelfAttentionConv WNet6LevelsConv UNet6LevelsConv
UNet7Levels WNet7Levels WNet7LevelsSelfAttention WNet7LevelsSelfAttentionConv WNet7LevelsConv UNet7LevelsConv)

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

python3 train_synth.py --config_file ./configurations/${model}.json --input_files ./pickle/ --model_name ${model}_synth_$(date '+%Y%m%d%H%M%S') --hpc 1;

