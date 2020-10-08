#!/bin/bash
#SBATCH -J DelSwitch
#SBATCH -p high
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o ~/DADES/DADES/Delineator/Logs/%N-%J.out
#SBATCH -e ~/DADES/DADES/Delineator/Logs/%N-%J.err
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK;

module load Python/3.6.4-foss-2017a;
source VirtEnv/DeepLearning/bin/activate;

cd ~/GitHub/DelineatorSwitchAndCompose;

python3 train.py --config_file ./configurations/MultiScaleUNet5Levels.json --input_files ./pickle/ --model_name Multi5All_2020_10-08;

