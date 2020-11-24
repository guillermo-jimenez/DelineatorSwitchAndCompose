#!/bin/bash
#SBATCH -J DelSwitch
#SBATCH -p high
#SBATCH --exclude=node0[19-21,25]
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-6
#SBATCH -o /homedtic/gjimenez/DADES/DADES/Delineator/Logs/%A-%a.out
#SBATCH -e /homedtic/gjimenez/DADES/DADES/Delineator/Logs/%A-%a.err

module load Python/3.6.4-foss-2017a;
module load PyTorch/1.6.0-foss-2017a-Python-3.6.4-CUDA-10.1.105;
module load OpenBLAS/0.2.19-foss-2017a-LAPACK-3.7.0;
module load OpenMPI/2.0.2-GCC-6.3.0-2.27;

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK;

source ~/VirtEnv/DeepLearning3/bin/activate;
cd ~/GitHub/DelineatorSwitchAndCompose;
python3 train_multi.py --config_file ./configurations/HPC/${SLURM_ARRAY_TASK_ID}.json --input_files ./pickle/ --model_name TESTF1Loss_${SLURM_ARRAY_TASK_ID} --hpc 1;

