#!/bin/bash

#SBATCH --job-name=qltf8_process
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=150:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=qltf8@nyu.edu
#SBATCH --output=slurm_%j.out
#SBATCH --gres=gpu:1

module load python3/intel/3.6.3
source /home/ql819/pytorch-gpu/py3.6.3/bin/activate


python ./train_better.py