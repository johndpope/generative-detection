#!/bin/bash

EMAIL_ADRESSE=xxxx@xxxxx
CONDA_ENV=test_mmdet3d

#SBATCH --job-name=enc_pretrain
#SBATCH --output=.out/slurm/%x/%J-%N-%t.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=$EMAIL_ADRESSE

source ~/.bashrc
conda activate $CONDA_ENV

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

srun python train.py -b configs/autoencoder/baseline/local_autoencoder_kl_16x16x16.yaml -t --name test --devices 1