#!/bin/bash

# Create a new conda environment
# conda create -n odvae

# Activate your conda environment
# conda activate odvae

# Install dependencies using Conda
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
conda install nvidia/label/cuda-11.6.2::cuda-toolkit
conda install pytorch-lightning=1.9.4 -c conda-forge
conda install pytorch3d::pytorch3d

# Install Python packages with pip (those managed in setup.py)
pip install -e .

# Alternatively, use pip to install remaining packages if setup.py does not cover them all
# pip install mmengine
# pip install "mmcv==2.0.0rc4" -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.12.0/index.html
# pip install mmdet3d
# pip install mmdet
# pip install -U openmim==0.3.8
# pip install omegaconf
# pip install einops
# pip install taming-transformers-rom1504
# pip install wandb