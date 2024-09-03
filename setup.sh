#!/bin/bash

# Install dependencies using Conda
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
conda install nvidia/label/cuda-11.6.2::cuda-toolkit
conda install pytorch-lightning=1.9.4 -c conda-forge
conda install pytorch3d::pytorch3d

# Install Python packages with pip (those managed in setup.py)
pip install -e .
