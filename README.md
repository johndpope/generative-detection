# OD-VAE: Inverting Generation for 3D Object Detection
This repository is the official implementation of [OD-VAE: Inverting Generation for 3D Object Detection]().

[arXiv]() | [BibTex](./CITATION.cff)

![OD-VAE and Latent Variables of an 3D Object.](./figs/pipeline.png)

[OD-VAE: Inverting Generation for 3D Object Detection]()

[Anonymous Author(s)]()

[Demo link]()

## Requirements

To install requirements:

1. Clone the repository:
```bash
git clone REPOSITORY_URL
```
2. Navigate into the cloned repository:
```bash
cd generative-detection
```
3. Create a new conda environment named `odvae`:
```bash
conda create --name odvae
```

4. Activate the new conda environment:
```bash
conda activate odvae
```

5. Install all requirements in this environment using the provided `setup.sh` script.
```bash
bash setup.sh
```

6. To initialize, fetch and checkout all the nested submodules:
```bash
git submodule update --init --recursive
```

7. Install the Python package:
```bash
pip install -e .
```

## Prepare nuScenes Dataset [2]
We use MMDetection3D as the base for our dataloader. Please follow the instructions given [here](https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/datasets/nuscenes.html) to download and prepare the nuScenes [2] dataset. Please place or symlink the directory with the prepared dataset at `data/nuscenes`.

## Prepare the Waymo Open Dataset [3]
We use MMDetection3D as the base for our dataloader. Please follow the instructions given [here](https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/datasets/waymo.html) to download and prepare the Waymo [3] dataset. Please place or symlink the directory with the prepared dataset at `data/waymo`.

## Training

To train the model in the paper on nuScenes [2], run this command:
```bash
srun python train.py -b configs/train_odvae_nusc.yaml -t --name odvae_train --devices 4
```

## Evaluation

To evaluate our model on nuScenes [2], run:
```bash
python eval.py --model-file trained_checkpoints/odvae_nusc.pth --benchmark nuscenes
```

To evaluate our model on Waymo Open Dataset [3], run:
```bash
python eval.py --model-file trained_checkpoints/odvae_waymo.pth --benchmark waymo
```

## Pre-trained Models

You can download our pretrained model here:
- [OD-VAE]() trained on the nuScenes dataset [2] using parameters specified in the config file `configs/train_odvae_nusc.yaml`
- [OD-VAE]() trained on the Waymo Open Dataset [3] using parameters specified in the config file `configs/train_odvae_waymo.yaml`

If you use any of these models in your work, we are always happy to receive a [citation](CITATION.cff)

## Results
Our model's performance is summarized in the tables below.

### [3D Object Detection on nuScenes [2]](https://paperswithcode.com/sota/3d-object-detection-on-nuscenes)

| Model name         | Metric 1        | Metric 2       |
| ------------------ |---------------- | -------------- |
| OD-VAE        |     xx%         |      xx%       |

### [3D Object Detection on Waymo Open Dataset [3]](https://paperswithcode.com/sota/3d-object-detection-on-waymo-vehicle)

| Model name         | Metric 1        | Metric 2       |
| ------------------ |---------------- | -------------- |
| OD-VAE        |     xx%         |      xx%       |


## Contributing
The code in this repository is released under the [CC0 1.0 Universal](LICENSE). We welcome any contributions to our repository via pull requests. 

## Acknowledgements
Our codebase for the architecture of training of the VAE builds heavily on the open-sourced repositories of [Latent Diffusion Models](https://github.com/CompVis/latent-diffusion/tree/a506df5756472e2ebaf9078affdde2c4f1502cd4) and [StyleGAN2](https://github.com/NVlabs/stylegan2-ada-pytorch). Thanks for open-sourcing!

## BibTeX
```bibtex
@article{anonymous_odvae_2024,
    author = Anonymous Authors,
    title = OD-VAE: Inverting Generation for 3D Object Detection,
    journal = {Journal Name},
    year = 2024,
    volume = {Volume},
    number = {Number},
    pages = {Pages},
    doi = {DOI},
    url = {URL}
}
```

## FAQ
- Downgrade MKL library to 2024.0.0 in case running `import torch` raises `undefined symbol: iJIT_NotifyEvent` from `torch/lib/libtorch_cpu.so`:
```bash
pip install mkl==2024.0.0
```
- In case you are having trouble resuming training from a checkpoint, please make sure to upgrade PyTorch version from v1.12.0 to v1.12.1! This issue is caused due to a bug in v1.12.0 that was fixed in v1.12.1.

## References
[1] Latent Diffusion Models: [ArXiv](https://arxiv.org/abs/2112.10752) | [GitHub](https://github.com/CompVis/latent-diffusion)

[2] nuScenes: [ArXiv](https://arxiv.org/abs/1903.11027)

[3] Waymo Open Dataset: [ArXiv](https://arxiv.org/abs/1912.04838)

[4] StyleGAN2: [ArXiv](https://arxiv.org/abs/2006.06676) | [GitHub](https://github.com/NVlabs/stylegan2-ada-pytorch)
