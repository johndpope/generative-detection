from setuptools import setup, find_packages

setup(
    name="your_project_name",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "mmengine",
        "mmcv==2.0.0rc4",
        "mmdet3d",
        "mmdet",
        "openmim==0.3.8",
        "omegaconf",
        "einops",
        "taming-transformers-rom1504",
        "wandb"
    ],
    dependency_links=[
        "https://download.openmmlab.com/mmcv/dist/cu116/torch1.12.0/index.html",
    ],
)
