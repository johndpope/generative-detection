# src/util/misc.py
import os
import sys
import logging
import json
from typing import List
import numbers
import torch
from torch import Tensor
from torchvision.transforms.functional import get_dimensions, pad, crop
from torchvision.transforms.transforms import CenterCrop
import numpy as np

class EasyDict(dict):
    """
    Code from https://github.com/makinacorpus/easydict
    
    A dictionary subclass that allows accessing keys as attributes.

    This class provides a convenient way to access dictionary keys as attributes.
    It inherits from the built-in `dict` class and overrides the `__setattr__` and `__setitem__` methods
    to enable attribute-style access and assignment.

    Example usage:
    ```
    d = EasyDict({'key1': 'value1', 'key2': 'value2'})
    print(d.key1)  # Output: 'value1'
    d.key3 = 'value3'
    print(d['key3'])  # Output: 'value3'
    ```

    Note: When assigning a value to an attribute, if the value is a dictionary, it will be converted to an `EasyDict`.
    Similarly, if the value is a list or tuple containing dictionaries, they will be converted to `EasyDict` as well.

    """

    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        else:
            d = dict(d)        
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        for k in self.__class__.__dict__.keys():
            if not (k.startswith('__') and k.endswith('__')) and k not in ('update', 'pop'):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = type(value)(self.__class__(x)
                     if isinstance(x, dict) else x for x in value)
        elif isinstance(value, dict) and not isinstance(value, EasyDict):
            value = EasyDict(value)
        super(EasyDict, self).__setattr__(name, value)
        super(EasyDict, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def update(self, e=None, **f):
        d = e or dict()
        d.update(f)
        for k in d:
            setattr(self, k, d[k])

    def pop(self, k, *args):
        if hasattr(self, k):
            delattr(self, k)
        return super(EasyDict, self).pop(k, *args)

def log_opts(opts):
    """Log the options."""
    logging.info("Options: %s", json.dumps(vars(opts), indent=2))

def set_submodule_paths(submodule_dir):
    """Set the paths for the submodules."""
    for submodule in os.listdir(submodule_dir):
        submodule_path = os.path.join(submodule_dir, submodule)
        if os.path.isdir(submodule_path):
            sys.path.append(submodule_path)

def set_cache_directories(opts):
    """Set environment variables for cache directories."""
    os.environ["TRANSFORMERS_CACHE"] = opts.transformers_cache
    os.environ["TORCH_HOME"] = opts.torch_home

def flip_tensor(t):
    flipped = t.cpu().numpy().copy()

    flipped = np.flip(flipped, axis=(2, 3))
    return torch.Tensor(flipped.copy()).to(t.device)

class ReflectPadCenterCrop(CenterCrop):
    """Crops the given image at the center.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    If image size is smaller than output size along any edge, image is padded with 0 and then center cropped.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
    """

    def __init__(self, size):
        super().__init__(size)

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        return reflect_pad_center_crop(img, self.size)

def reflect_pad_center_crop(img: Tensor, output_size: List[int]) -> Tensor:
    """Crops the given image at the center, padding using reflect mode.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    If image size is smaller than output size along any edge, image is padded with 0 and then center cropped.

    Args:
        img (PIL Image or Tensor): Image to be cropped.
        output_size (sequence or int): (height, width) of the crop box. If int or sequence with single int,
            it is used for both directions.

    Returns:
        PIL Image or Tensor: Cropped image.
    """
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    elif isinstance(output_size, (tuple, list)) and len(output_size) == 1:
        output_size = (output_size[0], output_size[0])

    _, image_height, image_width = get_dimensions(img)
    crop_height, crop_width = output_size

    if crop_width > image_width or crop_height > image_height:
        padding_ltrb = [
            (crop_width - image_width) // 2 if crop_width > image_width else 0,
            (crop_height - image_height) // 2 if crop_height > image_height else 0,
            (crop_width - image_width + 1) // 2 if crop_width > image_width else 0,
            (crop_height - image_height + 1) // 2 if crop_height > image_height else 0,
        ]
        img = pad(img, padding_ltrb, padding_mode='reflect') # Reflect padding, no fill needed
        _, image_height, image_width = get_dimensions(img)
        if crop_width == image_width and crop_height == image_height:
            return img

    crop_top = int(round((image_height - crop_height) / 2.0))
    crop_left = int(round((image_width - crop_width) / 2.0))
    return crop(img, crop_top, crop_left, crop_height, crop_width)