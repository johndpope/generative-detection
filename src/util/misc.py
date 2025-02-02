# src/util/misc.py
import os
import sys
import logging
import json
import numpy as np
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, num_channels=13, num_frequencies=6):
        super(PositionalEncoding, self).__init__()
        self.num_channels = num_channels
        self.num_frequencies = num_frequencies
        
        self.output_dim = num_channels * 2 * self.num_frequencies

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.unsqueeze(-1)  # Shape: (batch_size, 13, 1)
        
        # Prepare the frequency bands
        frequencies = torch.arange(self.num_frequencies, dtype=torch.float32).to(x.device)
        frequencies = 2 ** frequencies * torch.pi  # Shape: (6,)
        
        # Compute the positional encodings
        sinusoids = torch.einsum('bix,f->bixf', x, frequencies)  # Shape: (batch_size, 13, 1, 6)
        sinusoids = torch.cat([torch.sin(sinusoids), torch.cos(sinusoids)], dim=-1)  # Shape: (batch_size, 13, 1, 12)
        
        # Flatten the last two dimensions
        sinusoids = sinusoids.view(batch_size, self.output_dim)  # Shape: (batch_size, 156)
        
        return sinusoids

class EasyDict(dict):
    """
    Code from https://github.com/makinacorpus/easydict
    
    Copyright (C) 2007 Free Software Foundation, Inc. <http://fsf.org/>
    
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
