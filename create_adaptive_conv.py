import importlib
import dnnlib
import torch

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def main():
    synthesis_layer_kwargs = {
        "target": "src.modules.autoencodermodules.adaptiveconv.SynthesisLayer",
        "params":{
            "in_channels": 16,
            "out_channels": 16,
            "w_dim": 13,
            "resolution": 16,
            "kernel_size": 3,
            "up": 1,
            "use_noise": True,
            "activation": "lrelu",
            "resample_filter": None,
            "conv_clamp": 256,
            "channels_last": False
        }
    }

    synthesis_layer = instantiate_from_config(synthesis_layer_kwargs)

    batch_size = 8
    z_obj = torch.randn(batch_size, 16, 16, 16)
    pose = torch.randn(batch_size, 13)

    x = synthesis_layer(z_obj, pose)

if __name__ == "__main__":
    main()