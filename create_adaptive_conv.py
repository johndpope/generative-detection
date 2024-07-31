import importlib
import dnnlib
import torch
from src.util.misc import set_submodule_paths

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
    set_submodule_paths("submodules")
    model_cfg = {
        "base_learning_rate": 4.5e-6,
        "target": "src.models.autoencoder.AdaptivePoseAutoencoder",
        "params": {
            "quantize_obj": False,
            "quantize_pose": False,
            "quantconfig": {
                "target": "ldm.models.autoencoder.VectorQuantizer",
                "params": {
                    "n_e": 16384,
                    "e_dim": 4,
                    "remap": None,
                    "sane_index_shape": False,
                    "beta": 0.25
                }
            },
            "apply_convolutional_shift_img_space": False,
            "apply_convolutional_shift_latent_space": False,
            "monitor": "val/rec_loss",
            "embed_dim": 16,
            "euler_convention": "XYZ",
            "activation": "relu",
            "dropout_prob_init": 0.0,
            "dropout_prob_final": 0.0,
            "pose_conditioned_generation_steps": 0,
            "dropout_warmup_steps": 0,
            "train_on_yaw": True,
            "lossconfig": {
                "target": "src.modules.losses.PoseLoss",
                "params": {
                    "use_mask_loss": False,
                    "encoder_pretrain_steps": 0,
                    "codebook_weight": 1000,
                    "disc_start": 0,
                    "kl_weight_obj": 1.0e-05,
                    "kl_weight_bbox": 0.01,
                    "disc_weight": 0.5,
                    "pose_weight": 100000,
                    "fill_factor_weight": 100000,
                    "class_weight": 1000000,
                    "bbox_weight": 100000,
                    "pose_loss_fn": "l1",
                    "mask_weight": 0,
                    "mask_loss_fn": "l2",
                    "disc_in_channels": 3,
                    "num_classes": 5,
                    "dataset_stats_path": "dataset_stats/combined/all.pkl",
                    "train_on_yaw": True
                }
            },
            "pose_decoder_config": {
                "target": "src.modules.autoencodermodules.pose_decoder.PoseDecoderSpatialVAE",
                "params": {
                    "num_classes": 5,
                    "num_channels": 16,
                    "n": 16,
                    "m": 16,
                    "hidden_dim": 500,
                    "num_layers": 2,
                    "activation": "tanh",
                    "resid": False
                }
            },
            "pose_encoder_config": {
                "target": "src.modules.autoencodermodules.pose_encoder.PoseEncoderSpatialVAE",
                "params": {
                    "num_classes": 5,
                    "num_channels": 16,
                    "n": 16,
                    "m": 16,
                    "hidden_dim": 500,
                    "num_layers": 2,
                    "activation": "swish"
                }
            },
            "ddconfig": {
                "double_z": True,
                "z_channels": 16,
                "resolution": 64,
                "in_channels": 3,
                "out_ch": 3,
                "ch": 128,
                "ch_mult": [1, 1, 2, 2, 4],
                "num_res_blocks": 2,
                "attn_resolutions": [16],
                "dropout": 0.0
            }
        }
    }

    model = instantiate_from_config(model_cfg)





    # synthesis_layer_kwargs = {
    #     "target": "src.modules.autoencodermodules.adaptiveconv.SynthesisLayer",
    #     "params":{
    #         "in_channels": 16,
    #         "out_channels": 16,
    #         "w_dim": 13,
    #         "resolution": 16,
    #         "kernel_size": 3,
    #         "up": 1,
    #         "use_noise": True,
    #         "activation": "lrelu",
    #         "resample_filter": None,
    #         "conv_clamp": 256,
    #         "channels_last": False
    #     }
    # }

    # synthesis_layer = instantiate_from_config(synthesis_layer_kwargs)

    batch_size = 8
    # z_obj = torch.randn(batch_size, 16, 16, 16)
    pose = torch.randn(batch_size, 13)

    # x = synthesis_layer(z_obj, pose)
    input_im = torch.randn(batch_size, 3, 256, 256)
    out = model(input_im, pose)
    

if __name__ == "__main__":
    main()