# %%
import torch
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from src.util.misc import log_opts, set_submodule_paths, set_cache_directories
set_submodule_paths(submodule_dir="submodules")
from ldm.util import instantiate_from_config
from train import get_data
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import math
from src.data.preprocessing.data_modules import DataModuleFromConfig
from torchvision.utils import save_image

# %%

config_path = "configs/autoencoder/4_adaptive_conv/learnt_shift_mini.yaml"
checkpoint_path = "logs/2024-08-23T05-06-58_learnt_shift/checkpoints/last.ckpt"
config = OmegaConf.load(config_path)

def load_model(config, ckpt_path):
    model = instantiate_from_config(config.model)
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

model = load_model(config, checkpoint_path)
model.eval()

# %%
data = get_data(config)
iteration = iter(data.datasets['train'])

# %%
counter = 0
while True:
    batch = next(iteration)
    model.chunk_size = 1
    model.class_thresh = 0.0 # TODO: Set to 0.5 or other value that works on val set
    model.fill_factor_thresh = 0.0 # TODO: Set to 0.5 or other value that works on val set
    model.num_refinement_steps = 10
    model.ref_lr=1.0e0

    # Prepare Input
    input_patches = batch[model.image_rgb_key].to(model.device).unsqueeze(0) # torch.Size([B, 3, 256, 256])
    assert input_patches.dim() == 4 or (input_patches.dim() == 5 and input_patches.shape[0] == 1), f"Only supporting batch size 1. Input_patches shape: {input_patches.shape}"
    if input_patches.dim() == 5:
        input_patches = input_patches.squeeze(0) # torch.Size([B, 3, 256, 256])
    input_patches = model._rescale(input_patches) # torch.Size([B, 3, 256, 256])

    # Chunked dec_pose[..., :POSE_6D_DIM + LHW_DIM + FILL_FACTOR_DIM]
    all_objects = []
    all_poses = []
    all_patch_indices = []
    all_scores = []
    all_classes = []
    chunk_size = model.chunk_size
    with torch.no_grad():
        for i in range(0, len(input_patches), chunk_size):
            selected_patches = input_patches[i:i+chunk_size]
            global_patch_index = i + torch.arange(chunk_size)[:len(selected_patches)]
            selected_patch_indices, z_obj, dec_pose, score, class_idx = model._get_valid_patches(selected_patches, global_patch_index)
            all_patch_indices.append(selected_patch_indices)
            all_objects.append(z_obj)
            all_poses.append(dec_pose)
            all_scores.append(score)
            all_classes.append(class_idx)
            
    scores = torch.cat(all_scores)
                
    # Inference refinement
    all_patch_indices = torch.cat(all_patch_indices)
    if not len(all_patch_indices):
        print(torch.empty(0))
    all_z_objects = torch.cat(all_objects)
    all_z_poses = torch.cat(all_poses)
    patches_w_objs = input_patches[all_patch_indices]
    gt_x = batch[model.pose_key][0]
    gt_y = batch[model.pose_key][1]
    dec_pose_refined, gen_image, x_list, y_list, grad_x_list, grad_y_list = model._refinement_step(patches_w_objs, all_z_objects, all_z_poses)
    # print ground truth pose
    print("Refined pose: ", dec_pose_refined[0])

    # plot x_list, grad_x_list on same plot with legend, at each refinement step k. refinement step is x axis. draw line at ground truth pose.
    # Do the same for y_list and grad_y_list. 

    def plot_grads(x_list, grad_x_list, y_list, grad_y_list, gt_x, gt_y):
        x_list = x_list.squeeze().detach().cpu().numpy()
        grad_x_list = grad_x_list.squeeze().detach().cpu().numpy()
        y_list = y_list.squeeze().detach().cpu().numpy()
        grad_y_list = grad_y_list.squeeze().detach().cpu().numpy()
        gt_x = gt_x.squeeze().detach().cpu().numpy()
        gt_y = gt_y.squeeze().detach().cpu().numpy()

        refinement_steps = len(x_list)
        plt.figure(figsize=(10, 5))
        
        # Plot x_list and grad_x_list
        plt.subplot(1, 2, 1)
        plt.plot(range(refinement_steps), x_list, label='x_list')
        plt.plot(range(refinement_steps), grad_x_list, label='grad_x_list')
        plt.axhline(y=gt_x, color='r', linestyle='--', label='Ground Truth x')
        plt.xlabel('Refinement Step')
        plt.ylabel('Value')
        plt.title('X List and Grad X List')
        plt.legend()
        
        # Plot y_list and grad_y_list
        plt.subplot(1, 2, 2)
        plt.plot(range(refinement_steps), y_list, label='y_list')
        plt.plot(range(refinement_steps), grad_y_list, label='grad_y_list')
        plt.axhline(y=gt_y, color='r', linestyle='--', label='Ground Truth y')
        plt.xlabel('Refinement Step')
        plt.ylabel('Value')
        plt.title('Y List and Grad Y List')
        plt.legend()
        
        plt.tight_layout()
        
        # save plot
        plt.savefig(f"grads_{counter}.png")
        plt.show()
    
    def scale_to_0_1(img):
        img = img - img.min()
        img = img / img.max()
        return img

    # save input_patches
    save_image(scale_to_0_1(input_patches), f"input_patches_{counter}.png")
    plot_grads(x_list, grad_x_list, y_list, grad_y_list, gt_x, gt_y)
    save_image(scale_to_0_1(gen_image), f"gen_image_{counter}.png")
    counter += 1

    
