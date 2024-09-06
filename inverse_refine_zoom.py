# inverse_refinement.py
import os
import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf
import torchvision.transforms as transforms
from torchvision.utils import save_image
from src.util.misc import set_submodule_paths
set_submodule_paths(submodule_dir="submodules")
from ldm.util import instantiate_from_config
from train import get_data

def load_model(config, ckpt_path):
    model = instantiate_from_config(config.model)
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


PATCH_ANCHOR_SIZES = [50, 100, 200, 400]

def compute_z_crop(H, H_crop, x, y, z, multiplier, eps=1e-8):
    """
    Compute the cropped z-coordinate of an object given its 3D coordinates.

    Args:
        H (float): Height of the image.
        H_crop (float): Height of the cropped image.
        x (torch.Tensor): x-coordinate of the object.
        y (torch.Tensor): y-coordinate of the object.
        z (torch.Tensor): z-coordinate of the object.
        multiplier (float): Multiplier to adjust the z-coordinate.
        eps (float, optional): Small value to avoid division by zero. Defaults to 1e-8.

    Returns:
        torch.Tensor: Cropped z-coordinate of the object.
    """
    obj_dist_sq = x**2 + y**2 + z**2
    obj_dist = torch.sqrt(abs(obj_dist_sq + eps)).squeeze()
    multiplier = torch.tensor(multiplier)
    obj_dist_crop = obj_dist / multiplier
    z_crop = torch.sqrt(torch.clamp((obj_dist_crop**2 - x**2 - y**2), min=0.0) + eps)
    z_crop = z_crop.reshape_as(z)
    return z_crop

def get_min_max_multipliers(patch_size_original):
    # Find the index of patch_size_original
    patch_size_original = patch_size_original.max()
    index = PATCH_ANCHOR_SIZES.index(patch_size_original)

    # Determine the next smallest and next biggest sizes
    next_smallest = PATCH_ANCHOR_SIZES[index - 1] if index > 0 else None
    next_biggest = PATCH_ANCHOR_SIZES[index + 1] if index < len(PATCH_ANCHOR_SIZES) - 1 else None

    # Determine the threshold values
    lower_threshold = (patch_size_original + next_smallest) / 2 if next_smallest else patch_size_original
    upper_threshold = (patch_size_original + next_biggest) / 2 if next_biggest else patch_size_original

    # Function to find the multipliers
    def find_multipliers(patch_size_original, lower_threshold, upper_threshold):
        m_min = lower_threshold / patch_size_original
        m_max = upper_threshold / patch_size_original
        return m_min, m_max

    m_min, m_max = find_multipliers(patch_size_original, lower_threshold, upper_threshold)
    
    return m_min, m_max

def get_perturbed_z_fill(pose_gt, second_pose, fill_factor_gt, patch_size_original):

    # second_pose = torch.cat((snd_pose, snd_bbox, snd_fill, class_probs), dim=1)
    last_dim = len(second_pose) - (4+3+1)
    snd_pose, snd_bbox, snd_fill, class_probs = torch.split(second_pose, [4, 3, 1, last_dim], dim=0)
    fill_factor_gt = torch.zeros_like(pose_gt[0]) + fill_factor_gt
    pose_gt_full = torch.cat((pose_gt, snd_bbox, fill_factor_gt.unsqueeze(0), class_probs), dim=0)
    # Unperturbed pose (original second pose)
    unperturbed_pose = pose_gt_full.clone() # pose 1

    # Original image dimensions
    
    m_min, m_max = get_min_max_multipliers(patch_size_original=patch_size_original)
    
    # Generate evenly spaced multipliers
    H, W = patch_size_original[0,0], patch_size_original[0,1]
    multiplier = m_min
    # Perturb only the z-coordinate using evenly spaced multipliers
    # for multiplier in multipliers:
    perturbed_pose = pose_gt_full.clone()
    # Use the compute_z_crop function to calculate the new z value
    z_crop = compute_z_crop(H=H, H_crop=H * multiplier, x=perturbed_pose[0], y=perturbed_pose[1], z=perturbed_pose[2], multiplier=multiplier)
    
    # Update the z value of the perturbed pose
    perturbed_pose[2] = z_crop
    fill_factor_crop = fill_factor_gt * multiplier 

    # fill factor id
    perturbed_pose[7] = fill_factor_crop
    
    return z_crop, fill_factor_crop

def main():
    config_path = "configs/autoencoder/zoom/learnt_zoom.yaml"
    checkpoint_path = "logs/2024-08-30T10-55-19_learnt_zoom/checkpoints/last.ckpt"
    config = OmegaConf.load(config_path)

    model = load_model(config, checkpoint_path)
    model.eval()

    data = get_data(config)
    iteration = iter(data.datasets['validation'])

    counter = 0
    while True:
        k = 1 # get kth batch
        for i in range(k):
            batch = next(iteration)

        model.chunk_size = 1
        model.class_thresh = 0.0 # TODO: Set to 0.5 or other value that works on val set
        model.fill_factor_thresh = 0.0 # TODO: Set to 0.5 or other value that works on val set
        model.num_refinement_steps = 10
        # model.ref_lr=1.0e-1 # shift only

        model.ref_lr=1.0e-2 # zoom + fill only
        model.tv_loss_weight = 1.0e-4

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
        # gt_x = batch[model.pose_key][0]
        # gt_y = batch[model.pose_key][1]
        rgb_in, rgb_gt, pose_gt, segm_mask_gt, mask_2d_bbox, class_gt, class_gt_label, bbox_gt, fill_factor_gt, second_pose = model.get_all_inputs(batch)
        gt_z = pose_gt[2]
        gt_fill = fill_factor_gt
        perturbed_z, perturbed_fill = get_perturbed_z_fill(pose_gt=pose_gt,
                                        second_pose=second_pose,
                                        fill_factor_gt=batch[model.fill_factor_key], 
                                        patch_size_original=batch['patch_size'])
        save_dir = "images/inference_refinement_zoom"
        os.makedirs(save_dir, exist_ok=True)
        # def plot_grads(x_list, grad_x_list, y_list, grad_y_list, gt_x, gt_y, loss_list, tv_loss_list, post_fix=""):
        def plot_grads(x_list, grad_x_list, y_list, grad_y_list, gt_x, gt_y, loss_list, tv_loss_list, post_fix=""):

            x_list = x_list.squeeze().detach().cpu().numpy()
            grad_x_list = grad_x_list.squeeze().detach().cpu().numpy()
            y_list = y_list.squeeze().detach().cpu().numpy()
            grad_y_list = grad_y_list.squeeze().detach().cpu().numpy()
            gt_x = gt_x.squeeze().detach().cpu().numpy()
            gt_y = gt_y.squeeze().detach().cpu().numpy()
            loss_list = loss_list.squeeze().detach().cpu().numpy()

            refinement_steps = len(x_list)
            plt.figure(figsize=(15, 5))
            
            # Plot x_list and grad_x_list
            plt.subplot(1, 3, 1)
            plt.plot(range(refinement_steps), x_list, label='z_list')
            plt.plot(range(refinement_steps), grad_x_list, label='grad_z_list')
            plt.axhline(y=gt_x, color='r', linestyle='--', label='Ground Truth z')
            plt.xlabel('Refinement Step')
            plt.ylabel('Value')
            plt.title('Z List and Grad Z List')
            plt.legend()
            
            # Plot y_list and grad_y_list
            plt.subplot(1, 3, 2)
            plt.plot(range(refinement_steps), y_list, label='fill_factor_list')
            plt.plot(range(refinement_steps), grad_y_list, label='grad_fill_factor_list')
            plt.axhline(y=gt_y, color='r', linestyle='--', label='Ground Truth fill_factor')
            plt.xlabel('Refinement Step')
            plt.ylabel('Value')
            plt.title('fill_factor List and Grad fill_factor List')
            plt.legend()

            # Plot loss_list
            plt.subplot(1, 3, 3)
            plt.plot(range(refinement_steps), loss_list, label='Loss')
            plt.xlabel('Refinement Step')
            plt.ylabel('Loss')
            plt.title('Loss at Each Step')
            plt.legend()
            
            plt.tight_layout()
            
            # save plot
            plt.savefig(f"{save_dir}/grads_{counter}_{post_fix}.png")
        
        def scale_to_0_1(img):
            img = img - img.min()
            img = img / img.max()
            return img
        
        dec_pose_refined, gen_image, x_list, y_list, grad_x_list, grad_y_list, loss_list, tv_loss_list, gen_image_list = model._refinement_step(patches_w_objs, all_z_objects, all_z_poses, gt_x=None, gt_y=None, gt_z=perturbed_z, fill_factor_gt=perturbed_fill)
        # save input_patches
        os.makedirs(f"{save_dir}/input_patches", exist_ok=True)
        save_image(scale_to_0_1(input_patches), f"{save_dir}/input_patches/{counter}_gt.png")
        plot_grads(x_list, grad_x_list, y_list, grad_y_list, gt_z, gt_fill, loss_list, tv_loss_list, post_fix="gt")
        save_image(scale_to_0_1(gen_image), f"{save_dir}/gen_image_{counter}_gt.png")
        dec_pose_refined, gen_image, x_list, y_list, grad_x_list, grad_y_list, loss_list, tv_loss_list, gen_image_list_2 = model._refinement_step(patches_w_objs, all_z_objects, all_z_poses)
        save_image(scale_to_0_1(input_patches), f"{save_dir}/input_patches/{counter}_pred.png")
        plot_grads(x_list, grad_x_list, y_list, grad_y_list, gt_z, gt_fill, loss_list, tv_loss_list, post_fix="pred")
        save_image(scale_to_0_1(gen_image), f"{save_dir}/gen_image_{counter}_pred.png")

        for i in range(len(gen_image_list)):
            os.makedirs(f"{save_dir}/gen_image_gt/{counter}", exist_ok=True)
            save_image(scale_to_0_1(gen_image_list[i]), f"{save_dir}/gen_image_gt/{counter}/pred_{i}.png")

        for i in range(len(gen_image_list_2)):
            os.makedirs(f"{save_dir}/gen_image_pred/{counter}", exist_ok=True)
            save_image(scale_to_0_1(gen_image_list_2[i]), f"{save_dir}/gen_image_pred/{counter}/pred_{i}.png")
        counter += 1

if __name__ == "__main__":
    main()
