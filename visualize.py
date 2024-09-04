# visualize.py
from src.util.misc import set_submodule_paths
set_submodule_paths(submodule_dir="submodules")
import math
import os
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
import torchvision.transforms as transforms
from ldm.util import instantiate_from_config
from train import get_data
from pytorch_lightning.utilities.seed import seed_everything

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

def get_perturbed_z_second_pose_lists(pose_gt, second_pose, num_perturbations=10, min_multiplier=0.75, max_multiplier=1.5):
    perturbed_poses = []

    # Unperturbed pose (original second pose)
    unperturbed_pose = second_pose.clone()
    perturbed_poses.append(unperturbed_pose)

    # Generate evenly spaced multipliers
    multipliers = torch.linspace(min_multiplier, max_multiplier, steps=num_perturbations)

    # Original image dimensions
    H, W = 256, 256

    # Perturb only the z-coordinate using evenly spaced multipliers
    z_min = -1.5
    z_max = 1.5
    # TODO: change fill factor!!
    z_crop_list = torch.linspace(z_min, z_max, steps=num_perturbations)
    for i, multiplier in enumerate(multipliers):
        perturbed_pose = second_pose.clone()

        # Use the compute_z_crop function to calculate the new z value
        # z_crop = compute_z_crop(H=H, H_crop=H * multiplier, x=perturbed_pose[0, 0], y=perturbed_pose[0, 1], z=perturbed_pose[0, 2], multiplier=multiplier)
        z_crop = z_crop_list[i]
        # Update the z value of the perturbed pose
        perturbed_pose[0, 2] = z_crop

        # Add the perturbed pose to the list
        perturbed_poses.append(perturbed_pose)
    
    # Combine all poses into a single tensor
    second_poses_lists = torch.cat(perturbed_poses, dim=0)
    
    return second_poses_lists

def load_model(config, ckpt_path):
    num_classes = len(config.data.params.train.params.label_names)
    config.model.params.lossconfig.params.num_classes = num_classes
    config.model.params.pose_decoder_config.params.num_classes = num_classes
    model = instantiate_from_config(config.model)
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


def get_second_pose_lists(pose_gt, second_pose, num_points=4, grid_size=5):
    # Generate points between (x1, y1) and (x2, y2)
    x_points = torch.linspace(pose_gt[0, 0], second_pose[0, 0], steps=num_points + 2)
    y_points = torch.linspace(pose_gt[0, 1], second_pose[0, 1], steps=num_points + 2)
    
    # Combine x and y into points
    points_between = torch.stack((x_points, y_points), dim=1)
    
    x_y_lists_between = second_pose[0, 2:].repeat(num_points + 2, 1)
    second_poses_lists_between = torch.cat((points_between, x_y_lists_between), dim=1)

    # Generate grid points between (-1, -1) and (+1, +1)
    grid_x = torch.linspace(-1, 1, steps=grid_size)
    grid_y = torch.linspace(-1, 1, steps=grid_size)
    
    # Create meshgrid of x and y
    grid_x, grid_y = torch.meshgrid(grid_x, grid_y)
    grid_points = torch.stack((grid_x.flatten(), grid_y.flatten()), dim=1)
    
    x_y_lists_grid = second_pose[0, 2:].repeat(grid_size * grid_size, 1)
    second_poses_lists_grid = torch.cat((grid_points, x_y_lists_grid), dim=1)
    
    # Combine the two lists (between points and grid points)
    second_poses_lists = torch.cat((second_poses_lists_between, second_poses_lists_grid), dim=0)
    
    return second_poses_lists
def save_output(output, output_path, rescale=False):
    if rescale:
        output = (output + 1.0) / 2.0
    output = output.permute(0,2,3,1).squeeze().detach()
    output = output.numpy()
    output = (output * 255).astype(np.uint8)
    output_image = transforms.ToPILImage()(output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_image.save(output_path)

def save_lists_output(pred_obj_lists, inference_pose_lists):
    for i, pred_obj in enumerate(pred_obj_lists):
        pose = inference_pose_lists[i].squeeze(0)
        x = round(pose[0].item(), 1)
        y = round(pose[1].item(), 1)
        z = round(pose[2].item(), 2)  # Store z value with higher precision
        image_path = f"./image/image_{x}_{y}_{z}.png"
        save_output(pred_obj, image_path, rescale=True)

def inference_second_pose_lists(model, batch, num_points=10):
    pred_obj_lists = []
    inference_pose_lists = []
    with torch.no_grad():
        rgb_in, rgb_gt, pose_gt, segm_mask_gt, mask_2d_bbox, class_gt, class_gt_label, bbox_gt, fill_factor_gt, second_pose = model.get_all_inputs(batch)
        save_output(rgb_in, "./image/input.png", True)
        save_output(rgb_gt, "./image/gt.png", True)
        # second_poses_lists = get_second_pose_lists(pose_gt, second_pose, num_points)
        second_poses_lists = get_perturbed_z_second_pose_lists(pose_gt, second_pose, num_points)
        for pose in second_poses_lists:
            pred_obj, dec_pose, posterior_obj, bbox_posterior, q_loss, ind_obj, ind_pose = model(rgb_in, pose_gt, second_pose=pose.unsqueeze(0))
            pred_obj_lists.append(pred_obj)
            inference_pose_lists.append(pose.unsqueeze(0))
        
        return pred_obj_lists, inference_pose_lists

def plot_images_side_by_side(inference_pose_lists, folder_path='./image/', num_images=6, idx=0):
    folder_path = Path(folder_path)

    image_paths = [
        folder_path / f"image_{round(item.squeeze(0)[0].item(), 1)}_{round(item.squeeze(0)[1].item(), 1)}_{round(item.squeeze(0)[2].item(), 2)}.png"
        for item in inference_pose_lists
    ]
    images = [Image.open(image_path) for image_path in image_paths]
    input_image = Image.open(folder_path / "input.png")
    gt = Image.open(folder_path / "gt.png")

    # Calculate the total number of images (including input and gt)
    total_images = len(images) + 2  # Adding 2 for input and gt

    # Calculate the number of rows and columns for a square-like grid
    cols = math.ceil(math.sqrt(total_images))
    rows = math.ceil(total_images / cols)

    # Create a figure with a subplot for each image
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axs = axs.flatten()

    axs[0].imshow(input_image)
    axs[0].axis('off')
    axs[0].set_title("input image")

    axs[1].imshow(gt)
    axs[1].axis('off')
    axs[1].set_title("ground truth image")

    # Plot each image
    for ax, image, path in zip(axs[2:], images, image_paths):
        ax.imshow(image)
        ax.axis('off')  # Turn off axis labels
        ax.set_title((path).name)  # Set the title to the image name

    # Hide any remaining empty subplots
    for ax in axs[total_images:]:
        ax.axis('off')

    plt.tight_layout()
    os.makedirs(folder_path / "plots", exist_ok=True)
    plt.savefig(f"{folder_path}/plots/plot_{idx}.png")

def main():
    seed = 42
    seed_everything(seed)
    config_path = "configs/autoencoder/zoom/learnt_zoom_pixel_3.yaml"
    checkpoint_path = "logs/2024-09-01T20-59-14_learnt_zoom_pixel_3/checkpoints/last.ckpt"
    config = OmegaConf.load(config_path)
    model = load_model(config, checkpoint_path)
    model.eval()

    data = get_data(config)

    iteration = iter(data.datasets['validation'])
    idx = 3
    for _ in range(idx):
        batch = next(iteration)

    while True:
        for key in batch:
            if isinstance(batch[key], float) or isinstance(batch[key], int):
                batch[key] = torch.tensor([batch[key]])
            elif isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].unsqueeze(0)
        pred_obj_lists, inference_pose_lists = inference_second_pose_lists(model, batch)
        save_lists_output(pred_obj_lists, inference_pose_lists)
        plot_images_side_by_side(inference_pose_lists, num_images=len(pred_obj_lists), idx=idx)
        batch = next(iteration)
        idx +=1
if __name__ == "__main__":
    main()
