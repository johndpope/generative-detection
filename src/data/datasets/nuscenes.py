"""
src/data/nuscenes.py
"""
import os
import logging
import math
import random
import pickle as pkl
import numpy as np
import torch
from PIL import Image
from PIL.Image import Resampling
from mmdet3d.registry import DATASETS
from mmdet3d.datasets.nuscenes_dataset import NuScenesDataset as MMDetNuScenesDataset
from pytorch3d.transforms import euler_angles_to_matrix, se3_log_map
import torchvision.transforms as T
import torchvision.ops as ops
from torchvision.transforms.functional import pil_to_tensor
from src.util.misc import EasyDict as edict
from src.util.cameras import PatchPerspectiveCameras as PatchCameras
from src.util.cameras import z_world_to_learned
from src.data.specs import LABEL_NAME2ID, LABEL_ID2NAME, CAM_NAMESPACE, POSE_DIM, LHW_DIM, BBOX_3D_DIM

CAM_NAMESPACE = 'CAM'
CAMERAS = ["FRONT", "FRONT_RIGHT", "FRONT_LEFT", "BACK", "BACK_LEFT", "BACK_RIGHT"]
CAMERA_NAMES = [f"{CAM_NAMESPACE}_{camera}" for camera in CAMERAS]
CAM_NAME2CAM_ID = {cam_name: i for i, cam_name in enumerate(CAMERA_NAMES)}
CAM_ID2CAM_NAME = dict(enumerate(CAMERA_NAMES))

Z_NEAR = 0.01
Z_FAR = 60.0

NUSC_IMG_WIDTH = 1600
NUSC_IMG_HEIGHT = 900

PATCH_ANCHOR_SIZES = [50, 100, 200, 400]

class NuScenesBase(MMDetNuScenesDataset):
    """A class representing a dataset for NuScenes object detection.

    Args:
        data_root (str): The root directory of the dataset.
        label_names (list): A list of label names to be used for object detection.
        patch_height (int): The height of the patch to be extracted from the images.
        patch_aspect_ratio (float): The aspect ratio of the patch to be extracted from the images.
        is_sweep (bool): Whether the dataset contains sweep data.
        perturb_center (bool): Whether to perturb the center of the patch.
        perturb_scale (bool): Whether to perturb the scale of the patch.
        negative_sample_prob (float): The probability of sampling negative samples.
        h_minmax_dir (str): The directory containing the hmin and hmax dictionaries.
        perturb_prob (float): The probability of perturbing the patch.
        patch_center_rad_init (float): The initial radius of the patch center.
        perturb_yaw (bool): Whether to perturb the yaw of the patch.
        allow_zoomout (bool): Whether to allow zooming out of the patch.
        **kwargs: Additional keyword arguments.

    Attributes:
        data_root (str): The root directory of the dataset.
        img_root (str): The directory containing the images.
        label_names (list): A list of label names to be used for object detection.
        allow_zoomout (bool): Whether to allow zooming out of the patch.
        label_ids (list): A list of label IDs corresponding to the label names.
        patch_aspect_ratio (float): The aspect ratio of the patch to be extracted from the images.
        patch_size_return (tuple): The size of the patch to be returned.
        shift_center (bool): Whether to shift the center of the patch.
        label_id2class_id (dict): A mapping from label IDs to class IDs.
        class_id2label_id (dict): A mapping from class IDs to label IDs.
        hmin_dict (dict): A dictionary containing the hmin values.
        hmax_dict (dict): A dictionary containing the hmax values.
        negative_sample_prob (float): The probability of sampling negative samples.
        perturb_yaw (bool): Whether to perturb the yaw of the patch.
        perturb_z (bool): Whether to perturb the scale of the patch.
        DEBUG (bool): Whether to enable debug mode.
        perturb_prob (float): The probability of perturbing the patch.
        patch_center_rad_init (torch.Tensor): The initial radius of the patch center.
        patch_center_rad (torch.Tensor): The current radius of the patch center.
        count (int): The number of samples in the dataset.

    Methods:
        __len__(): Returns the total number of samples in the dataset.
        _get_patch_dims(bbox, patch_center_2d, img_size): Calculates the dimensions of the patch based on the bounding box and patch center.
        _get_instance_mask(bbox, bbox_patch, patch): Generates a mask for the patch based on the bounding box.
        _get_instance_patch(img_path, cam_instance): Retrieves the patch and related information for a given image and camera instance.
        compute_z_crop(H, H_crop, x, y, z, multiplier, eps): Computes the z crop for the patch.
        get_min_max_multipliers(patch_size_original): Returns the minimum and maximum multipliers for the patch size.

    """
    def __init__(self, data_root, label_names, patch_height=256, patch_aspect_ratio=1.,
                is_sweep=False, perturb_center=True, perturb_scale=False, 
                negative_sample_prob=0.5, h_minmax_dir = "dataset_stats/combined", 
                perturb_prob=0.0, patch_center_rad_init=0.5, 
                perturb_yaw=False, allow_zoomout=False,
                 **kwargs):
        # Setup directory
        self.data_root = data_root
        self.img_root = os.path.join(data_root, "sweeps" if is_sweep else "samples")
        super().__init__(data_root=data_root, **kwargs)
        # Setup class labels and ids
        self.label_names = label_names
        self.allow_zoomout = allow_zoomout
        self.label_ids = [LABEL_NAME2ID[label_name] for label_name in LABEL_NAME2ID.keys() if label_name in label_names]
        logging.info("Using label names: %s, label ids: %s", self.label_names, self.label_ids)
        # Setup patch
        self.patch_aspect_ratio = patch_aspect_ratio
        self.patch_size_return = (patch_height, int(patch_height * patch_aspect_ratio)) # aspect ratio is width/height
        # Setup patch shifts and scale
        self.shift_center = perturb_center if self.split != "test" else False
        # Define mapping from nuscenes label ids to our label ids depending on num of classes we predict
        self.label_id2class_id = {label : i for i, label in enumerate(self.label_ids)}  
        self.class_id2label_id = {v: k for k, v in self.label_id2class_id.items()}
        # Load hmin and hmax dicts
        hmin_path = os.path.join(h_minmax_dir, "hmin.pkl")
        hmax_path = os.path.join(h_minmax_dir, "hmax.pkl")
        with open(hmin_path, "rb") as f:
            self.hmin_dict = pkl.load(f)
        with open(hmax_path, "rb") as f:
            self.hmax_dict = pkl.load(f)
        # Set sampling probability for negative samples
        self.negative_sample_prob = negative_sample_prob if "background" in self.label_names else 0.0
        self.perturb_yaw = perturb_yaw
        self.perturb_z = perturb_scale if self.split != "test" else False
        self.DEBUG = False
        self.perturb_prob = perturb_prob
        self.patch_center_rad_init = torch.tensor(patch_center_rad_init)
        self.patch_center_rad = self.patch_center_rad_init # default 0.5
        assert not self.patch_center_rad.is_cuda, "self.patch_center_rad.device must be on cpu"
        self.count = 0
    def __len__(self):
        self.num_samples = super().__len__()
        self.num_cameras = len(CAMERA_NAMES)
        return self.num_samples * self.num_cameras
    
    def _get_patch_dims(self, bbox, patch_center_2d, img_size):
        # Object BBOX
        center_pixel_loc = np.round(patch_center_2d).astype(np.int32)
        x1, y1, x2, y2 = np.round(bbox).astype(np.int32) 
        width_2dbbox = x2 - x1
        height_2dbbox = y2 - y1
        # get max side length
        max_dim = max(width_2dbbox, height_2dbbox)
        
        # Get closest pre-defined patch size
        dim_diffs = [abs(max_dim - patch_size) for patch_size in PATCH_ANCHOR_SIZES]
        patch_size = PATCH_ANCHOR_SIZES[dim_diffs.index(min(dim_diffs))]
        
        # Define crop dimension for patch from image
        x1 = center_pixel_loc[0] - patch_size // 2
        y1 = center_pixel_loc[1] - patch_size // 2
        x2 = center_pixel_loc[0] + patch_size // 2
        y2 = center_pixel_loc[1] + patch_size // 2
        
        # Handle Out of Bounds Edge Cases
        if x2 >= img_size[0] or y2 >= img_size[1] or x1 <= 0 or y1 <= 0:
            # Move patch back into image
            delta_x = (max(0, x1) - x1) + (min(img_size[0], x2) - x2)
            delta_y = (max(0, y1) - y1) + (min(img_size[1], y2) - y2)

            x1 = x1 + delta_x
            y1 = y1 + delta_y
            x2 = x2 + delta_x
            y2 = y2 + delta_y
        
        patch_center_2d = (x1 + patch_size // 2, y1 + patch_size // 2)
        
        # Get pixels in y direction not covered by object
        padding_pixels = max(patch_size - height_2dbbox, 0)
        
        assert np.abs(x1 - x2) == np.abs(y1 - y2), f"Patch is not a square: {x1, y1, x2, y2}"
        assert np.abs(x1 - x2) in PATCH_ANCHOR_SIZES and np.abs(y1 - y2) in PATCH_ANCHOR_SIZES, f"Patch size is not in PATCH_ANCHOR_SIZES: {x1, y1, x2, y2}"
        
        return [x1, y1, x2, y2], patch_center_2d, padding_pixels
        
    def _get_instance_mask(self, bbox, bbox_patch, patch):
        # create a boolean mask for patch with gt 2d bbox as coordinates (x1, y1, x2, y2)
        mask_bool = np.zeros((patch.size[1], patch.size[0]), dtype=bool)
        
        pixel_delta = (np.array(bbox) - np.array(bbox_patch)).astype(np.int32)
        pixel_delta[:2] = np.maximum(pixel_delta[:2], 0)
                
        # Set the area inside the bounding box to True
        mask_bool[pixel_delta[1]:patch.size[1]+pixel_delta[3], pixel_delta[0]:patch.size[0]+pixel_delta[2]] = 1
        mask_pil = Image.fromarray(mask_bool)
        return mask_pil.resize((self.patch_size_return[0], self.patch_size_return[1]), resample=Resampling.NEAREST, reducing_gap=1.0)
    
    def _get_instance_patch(self, img_path, cam_instance):
        # return croped list of images as defined by 2d bbox for each instance
        # load image from path using PIL
        img_pil = Image.open(img_path)
        if img_pil is None:
            raise FileNotFoundError(f"Image not found at {img_path}")
        
        # bounding box annotation (exterior rectangle of the projected 3D box), a list arrange as [x1, y1, x2, y2].
        bbox = cam_instance.bbox 
        # Projected center location on the image, a list has shape (2,)
        patch_center_2d = cam_instance.patch_center_2d 
        
        if self.DEBUG:
            # colorize center of 2d bbox
            img_pil.putpixel((np.clip(int(cam_instance.patch_center_2d_original[0]), 0, img_pil.size[0]-1),
                            np.clip(int(cam_instance.patch_center_2d_original[1]), 0, img_pil.size[1]-1)), (255, 0, 0))
        
        # If center_2d is out bounds, return None, None, None, None since < 50% of the object is visible
        if patch_center_2d[0] < 0 or patch_center_2d[1] < 0 or patch_center_2d[0] >= img_pil.size[0] or patch_center_2d[1] >= img_pil.size[1]:
            return None, None, None, None, None, None
        
        # Crop Patch from Image around 2D BBOX
        patch_box, patch_center_2d, padding_pixels = self._get_patch_dims(bbox, patch_center_2d, img_pil.size)
        if self.DEBUG:
            # colorize center of patch
            img_pil.putpixel((int(patch_center_2d[0]), int(patch_center_2d[1])), (0, 0, 255))
        
        patch = img_pil.crop(patch_box) # left, upper, right, lowe
        patch_size_anchor = torch.tensor(patch.size, dtype=torch.float32)
        
        # Ratio of original image to resized image
        resampling_factor = (self.patch_size_return[0] / patch.size[0], self.patch_size_return[1] / patch.size[1])
        assert resampling_factor[0] == resampling_factor[1], "resampling factor of width and height must be the same but they are not."
        # Resize Patch to Size expected by Encoder
        patch_resized = patch.resize((self.patch_size_return[0], self.patch_size_return[1]), resample=Resampling.BILINEAR, reducing_gap=1.0)
                
        transform = T.Compose([T.ToTensor()])
        patch_resized_tensor = transform(patch_resized)
        mask = self._get_instance_mask(bbox, patch_box, patch)
        mask = transform(mask)
        
        if resampling_factor is None:
            a=0
        padding_pixels_resampled = padding_pixels * resampling_factor[0]
        
        return patch_resized_tensor, patch_center_2d, patch_size_anchor, resampling_factor, padding_pixels_resampled, mask
    
    def compute_z_crop(self, H, H_crop, x, y, z, multiplier, eps=1e-8):
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

    def get_min_max_multipliers(self, patch_size_original):
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
    
    def get_perturbed_depth_crop(self, pose_6d, original_crop, fill_factor, patch_size_original, original_mask, p=0.95):
        
        x, y, z = pose_6d[:, 0], pose_6d[:, 1], pose_6d[:, 2]
        _, H, W = original_crop.shape

        m_min, m_max = self.get_min_max_multipliers(patch_size_original)
        
        max_zoom_mult = max(x, y, m_min) # since each anchor is 2x the previous
        min_zoom_mult = m_max if self.allow_zoomout else 1.0 # no zoomout

        if np.random.rand() < p:
            # Sample from a uniform distribution
            multiplier = np.random.uniform(low=max_zoom_mult, high=min_zoom_mult)
        else:
            # Sample from a Gaussian distribution
            mean = 1.0
            std_dev = (min_zoom_mult - max_zoom_mult) / 6.0  # std_dev such that 99.7% values are within the range
            std_dev = max(std_dev, 0.01)  # minimum std_dev is 0.01
            multiplier = np.random.normal(loc=mean, scale=std_dev)

        x_crop = x / multiplier
        y_crop = y / multiplier
        H_crop = H * multiplier
        W_crop = H_crop * self.patch_aspect_ratio

        z_crop = self.compute_z_crop(H, H_crop, x, y, z, multiplier)
        fill_factor_cropped = fill_factor * multiplier 

        center_cropper = T.CenterCrop((int(H_crop), int(W_crop)))
        original_crop_recropped = center_cropper(original_crop) 
        resizer = T.Resize(size=(self.patch_size_return[0], self.patch_size_return[1]),interpolation=T.InterpolationMode.BILINEAR)
        original_crop_recropped_resized = resizer(original_crop_recropped)
        
        pose_6d[:, 0] = x_crop
        pose_6d[:, 1] = y_crop
        pose_6d[:, 2] = z_crop

        original_mask_zoomed = center_cropper(original_mask) 
        original_mask_zoomed_resized = resizer(original_mask_zoomed)
        
        return pose_6d, original_crop_recropped_resized, fill_factor_cropped, original_mask_zoomed_resized, multiplier
        
    def _get_pose_6d_lhw(self, camera, cam_instance):
        
        padding_pixels_resampled = cam_instance.fill_factor * self.patch_size_return[0]
        x, y, z, l, h, w, yaw = cam_instance.bbox_3d # in camera coordinate system? need to convert to patch NDC
        roll, pitch = 0.0, 0.0 # roll and pitch are 0 for all instances in nuscenes dataset
        
        object_centroid_3D = (x, y, z)
        patch_center_2d = cam_instance.patch_center_2d
        
        if len(patch_center_2d) == 2:
            # add batch dimension
            patch_center_2d = torch.tensor(patch_center_2d, dtype=torch.float32).unsqueeze(0)

        object_centroid_3D = torch.tensor(object_centroid_3D, dtype=torch.float32).reshape(1, 1, 3)
        
        assert object_centroid_3D.dim() in [3, 2], f"object_centroid_3D dim is {object_centroid_3D.dim()}"        
        assert isinstance(object_centroid_3D, torch.Tensor), "object_centroid_3D is not a torch tensor"

        point_patch_ndc = camera.transform_points_patch_ndc(points=object_centroid_3D,
                                                            patch_size=cam_instance.patch_size, # add delta
                                                            patch_center=patch_center_2d,
                                                            cam_instance=cam_instance) # add scale
        
        if self.DEBUG:
            debug_patch_img = T.ToPILImage()(cam_instance.patch)
            p_w = debug_patch_img.size[0]
            p_h = debug_patch_img.size[1]
            # colorize center of 3d bbox

            is_outofbounds = point_patch_ndc[0] < -1 or point_patch_ndc[0] > 1 or point_patch_ndc[1] < -1 or point_patch_ndc[1] > 1
            # only colorize if point_patch_ndc is within bounds
            if not is_outofbounds:
                debug_patch_img.putpixel((int(point_patch_ndc[0] * p_w/2 + p_w/2), int(point_patch_ndc[1] * p_h/2 + p_h/2)), (0, 255, 0))
        else:
            debug_patch_img = None
        
        z_world = z
        
        def get_zminmax(min_val, max_val, focal_length, patch_height):
            zmin = -(min_val * focal_length.squeeze()[0]) / (patch_height - padding_pixels_resampled)
            zmax = -(max_val * focal_length.squeeze()[0]) / (patch_height - padding_pixels_resampled)
            return zmin, zmax
        bbox_label_idx = cam_instance.bbox_label
        bbox_label_name = LABEL_ID2NAME[bbox_label_idx]
        assert bbox_label_name != "background", "cannot get zminmax for background class"
        min_val = self.hmin_dict[bbox_label_name]
        max_val = self.hmax_dict[bbox_label_name]
        
        zmin, zmax = get_zminmax(min_val=min_val, max_val=max_val, 
                                focal_length=camera.focal_length, 
                                patch_height=self.patch_size_return[0])
        
        z_learned = z_world_to_learned(z_world=z_world, zmin=zmin, zmax=zmax, 
                                    patch_resampling_factor=cam_instance.resampling_factor[0])
        
        if point_patch_ndc.dim() == 3:
            point_patch_ndc = point_patch_ndc.view(-1)
        x_patch, y_patch, _ = point_patch_ndc
        
        euler_angles = torch.tensor([pitch, roll, yaw], dtype=torch.float32)
        convention = "XYZ"
        R = euler_angles_to_matrix(euler_angles, convention)
        
        translation = torch.tensor([x_patch, y_patch, z_learned], dtype=torch.float32)
        # A SE(3) matrix has the following form: ` [ R 0 ] [ T 1 ] , `
        se3_exp_map_matrix = torch.eye(4)
        se3_exp_map_matrix[:3, :3] = R 
        se3_exp_map_matrix[:3, 3] = translation

        # form must be ` [ R 0 ] [ T 1 ] , ` so need to transpose
        se3_exp_map_matrix = se3_exp_map_matrix.T
        
        # add batch dim if not present
        if se3_exp_map_matrix.dim() == 2:
            se3_exp_map_matrix = se3_exp_map_matrix.unsqueeze(0)
        
        try:
            pose_6d = se3_log_map(se3_exp_map_matrix) # 6d vector
        except Exception as e:
            logging.info("Error in se3_log_map", e)
            return None, None
        
        # l and w pred as aspect ratio wrt h
        l_pred = l / h
        h_pred = h
        w_pred = w / h
        
        bbox_sizes = torch.tensor([l_pred, h_pred, w_pred], dtype=torch.float32)
        
        # remove v1, v2 from pose_6d = t1, t2, t3, v1, v2, v3
        pose_6d_no_v1v2 = torch.zeros_like(pose_6d[:, :POSE_DIM])
        pose_6d_no_v1v2[:, :3] = pose_6d[:, :POSE_DIM-1]
        # set v3 as last val
        pose_6d_no_v1v2[:, -1] = pose_6d[:, -1]
        
        # Manually set translation
        pose_6d_no_v1v2[:, :3] = translation
        
        return pose_6d_no_v1v2, bbox_sizes, yaw, debug_patch_img
    
    def _get_shifted_patch_center(self, center_2d, bbox, p=0.7, eps=1e-8):
        x1, y1, x2, y2 = bbox
        center_x, center_y = center_2d

        # Calculate width and height of the bounding box
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        
        # Calculate the maximum shift distance
        assert not self.patch_center_rad.is_cuda, "self.patch_center_rad.device must be on cpu"
        r_max_shift = self.patch_center_rad * min(bbox_width, bbox_height)
        
        # Random perturbation in x direction within the max perturbation limit
        r_max_shift = r_max_shift.clone().detach().cpu().numpy()

        # Randomly decide between uniform or Gaussian shift based on probability p
        if np.random.rand() < p:
            # Uniform distribution for shift
            x_shifted = np.random.uniform(-r_max_shift, r_max_shift)
        
            # Calculate corresponding y perturbation to maintain visibility condition
            max_y_shift = np.sqrt(abs(r_max_shift**2 - x_shifted**2) + eps)
            y_shifted = np.random.uniform(-max_y_shift, max_y_shift)
        else:
            # Gaussian distribution for shift (close to 0)
            x_shifted = np.random.normal(loc=0, scale=r_max_shift / 4)
            
            # Calculate corresponding y perturbation to maintain visibility condition
            max_y_shift = np.sqrt(abs(r_max_shift**2 - x_shifted**2) + eps)
            y_shifted = np.random.normal(loc=0, scale=max_y_shift / 4)
        
        # Perturbed center coordinates
        shifted_center_x = self._get_shifted_coord(center_x, x_shifted)
        shifted_center_y = self._get_shifted_coord(center_y, y_shifted) 
        return [shifted_center_x, shifted_center_y]

    def _get_shifted_coord(self, center, shifted):
        # Perturbed center coordinates
        assert not math.isnan(center)
        shifted = 0.0 if math.isnan(shifted) else shifted
        return int(center + shifted)
    
    def _get_patchGT(self, cam_instance, img_path, cam2img, postfix=""):
        cam_instance = edict(cam_instance)
        
        # Mask needs 
        cam_instance.patch_center_2d_original = cam_instance.center_2d
        if self.shift_center:
            # Random shift of 2D BBOX Center
            shifted_patch_center = self._get_shifted_patch_center(cam_instance.center_2d, cam_instance.bbox)
            # Replace Center 2D with shifted center inside image bounds
            shifted_patch_center = [np.clip(int(shifted_patch_center[0]), 0, NUSC_IMG_WIDTH-1), np.clip(int(shifted_patch_center[1]), 0, NUSC_IMG_HEIGHT-1)]
            cam_instance.center_2d = shifted_patch_center
            cam_instance.patch_center_2d = shifted_patch_center
        else:
            cam_instance.patch_center_2d = cam_instance.center_2d
        
        # Crop patch from origibal image
        patch, patch_center_2d, patch_size_original, resampling_factor, padding_pixels_resampled, mask_2d_bbox = self._get_instance_patch(img_path, cam_instance)
        
        if patch is None or patch_size_original is None:
            return None
        
        # Compute fill factor in pixels
        fill_factor = padding_pixels_resampled / self.patch_size_return[0]

        # if no batch dimension, add it
        if patch_size_original.dim() == 1:
            patch_size_original = patch_size_original.unsqueeze(0)

        # Setup Patch Camera
        image_size = [(NUSC_IMG_HEIGHT, NUSC_IMG_WIDTH)]
        
        # make 4x4 matrix from 3x3 camera matrix
        K = torch.zeros(4, 4, dtype=torch.float32)
        
        K[:3, :3] = torch.tensor(cam2img, dtype=torch.float32)
        
        K[2, 2] = 0.0
        K[2, 3] = 1.0
        K[3, 2] = 1.0
        K = K.clone().unsqueeze(0) # add batch dimension
        
        principal_point = K[..., :2, 2] # shape (1, 2)
        # negate focal length
        focal_length = -K[..., 0, 0] # shape (1,)
        
        # Create Camera Object
        camera = PatchCameras(
            focal_length=focal_length,
            principal_point=principal_point,
            znear=Z_NEAR,
            zfar=Z_FAR,
            device="cpu",
            image_size=image_size)
        
        # Update Cam Instance with new patch and center_2d
        cam_instance.update({'patch': patch,
                            'patch_center': patch_center_2d,
                            'center_2d': patch_center_2d,
                            'patch_size': patch_size_original,
                            'resampling_factor': resampling_factor,
                            'fill_factor': fill_factor,
                            'mask_2d_bbox': mask_2d_bbox})
        
        pose_6d, bbox_sizes, yaw, debug_patch_img = self._get_pose_6d_lhw(camera, cam_instance)
        if debug_patch_img is not None and self.DEBUG:
            cam_instance.patch = T.ToTensor()(debug_patch_img)
        
        zoom_multiplier = 1.0
        if self.perturb_z and (postfix != ""):
            pose_6d_new, patch_new, fill_factor_new, mask_2d_bbox_new, zoom_multiplier = self.get_perturbed_depth_crop(pose_6d, patch, fill_factor, patch_size_original, mask_2d_bbox)
            pose_6d = pose_6d_new.reshape_as(pose_6d)
            patch = patch_new.reshape_as(patch)
            mask_2d_bbox = mask_2d_bbox_new.reshape_as(mask_2d_bbox)
            fill_factor = float(fill_factor_new)
            
            cam_instance.update({'patch': patch,
                                'fill_factor': fill_factor,
                                'mask_2d_bbox':mask_2d_bbox})
        
        cam_instance.zoom_multiplier = torch.tensor(zoom_multiplier).squeeze()
        
        cam_instance.pose_6d, cam_instance.bbox_sizes, cam_instance.yaw = pose_6d, bbox_sizes, yaw

        if cam_instance.pose_6d is None or cam_instance.bbox_sizes is None:
            return None
        
        cam_instance.pose_6d = cam_instance.pose_6d.reshape([4])

        # Store Camera Parameters
        cam_instance.camera_params = {
            "focal_length": focal_length,
            "principal_point": principal_point,
            "znear": Z_NEAR,
            "device": "cpu",
            "zfar": Z_FAR,
            "image_size": image_size,
        }
        
        # Extract Class ID
        cam_instance = self._extract_class_id(cam_instance, cam_instance.bbox_label)
        return cam_instance

    def _extract_class_id(self, cam_instance, class_id):
        cam_instance.class_id = self.label_id2class_id[class_id]
        cam_instance.class_name = LABEL_ID2NAME[class_id]
        cam_instance.original_class_id = class_id
        return cam_instance

    def get_background_patch(self, background_patch):
        ret = edict()
        background_patch_original_size = background_patch.size
        background_patch = background_patch.resize(self.patch_size_return, resample=Resampling.BILINEAR)
        transform = T.Compose([T.ToTensor()])
        background_patch_tensor = transform(background_patch)
        ret.patch = background_patch_tensor
        ret.pose_6d = torch.zeros(POSE_DIM, dtype=torch.float32).squeeze(0)
        ret.bbox_sizes = torch.zeros(LHW_DIM, dtype=torch.float32)
        ret.bbox_3d_gt = torch.zeros(BBOX_3D_DIM, dtype=torch.float32)
        ret.resampling_factor = torch.tensor((self.patch_size_return[0] / background_patch_original_size[0],self.patch_size_return[1] / background_patch_original_size[1]))
        
        ret.yaw = 0.0
        ret.fill_factor = 0.0
        mask_2d_bbox = torch.zeros(self.patch_size_return[0],self.patch_size_return[1], dtype=torch.float32)
        if mask_2d_bbox.dim() == 2:
            mask_2d_bbox = mask_2d_bbox.unsqueeze(0)
        ret.mask_2d_bbox = mask_2d_bbox
        
        # Extract Class ID
        return self._extract_class_id(ret, LABEL_NAME2ID['background'])
    
    def __get_new_item__(self, idx):
        # iter next sample if no instances present
        # prevent querying idx outside of bounds
        if idx + 1 >= self.__len__():
            return self.__getitem__(0)
        else:
            return self.__getitem__(idx + 1)
        
    
    def __getitem__(self, idx):
        ret = edict()
        
        sample_idx = idx // self.num_cameras
        cam_idx = idx % self.num_cameras
        sample_info = super().__getitem__(sample_idx)
        cam_name = CAM_ID2CAM_NAME[cam_idx]
        ret.sample_idx = sample_idx
        ret.cam_idx = cam_idx
        ret.cam_name = cam_name
        # Get image info for the selected camera
        sample_img_info = edict(sample_info['images'][cam_name])
        ret.update(sample_img_info)
        
        img_file = sample_img_info.img_path.split("/")[-1]
        
        if self.split == 'test':
            intrinsics = torch.tensor(sample_img_info.cam2img)
            ret.camera_params = {
                "focal_length": -intrinsics[..., 0, 0],
                "principal_point": intrinsics[..., :2, 2],
                "znear": Z_NEAR,
                "zfar": Z_FAR,
                "device": "cpu",
                "image_size": [(NUSC_IMG_HEIGHT, NUSC_IMG_WIDTH)]}

            full_img_path = os.path.join(self.img_root, cam_name, img_file)
            image_crops, patch_centers, patch_sizes = self._get_all_image_crops(full_img_path)
            ret.patch = image_crops
            ret.patch_size = patch_sizes
            ret.patch_center_2d = patch_centers 
            return ret
        
        # List of dicts for each instance in the current camera image
        cam_instances = sample_info['cam_instances'][cam_name]
        # Extract object bbox, class and filter out instances not in label_names
        cam_instances = [cam_instance for cam_instance in cam_instances if (cam_instance['bbox_label'] in self.label_ids and 
                (cam_instance['center_2d'][0] > 0. and cam_instance['center_2d'][0] < NUSC_IMG_WIDTH))]
        
        if np.random.rand() <= (1. - self.negative_sample_prob):
            # Get a random crop of an instance with at least 50% overlap
            if not cam_instances:

                return self.__get_new_item__(idx)
            
            # Choose one random object from all objects in the image
            random_idx = np.random.randint(0, len(cam_instances))
            cam_instance = cam_instances[random_idx]
            
            # Get cropped image patch and 6d pose for the object instance       
            patch_obj = self._get_patchGT(cam_instance,
                                        img_path=os.path.join(self.img_root, cam_name, img_file),
                                        cam2img=sample_img_info.cam2img,
                                        postfix="")
            if patch_obj is None:
                return self.__get_new_item__(idx)
            ret.bbox_3d_gt = patch_obj.bbox_3d
            ret.update(patch_obj)

            ret.zoom_multiplier = patch_obj.zoom_multiplier
            # get a second crop of the same object instance again
            # TODO: Make optional
            # only get second crop with probability perturb_prob, else copy first crop dict to second crop dict
            # if np.random.rand() <= self.perturb_prob:
            patch_obj_2 = self._get_patchGT(cam_instance,
                                            img_path=os.path.join(self.img_root, cam_name, img_file),
                                            cam2img=sample_img_info.cam2img,
                                            postfix="_2")
            ret.update({f"{k}_2": v for k,v in patch_obj_2.items()})
            ret.bbox_3d_gt_2 = patch_obj_2.bbox_3d
            if patch_obj is None or patch_obj_2 is None:
                return self.__get_new_item__(idx)

            ret.zoom_multiplier_2 = patch_obj_2.zoom_multiplier

            patch_center_2d = torch.tensor(patch_obj.center_2d).float()
            ret.patch_center_2d = patch_center_2d
            
            for key, value in ret.camera_params.items():
                ret[key] = value
    
        else:  # get random crop without overlap
            # bbox = [x1, y1, x2, y2]
            bbox_2d_list = [cam_instance['bbox'] for cam_instance in cam_instances]
            # get random crop from image, and make sure it does not overlap with any bbox
            img_path = os.path.join(self.img_root, cam_name, img_file)
            img_pil = Image.open(img_path)
            if img_pil is None:
                raise FileNotFoundError(f"Image not found at {img_path}")
            
            background_patch, center_2d = self.get_random_crop_without_overlap(img_pil, bbox_2d_list, PATCH_ANCHOR_SIZES)
            if background_patch is None:
                return self.__get_new_item__(idx)

            ret.class_id = self.label_id2class_id[LABEL_NAME2ID['background']]

            ret.original_class_id = LABEL_NAME2ID['background']
            ret.class_name = LABEL_ID2NAME[LABEL_NAME2ID['background']]
            ret.patch_size = torch.tensor(self.patch_size_return, dtype=torch.float32).unsqueeze(0) # torch.Size([1, 2])
            ret.patch_center_2d = torch.tensor([self.patch_size_return[0] // 2,self.patch_size_return[1] // 2], dtype=torch.float32)
            # ret.pose_6d_perturbed = torch.zeros(POSE_DIM, dtype=torch.float32).unsqueeze(0)
            # ret.yaw_perturbed = 0.0
            ret.bbox_label = LABEL_NAME2ID['background']
            camera_params = {
                "focal_length": torch.tensor([0.0], dtype=torch.float32),
                "principal_point": torch.tensor([[0.0, 0.0]], dtype=torch.float32),
                "znear": 0.0,
                "device": "cpu",
                "zfar": 0.0,
                "image_size": torch.tensor([(0, 0)], dtype=torch.float32),
            }
            ret.update(camera_params)
        
            background_patch_dict = self.get_background_patch(background_patch)
            if background_patch_dict is None:
                return self.__get_new_item__(idx)

            ret.update(background_patch_dict)
            ret.update({f"{k}_2": v for k, v in background_patch_dict.items()})

            patch_center_2d = torch.tensor(center_2d).float()
            ret.patch_center_2d = patch_center_2d

            for key in ['attr_label', 'attr_label_2', 'bbox', 'bbox_2', 'bbox_3d', 'bbox_3d_2', 'bbox_3d_isvalid', \
                        'bbox_3d_isvalid_2', 'bbox_label_2', 'bbox_label_3d', 'bbox_label_3d_2', 'camera_params', \
                        'camera_params_2', 'center_2d', 'center_2d_2', 'depth', 'depth_2', 'patch_center', 'patch_center_2', \
                        'patch_center_2d_2', 'patch_center_2d_original', 'patch_center_2d_original_2', 'velocity', 'velocity_2', \
                        'zoom_multiplier', 'zoom_multiplier_2', 'patch_size_2']:
                ret[key] = self.default_value(key)
            
        for key in ["cam_name", "img_path", "sample_data_token", "cam2img", "cam2ego", "class_name", "bbox_3d_gt_2", "lidar2cam", "bbox_3d_gt", "resampling_factor", "resampling_factor_2", "device", "image_size"]:
            value = ret[key]
            if isinstance(value, list) or isinstance(value, tuple):
                ret[key] = torch.tensor(value, dtype=torch.float32)
        self.count += 1
        return ret
    
    def get_random_crop_without_overlap(self, img_pil, bbox_2d_list, patch_sizes):
        width, height = img_pil.size
        bbox_tensor = torch.tensor(bbox_2d_list, dtype=torch.float)
        is_found = False
        timeout_iters = 10
        iters = 0
        while not is_found and iters < timeout_iters:
            # Randomly choose a patch size
            patch_size = random.choice(patch_sizes)
            crop_width = crop_height = patch_size
            
            # Randomly select the top-left corner of the crop
            crop_x = random.randint(0, width - crop_width)
            crop_y = random.randint(0, height - crop_height)
            
            # Define the crop as a bounding box
            crop_box = torch.tensor([[crop_x, crop_y, crop_x + crop_width, crop_y + crop_height]], dtype=torch.float)

            # Calculate IoU between the crop box and all bounding boxes
            if len(bbox_tensor) == 0: # no instances in the image, so any crop is valid
                is_found = True
                break
            iou = ops.box_iou(crop_box, bbox_tensor)
            
            # Check if there is a 50% overlap with any bounding box
            if torch.all(iou < 0.5):
                is_found = True
            
            iters += 1
            if iters >= timeout_iters:
                return None
            
        center_2d = torch.tensor([crop_x + crop_width // 2, crop_y + crop_height // 2], dtype=torch.float)

        return img_pil.crop((crop_x, crop_y, crop_x + crop_width, crop_y + crop_height)), center_2d
    
    def _get_all_image_crops(self, image_path):
        all_patches = []
        all_patch_centers = []
        all_patch_sizes = []
        for patch_size in PATCH_ANCHOR_SIZES:
            patches_res_i, patch_center_2d_res_i, patch_size_res_i = self._crop_image_into_squares(image_path, patch_size)
            all_patches.append(patches_res_i)
            all_patch_centers.append(patch_center_2d_res_i)
            all_patch_sizes.append(patch_size_res_i)        
        return torch.cat(all_patches, dim=0), torch.cat(all_patch_centers, dim=0), torch.cat(all_patch_sizes, dim=0)
    
    def _crop_image_into_squares(self, image_path, patch_size):
        # Open the image file
        with Image.open(image_path) as img:
            img_width, img_height = img.size
            height_offset = 0
            if img_height % patch_size != 0:
                num_patches_h = int(np.ceil(img_height / patch_size))
                height_offset = (patch_size - (patch_size - img_height // num_patches_h)) // 2 
                
            width_offset = 0
            if img_width % patch_size != 0:
                num_patches_w = int(np.ceil(img_width / patch_size))
                width_offset = (patch_size - (patch_size - img_width // num_patches_w)) // 2 
            
            patches = []
            patch_center = []
            pacth_wh = []
            
            # Crop the image into patches
            for i in range(0, img_height, patch_size):
                i = i - height_offset * (i // patch_size)
                for j in range(0, img_width, patch_size):
                    j = j - width_offset * (j //patch_size)
                    box = (j, i, j + patch_size, i + patch_size)
                    patch = img.crop(box).resize(self.patch_size_return, resample=Resampling.BILINEAR)
                    patches.append(pil_to_tensor(patch))
                    patch_center.append([j + patch_size // 2, i + patch_size // 2])
                    pacth_wh.append([patch_size, patch_size])
        
        patches_cropped = torch.stack(patches).float()/255.
        patch_center_2d = torch.tensor(patch_center, dtype=torch.float32)
        patch_sizes = torch.tensor(pacth_wh, dtype=torch.float32)
        return patches_cropped, patch_center_2d, patch_sizes

    def default_value(self, key): #hardcode, ugly
        if "patch_size" in key:
            return torch.tensor([[0, 0]])
        if "patch_center_2d" in key or "velocity" in key:
            return [0, 0]
        elif "center" in key:
            return (0, 0)
        
        if "camera_params" in key:
            return {'focal_length': torch.tensor([0.]), 'principal_point': torch.tensor([[0., 0.]]), 'znear': 0.0, 'device': 'cpu', 'zfar': 0.0, 'image_size': [(0, 0)]}
        
        if "isvalid" in key:
            return False
        if "label" in key:
            return 0
        if "bbox_3d" in key:
            return [0, 0, 0, 0, 0, 0, 0]
        if "bbox" in key:
            return [0, 0, 0, 0]

        if "zoom" in key:
            return torch.tensor(0)
        else:
            return 0

@DATASETS.register_module()
class NuScenesTrain(NuScenesBase):
    """
    Dataset class for training data in the NuScenes dataset.
    Inherits from the NuScenesBase class.
    """
    def __init__(self, **kwargs):
        self.split = "train"
        self.ann_file = "nuscenes_infos_train.pkl"
        kwargs.update({"ann_file": self.ann_file})
        super().__init__(**kwargs)
        
@DATASETS.register_module()     
class NuScenesValidation(NuScenesBase):
    """
    Dataset class for validation data in the NuScenes dataset.
    Inherits from the NuScenesBase class.
    """
    def __init__(self, **kwargs):
        self.split = "validation"
        self.ann_file = "nuscenes_infos_val.pkl"
        kwargs.update({"ann_file": self.ann_file})
        super().__init__(**kwargs)

@DATASETS.register_module()   
class NuScenesTest(NuScenesBase):
    """
    Dataset class for test data in the NuScenes dataset.
    Inherits from the NuScenesBase class.
    """
    def __init__(self, **kwargs):
        self.split = "test"
        ann_file = "nuscenes_infos_test.pkl"
        kwargs.update({"ann_file": ann_file})
        super().__init__(**kwargs)

@DATASETS.register_module()
class NuScenesTrainMini(NuScenesBase):
    """
    Dataset class for mini training data in the NuScenes dataset.
    Inherits from the NuScenesBase class.
    """
    def __init__(self, data_root=None, **kwargs):
        self.split = "train-mini"
        ann_file = "nuscenes_mini_infos_train.pkl"
        kwargs.update({"data_root": data_root, "ann_file": ann_file})
        super().__init__(**kwargs)
        
@DATASETS.register_module()
class NuScenesValidationMini(NuScenesBase):
    """
    Dataset class for mini validation data in the NuScenes dataset.
    Inherits from the NuScenesBase class.
    """
    def __init__(self, data_root=None, **kwargs):
        self.split = "val-mini"
        ann_file = "nuscenes_mini_infos_val.pkl"
        kwargs.update({"data_root": data_root, "ann_file": ann_file})
        super().__init__(**kwargs)
