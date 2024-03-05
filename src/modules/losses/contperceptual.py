# src/modules/losses/contperceptual.py
import torch.nn as nn
from ldm.modules.losses.contperceptual import LPIPSWithDiscriminator as LPIPSWithDiscriminator_LDM
# from se3.homtrans3d import T2xyzrpy
import math

SE3_DIM = 16

class LPIPSWithDiscriminator(LPIPSWithDiscriminator_LDM):
    """LPIPS loss with discriminator."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
class PoseLoss(LPIPSWithDiscriminator_LDM):
    """LPIPS loss with discriminator."""
    def __init__(self, pose_weight=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pose_weight = pose_weight
        self.pose_loss = nn.MSELoss()

    def compute_pose_loss(self, pred, gt):
        return self.pose_loss(pred, gt)
            
    def forward(self, inputs, reconstructions, pose_inputs, pose_reconstructions,
                posteriors, optimizer_idx, global_step, 
                last_layer=None, cond=None, split="train",
                weights=None):
        
        loss, log \
            = super().forward(inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer, cond, split,
                weights)
        assert pose_inputs.shape == pose_reconstructions.shape
        assert pose_inputs.shape[1] == int(math.sqrt(SE3_DIM)), pose_inputs.shape[2] == int(math.sqrt(SE3_DIM))

        pose_loss = self.compute_pose_loss(pose_inputs, pose_reconstructions)
        weighted_pose_loss = self.pose_weight * pose_loss
        print("total loss without pose loss: ", loss)
        print("pose loss: ", pose_loss)
        print("weighted pose loss: ", weighted_pose_loss)
        loss += weighted_pose_loss 
        
        log["{}/pose_loss".format(split)] = pose_loss.clone().detach().mean()
        log["{}/weighted_pose_loss".format(split)] = weighted_pose_loss.clone().detach().mean()
        log["{}/total_loss".format(split)] = loss.clone().detach().mean()
        
        return loss, log
            