# This file code the dynamic loss functions

import torch
import numpy as np

import pypose as pp
from torch import nn


from ugraph_helper import Precompute_helper
import roma
import torch.nn.functional as F

def get_dynamic_loss(precompute: Precompute_helper, ts: torch.Tensor):
    # ts : [batch of time frames]

    transls=precompute.params['transls'] # all
    oris_wxyz=precompute.params['oris_wxyz'] # all

    oris_wxyz = F.normalize(oris_wxyz, p=2, dim=-1)
    # oris_xyzw=roma.quat_xyzw_to_wxyz(oris_wxyz) # [BUG]
    oris_xyzw=roma.quat_wxyz_to_xyzw(oris_wxyz)

    assert torch.norm(oris_xyzw[3,5],p=2)-1<1e-6

    # velocity loss
    # loss_velocity = torch.mean(torch.abs(transls[:-1] - transls[1:]))
    # loss_ori_velocity = torch.mean(torch.abs(oris_xyzw[:-1] - oris_xyzw[1:]))
    loss_velocity = torch.mean(torch.square(transls[:-1] - transls[1:]))
    loss_ori_velocity = torch.mean(torch.square(oris_xyzw[:-1] - oris_xyzw[1:]))

    # accleration loss
    loss_accel = compute_accel_loss(transls)
    loss_ori_accel = compute_accel_loss(oris_xyzw)
    

    
    dynamic_loss = 100.0*loss_velocity + 100.0*loss_ori_velocity + 1.0*loss_accel + 1.0*loss_ori_accel
    stats_dy={
        "dynamic_loss/loss_velocity": loss_velocity.item(),
        "dynamic_loss/loss_ori_velocity": loss_ori_velocity.item(),
        "dynamic_loss/loss_accel": loss_accel.item(),
        "dynamic_loss/loss_ori_accel": loss_ori_accel.item(),
    }

    # stats_graph = {**stats_key, **stats_nonkey}
    
    return dynamic_loss, stats_dy

def get_dynamic_loss_by_input(transls,oris_quat, ts: torch.Tensor):
    # oris_quat: wxyz or xyzw
    # ts : [batch of time frames]

    oris_quat = F.normalize(oris_quat, p=2, dim=-1)
    # oris_xyzw=roma.quat_xyzw_to_wxyz(oris_wxyz) [BUG]
    # oris_xyzw=roma.quat_wxyz_to_xyzw(oris_wxyz)

    assert torch.norm(oris_quat[3,5],p=2)-1<1e-6

    # velocity loss
    loss_velocity = torch.mean(torch.abs(transls[:-1] - transls[1:]))
    loss_ori_velocity = torch.mean(torch.abs(oris_quat[:-1] - oris_quat[1:]))

    # accleration loss
    loss_accel = compute_accel_loss(transls)
    loss_ori_accel = compute_accel_loss(oris_quat)
    

    
    dynamic_loss = 0.1*loss_velocity + 0.1*loss_ori_velocity + 0.1*loss_accel + 0.1*loss_ori_accel
    stats_dy={
        "dynamic_loss/loss_velocity": loss_velocity.item(),
        "dynamic_loss/loss_ori_velocity": loss_ori_velocity.item(),
        "dynamic_loss/loss_accel": loss_accel.item(),
        "dynamic_loss/loss_ori_accel": loss_ori_accel.item(),
    }

    # stats_graph = {**stats_key, **stats_nonkey}
    
    return dynamic_loss, stats_dy


def compute_accel_loss(transls):
    # transls [nt,n,3] or oris [nt,n,4]
    accel = 2 * transls[1:-1] - transls[:-2] - transls[2:] # [nt-2,n,3/4]
    loss = accel.norm(dim=-1).mean() # [nt-2,n,3/4] => [nt-2,n]=> [1]
    return loss