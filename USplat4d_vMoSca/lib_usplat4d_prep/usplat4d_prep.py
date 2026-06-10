import sys
import os
sys.path.append(os.path.dirname("../lib_render"))
sys.path.append(os.path.dirname("../lib_mosca/gs_utils"))
sys.path.append(os.path.dirname(".."))

import torch
import numpy as np
from scipy.spatial.transform import Rotation
import pickle
import random
import glob

from torch import nn

from itertools import chain
from scipy.spatial import KDTree,Delaunay,distance_matrix
import open3d as o3d
from collections import defaultdict
import roma

from mpl_toolkits.mplot3d import Axes3D

# import hdbscan
from sklearn.cluster import DBSCAN
from tqdm import tqdm

from lib_mosca.mosca import __DQB_warp2__, __LBS_warp2__

from lib_mosca.gs_utils.utils_helper import project_points, draw_projected_points, draw_projected_points_masked, valid_and_visible, overlay_mask_yellow, stack_images_side_by_side, compute_depth_gradient, associate_pts_value, draw_projected_points_colored
from lib_mosca.gs_utils.som_vis_utils import (
    apply_depth_colormap,
)

import matplotlib.pyplot as plt
import argparse
import logging, glob, sys, os, shutil, os.path as osp

def get_order(arr):
    """
    Computes the rank (order) of each element in the array.

    Args:
        arr (np.ndarray): Input 1D array.

    Returns:
        np.ndarray: Array where the i-th element is the rank of the i-th element in the input array.
    """
    # Get the indices that would sort the array
    sorted_indices = np.argsort(arr)
    
    # Create an array to store the ranks
    order = np.empty_like(sorted_indices)
    
    # Assign ranks
    order[sorted_indices] = np.arange(len(arr))
    
    return order

def count_out_dict_files(folder_path):
    # Use glob to find files matching the pattern
    file_pattern = os.path.join(folder_path, 'out_dict_*.pkl')
    files = glob.glob(file_pattern)
    return len(files)

def calculate_edge_relative_orientation(Ts): # 854 pts will use ?? GB memory!!!
    nk=Ts.shape[1]
    Ts1=Ts.unsqueeze(2).repeat(1,1,nk,1)
    Ts2=Ts.unsqueeze(1).repeat(1,nk,1,1)
    Ts12=(Ts1.Inv()*Ts2)
    # Ts_rel=relativelieAlgebra_pp(Ts1,Ts2)
    return Ts12

def get_navie_ori_distances(Oris_rel): # TODO
    Oris_rel_so3=Oris_rel.Log() # (nt, nk, nk, 3)
    norm_oris_rel_so3=torch.norm(Oris_rel_so3.tensor(),dim=-1) # (nt, nk, nk)
    norm_ori_rel_so3=torch.mean(norm_oris_rel_so3,axis=0) # (nk, nk)
    return 1.0 - torch.cos(norm_ori_rel_so3) # (nk, nk)

def compute_ori_distances_chunked(Oris_xyzw, chunk_size=50):
    """Chunked replacement for calculate_edge_relative_orientation + get_navie_ori_distances.
    Avoids (nt, nk, nk, 4) tensor: for nt=427,nk=1500 that is 15 GB x3 = 46 GB OOM.
    Processes chunk_size rows of nk at a time; peak ~2 GB per chunk at chunk_size=50.
    Returns (nk, nk) float32 orientation distance matrix.
    """
    import pypose as pp
    nt = Oris_xyzw.shape[0]
    nk = Oris_xyzw.shape[1]
    result = torch.zeros(nk, nk, dtype=torch.float32)
    raw = Oris_xyzw.tensor()  # (nt, nk, 4)
    for i_start in range(0, nk, chunk_size):
        i_end = min(i_start + chunk_size, nk)
        c = i_end - i_start
        chunk_i = raw[:, i_start:i_end, :].unsqueeze(2).expand(-1, -1, nk, -1).contiguous()  # (nt, c, nk, 4)
        all_j = raw.unsqueeze(1).expand(-1, c, -1, -1).contiguous()  # (nt, c, nk, 4)
        Ts1_pp = pp.LieTensor(chunk_i, ltype=pp.SO3_type)
        Ts2_pp = pp.LieTensor(all_j, ltype=pp.SO3_type)
        Ts12 = Ts1_pp.Inv() * Ts2_pp  # (nt, c, nk) SO3
        log_chunk = Ts12.Log()  # (nt, c, nk, 3)
        norm_chunk = torch.norm(log_chunk.tensor(), dim=-1)  # (nt, c, nk)
        mean_chunk = torch.mean(norm_chunk, dim=0)  # (c, nk)
        result[i_start:i_end] = 1.0 - torch.cos(mean_chunk)
        del chunk_i, all_j, Ts1_pp, Ts2_pp, Ts12, log_chunk, norm_chunk, mean_chunk
    return result


def compute_accel_loss(transls):
    # transls [nt,n,3] or oris [nt,n,4]
    accel = 2 * transls[1:-1] - transls[:-2] - transls[2:] # [nt-2,n,3/4]
    loss = accel.norm(dim=-1).mean() # [nt-2,n,3/4] => [nt-2,n]=> [1]
    return loss
def cal_rigidity_loss(transls,oris_xyzw,edges_local,weight_edges,dis_local_canonical_edge):
    # transls.shape # [nt, num_nodes, 3]
    # oris_xyzw.shape # [nt, num_nodes, 4]
    # edges_local.shape # [num_edges, 2] # [...[center id, neighbor id]...]
    # weight_edges.shape # [num_edges]
    # dis_local_canonical_edge.shape # [num_edges]
    

    # 1. compute local rigidity loss for transls
    transls_diff_edges=transls[:,edges_local[:,1]]-transls[:,edges_local[:,0]] # [nt, num_edges, 3]

    # t, t-1
    oris_xyzw_tdiff=roma.quat_product(oris_xyzw[:-1],roma.quat_conjugation(oris_xyzw[1:])) # [nt-1, num_nodes, 4] # R{i,t-1} * R{i,t}^-1 # Conjugation of a unit quaternion is equal to its inverse. # quat_inverse(quat)
    oris_xyzw_tdiff_edges=oris_xyzw_tdiff[:,edges_local[:,0]] # [nt-1, num_edges, 4]
    transls_diff_cal_edges=roma.utils.quat_action(oris_xyzw_tdiff_edges,transls_diff_edges[1:]) # [nt-1, num_edges, 3] # = R*t
    transls_diff_current=transls_diff_edges[:-1] # [nt-1, num_edges, 3]
    dis_diff=torch.norm(transls_diff_cal_edges-transls_diff_current,dim=-1) # [nt-1, num_edges]
    error_local_rigidity=dis_diff*weight_edges[None,:] # [nt-1, num_edges]
    loss_local_rigidity=error_local_rigidity.mean() # scalar

    # 2. compute local rigidity loss for oris
    qq_j_edges=oris_xyzw_tdiff[:,edges_local[:,1]] # [nt-1, num_edges, 4]
    qq_i_edges=oris_xyzw_tdiff[:,edges_local[:,0]]
    qq_diff=torch.norm(qq_j_edges-qq_i_edges,dim=-1) # [nt-1, num_edges] ### TODO: can change to SO3 version
    error_local_rigidity_ori=qq_diff*weight_edges[None,:] # [nt-1, num_edges]
    loss_local_rigidity_ori=error_local_rigidity_ori.mean() # scalar

    # 3. compute global rigidity loss
    dis_local_edge=torch.norm(transls_diff_edges,dim=-1) # [nt, num_edges]
    dis_local_diff=torch.abs(dis_local_canonical_edge[None,:]-dis_local_edge) # [nt, num_edge]
    error_local_global=dis_local_diff*weight_edges[None,:] # [nt, num_edge]
    loss_local_global=error_local_global.mean() # scalar

    loss = loss_local_rigidity+loss_local_rigidity_ori+100*loss_local_global
    stats = {
        "loss_local_rigidity": loss_local_rigidity,
        "loss_local_rigidity_ori": loss_local_rigidity_ori,
        "loss_local_global": loss_local_global,
    }
    
    return loss, stats

def cal_dt_loss(delta_t,oris_xyzw,edges_local,weight_edges,transls_diff_edges):
    oris_xyzw_tdiff_dt=roma.quat_product(oris_xyzw[:-delta_t],roma.quat_conjugation(oris_xyzw[delta_t:])) # [nt-1, num_nodes, 4] # R{i,t-1} * R{i,t}^-1 # Conjugation of a unit quaternion is equal to its inverse. # quat_inverse(quat)
    oris_xyzw_tdiff_edges_dt=oris_xyzw_tdiff_dt[:,edges_local[:,0]] # [nt-1, num_edges, 4]
    transls_diff_cal_edges_dt=roma.utils.quat_action(oris_xyzw_tdiff_edges_dt,transls_diff_edges[delta_t:]) # [nt-1, num_edges, 3] # = R*t
    transls_diff_current_dt=transls_diff_edges[:-delta_t] # [nt-1, num_edges, 3]
    dis_diff_dt=torch.norm(transls_diff_cal_edges_dt-transls_diff_current_dt,dim=-1) # [nt-1, num_edges]
    error_local_rigidity_dt=dis_diff_dt*weight_edges[None,:] # [nt-1, num_edges]
    loss_local_rigidity_dt=error_local_rigidity_dt.mean() # scalar

    # 2. compute local rigidity loss for oris
    qq_j_edges_dt=oris_xyzw_tdiff_dt[:,edges_local[:,1]] # [nt-1, num_edges, 4]
    qq_i_edges_dt=oris_xyzw_tdiff_dt[:,edges_local[:,0]]
    qq_diff_dt=torch.norm(qq_j_edges_dt-qq_i_edges_dt,dim=-1) # [nt-1, num_edges] ### TODO: can change to SO3 version
    error_local_rigidity_ori_dt=qq_diff_dt*weight_edges[None,:] # [nt-1, num_edges]
    loss_local_rigidity_ori_dt=error_local_rigidity_ori_dt.mean() # scalar

    return loss_local_rigidity_dt,loss_local_rigidity_ori_dt

def cal_rigidity_loss_extra(transls,oris_xyzw,edges_local,weight_edges,dis_local_canonical_edge):
    # transls.shape # [nt, num_nodes, 3]
    # oris_xyzw.shape # [nt, num_nodes, 4]
    # edges_local.shape # [num_edges, 2] # [...[center id, neighbor id]...]
    # weight_edges.shape # [num_edges]
    # dis_local_canonical_edge.shape # [num_edges]
    
    nt=transls.shape[0]

    # 1. compute local rigidity loss for transls
    transls_diff_edges=transls[:,edges_local[:,1]]-transls[:,edges_local[:,0]] # [nt, num_edges, 3]

    # t, t-1
    oris_xyzw_tdiff=roma.quat_product(oris_xyzw[:-1],roma.quat_conjugation(oris_xyzw[1:])) # [nt-1, num_nodes, 4] # R{i,t-1} * R{i,t}^-1 # Conjugation of a unit quaternion is equal to its inverse. # quat_inverse(quat)
    oris_xyzw_tdiff_edges=oris_xyzw_tdiff[:,edges_local[:,0]] # [nt-1, num_edges, 4]
    transls_diff_cal_edges=roma.utils.quat_action(oris_xyzw_tdiff_edges,transls_diff_edges[1:]) # [nt-1, num_edges, 3] # = R*t
    transls_diff_current=transls_diff_edges[:-1] # [nt-1, num_edges, 3]
    dis_diff=torch.norm(transls_diff_cal_edges-transls_diff_current,dim=-1) # [nt-1, num_edges]
    error_local_rigidity=dis_diff*weight_edges[None,:] # [nt-1, num_edges]
    loss_local_rigidity=error_local_rigidity.mean() # scalar

    # 2. compute local rigidity loss for oris
    qq_j_edges=oris_xyzw_tdiff[:,edges_local[:,1]] # [nt-1, num_edges, 4]
    qq_i_edges=oris_xyzw_tdiff[:,edges_local[:,0]]
    qq_diff=torch.norm(qq_j_edges-qq_i_edges,dim=-1) # [nt-1, num_edges] ### TODO: can change to SO3 version
    error_local_rigidity_ori=qq_diff*weight_edges[None,:] # [nt-1, num_edges]
    loss_local_rigidity_ori=error_local_rigidity_ori.mean() # scalar

    # 3. compute global rigidity loss
    dis_local_edge=torch.norm(transls_diff_edges,dim=-1) # [nt, num_edges]
    dis_local_diff=torch.abs(dis_local_canonical_edge[None,:]-dis_local_edge) # [nt, num_edge]
    error_local_global=dis_local_diff*weight_edges[None,:] # [nt, num_edge]
    loss_local_global=error_local_global.mean() # scalar

    if nt<100:
        delta_t=int(nt/2)
    else:
        delta_t=50
    loss_local_rigidity_50,loss_local_rigidity_ori_50=cal_dt_loss(delta_t,oris_xyzw,edges_local,weight_edges,transls_diff_edges)

    loss_local_rigidity_10,loss_local_rigidity_ori_10=cal_dt_loss(10,oris_xyzw,edges_local,weight_edges,transls_diff_edges)

    loss = loss_local_rigidity+loss_local_rigidity_ori+100*loss_local_global+100*loss_local_rigidity_50+100*loss_local_rigidity_10#+100*loss_local_rigidity_ori_dt
    stats = {
        "loss_local_rigidity": loss_local_rigidity.item(),
        "loss_local_rigidity_ori": loss_local_rigidity_ori.item(),
        "loss_local_global": loss_local_global.item(),
        "loss_local_rigidity_50": loss_local_rigidity_50.item(),
        "loss_local_rigidity_10": loss_local_rigidity_10.item(),
    }
    
    return loss, stats

def generate_ratio_binary_tensor(n, ratio_0=0.66):
    # assert abs(ratio_0 + ratio_1 - 1.0) < 1e-6, "Ratios must sum to 1"
    ratio_1=1-ratio_0
    
    num_0 = int(round(n * ratio_0))
    num_1 = n - num_0  # to ensure total length is exactly n
    
    tensor = torch.cat([
        torch.zeros(num_0, dtype=torch.bool),
        torch.ones(num_1, dtype=torch.bool)
    ])
    
    shuffled_tensor = tensor[torch.randperm(n)]
    return shuffled_tensor

def generate_ratio_binary_tensor(n, ratio_0=0.66):
    # assert abs(ratio_0 + ratio_1 - 1.0) < 1e-6, "Ratios must sum to 1"
    ratio_1=1-ratio_0
    
    num_0 = int(round(n * ratio_0))
    num_1 = n - num_0  # to ensure total length is exactly n
    
    tensor = torch.cat([
        torch.zeros(num_0, dtype=torch.bool),
        torch.ones(num_1, dtype=torch.bool)
    ])
    
    shuffled_tensor = tensor[torch.randperm(n)]
    return shuffled_tensor

# optimize_overall_with_orientation version of 
def optimize_overall_with_orientation_edges(
    edges,
    Transls, Transls_optimizable, Mask, Dists_optimizable, weight_distance_optimizable,
    Oris_xyzw, Oris_xyzw_optimizable, Ori_distances_optimizable, weight_ori_distance_optimizable, 
    Dists_observed_mean, Dists_observed_std, Mask_observed,Nearby_mask,
    W2Cs,Depths_info_perG_key,
    lr=1e-1, max_epochs=10000, min_lr=1e-6, log_interval=100
):

    nt = Transls.shape[0]
    nk = Transls.shape[1]
    ne = edges.shape[0]

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Transls_optimizable = nn.Parameter(Transls_optimizable.to(device))
    Dists_optimizable = nn.Parameter(Dists_optimizable.to(device))
    weight_distance_optimizable = nn.Parameter(weight_distance_optimizable.to(device))
    Oris_xyzw_optimizable = nn.Parameter(Oris_xyzw_optimizable.tensor().to(device))
    Ori_distances_optimizable = nn.Parameter(Ori_distances_optimizable.to(device))
    Transls = Transls.to(device)
    Oris_xyzw = Oris_xyzw.to(device)
    Mask = Mask.to(device)
    Depths_info_perG_key=Depths_info_perG_key.to(device)

    optimizer = torch.optim.Adam(
        [Transls_optimizable, Dists_optimizable, weight_distance_optimizable, Oris_xyzw_optimizable, Ori_distances_optimizable], lr=lr
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, threshold=1e-3, min_lr=min_lr
    )

    # Distance matrix helper function
    def distance_matrix(x, y):
        return torch.cdist(x, y, p=2)

    W2Cs_R=torch.tensor(W2Cs[:,:3,:3],dtype=torch.float32,device=device) # [nt,3,3]
    I0=torch.eye(3,device=device).expand(nt,nk,-1,-1).clone() # [nt,nk,3,3]
    I0[...,-1,-1]=Depths_info_perG_key # 0.01 camel # CHECK TODO
    I1=W2Cs_R.transpose(-1, -2)[:,None,:,:]@I0@W2Cs_R[:,None,:,:] # [nt,nk,3,3]

    # Optimization loop
    for epoch in tqdm(range(max_epochs), desc="Training Epochs", unit="epoch"):
        optimizer.zero_grad()

        Oris_xyzw_optimizable_normed=Oris_xyzw_optimizable/Oris_xyzw_optimizable.norm(dim=-1,keepdim=True)
        
        a1=(Transls_optimizable-Transls)[:,:,None,:]  # (nt, nk, 1, 3)
        a1_T=a1.transpose(-1, -2)
        squared_norm=(a1@I1@a1_T).squeeze(-1,-2)
        # squared_norm=(a1@a1_T).squeeze(-1,-2) # test
        Dist_confident_pts_error = torch.sqrt(squared_norm + 1e-8) # add epsilon for numerical stability

        # Loss: Confident points distance error
        loss_dist_confident_pts = torch.mean(Dist_confident_pts_error[Mask]) # full model

        mask_edge=generate_ratio_binary_tensor(n=edges.shape[0], ratio_0=0.3)
        edges_mask=edges[mask_edge]
        weight_normed=torch.abs(weight_distance_optimizable)/torch.norm(weight_distance_optimizable,dim=-1,p=1)[:,None] # (nu,nk)
        weight_edges=weight_normed[edges_mask[:,0],edges_mask[:,1]] # (ne)
        dis_local_canonical_edge=Dists_optimizable[edges_mask[:,0],edges_mask[:,1]] # (ne) CHECK order
        loss_rigidity,stats_rigidity=cal_rigidity_loss_extra(Transls_optimizable,Oris_xyzw_optimizable_normed,edges_mask,weight_edges,dis_local_canonical_edge)

        # Loss: Regularization on weight distance
        loss_sparsity=torch.mean(torch.norm(weight_normed,p=1,dim=-1)/torch.norm(weight_normed, p=2,dim=-1) )-1 # float('inf')

        loss_accel = compute_accel_loss(Transls_optimizable)

        a_loss_dist_confident_pts=1*loss_dist_confident_pts
        a_loss_rigidity=1*loss_rigidity
        a_loss_sparsity=1e-1/loss_sparsity
        a_loss_accel=0.1*loss_accel
        
        total_loss = a_loss_dist_confident_pts + a_loss_rigidity + a_loss_accel # 

        # Backpropagation and optimization step
        total_loss.backward()
        optimizer.step()

        # Adjust learning rate with the scheduler
        scheduler.step(total_loss)
        
        # Logging progress
        rigidity1=stats_rigidity["loss_local_rigidity"]
        rigidity_ori2=stats_rigidity["loss_local_rigidity_ori"]
        global3=stats_rigidity["loss_local_global"]
        if epoch % log_interval == 0 or epoch == max_epochs - 1 or optimizer.param_groups[0]['lr'] < min_lr * 2:
            print(f"Epoch {epoch}, L: {total_loss.item():.6f}, "
                  f"ConfDist: {a_loss_dist_confident_pts.item():.6f}, "
                  f"a_ridigity: {a_loss_rigidity.item():.6f}, "
                  f"a_loss_accel: {a_loss_accel.item():.6f}, "
                  f"a_loss_sparsity: {a_loss_sparsity.item():.6f}, "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")
            print(f"weight_normed: {(weight_normed>0.1).sum().item():.6f}")

        # Early stopping
        if optimizer.param_groups[0]['lr'] < min_lr * 2: #and epoch > 5000: #and a_loss_dist < 1e-3 and a_loss_oris_distance_confident_pts < 1e-1:
            print(f"Stopping early at epoch {epoch}.")
            break
    
    Oris_xyzw_optimizable_normed=Oris_xyzw_optimizable/Oris_xyzw_optimizable.norm(dim=-1,keepdim=True)

    return Transls_optimizable, Dists_optimizable, weight_distance_optimizable, Oris_xyzw_optimizable_normed, Ori_distances_optimizable


def get_knn_edges(n_neighbors,Transls,Contribs,Transls2,Contribs2,contrib_threshold, flag_remove_self=False):
    # choice 2: knn graph
    def o3d_knn_p(pts2,pts, num_knn, flag_remove_self=False):
        indices = []
        sq_dists = []
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, np.float64))
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        for p in pts2:
            if not flag_remove_self:
                [_, i, d] = pcd_tree.search_knn_vector_3d(p, num_knn)
                indices.append(i[0:])
                sq_dists.append(d[0:])
            else:
                [_, i, d] = pcd_tree.search_knn_vector_3d(p, num_knn + 1)
                indices.append(i[1:])
                sq_dists.append(d[1:])
        return np.array(sq_dists), np.array(indices)

    nk=Transls.shape[1]
    nu=Transls2.shape[1]

    i_s_knn_others,i_s_knn_is=[],[]
    # n_neighbors=5
    for i in range(nu):
        t_most_confident=np.argmax(Contribs2[:,i]) # np.argsort(Contribs2[:,i])[-1]
        eff_map=Contribs[t_most_confident]>contrib_threshold # key points map at t_most_confident
        pts=Transls[t_most_confident][eff_map] # confident key points at t_most_confident
        i_s=torch.arange(nk)[eff_map] # confident key points indices at t_most_confident
        pt2=Transls2[t_most_confident,i] # i-th other points at t_most_confident
        pts2=[pt2] # just one point
        sq_dists, indices = o3d_knn_p(pts2, pts, n_neighbors, flag_remove_self)
        sq_dists,indices= sq_dists[0],indices[0]
        i_s_knn=i_s[indices] # usage: ids[i_s_knn]
        i_s_knn_others.append(i_s_knn)
        i_s_knn_is.append(i*torch.ones_like(i_s_knn))
    i_s_knn_is=torch.cat(i_s_knn_is)
    i_s_knn_others=torch.cat(i_s_knn_others)
    edges_others=torch.stack([i_s_knn_is,i_s_knn_others],dim=-1)
    return edges_others


    
def get_knn_edges_naive(n_neighbors, Transls, Transls2, flag_remove_self=False):
    """
    Naive temporal k-NN graph construction using average distance over time.

    Args:
        n_neighbors (int): Number of neighbors.
        Transls (torch.Tensor): (T, N1, D) tensor of reference/key points.
        Transls2 (torch.Tensor): (T, N2, D) tensor of query points.
        flag_remove_self (bool): If True and N1 == N2, removes self-matches.

    Returns:
        edges_others (torch.Tensor): (N2 * k, 2) edge index pairs [Transls2_index, Transls_index].
        dists (torch.Tensor): (N2 * k,) mean distances corresponding to each edge.
    """
    T, N1, D = Transls.shape
    N2 = Transls2.shape[1]

    # Compute temporal average pairwise distances
    dist_list = []
    for t in range(T):
        dist = (Transls2[t][:, None, :] - Transls[t][None, :, :]).norm(dim=-1)  # (N2, N1)
        dist_list.append(dist)
    Dist = torch.stack(dist_list, dim=0).mean(dim=0)  # (N2, N1)

    if flag_remove_self and N1 == N2:
        diag = torch.arange(N2, device=Dist.device)
        Dist[diag, diag] = float('inf')

    # Get top-k nearest neighbors and distances
    dists, knn_indices = torch.topk(Dist, k=n_neighbors, dim=1, largest=False)  # (N2, k)

    # Build edge index pairs
    row_indices = torch.arange(N2, device=Dist.device).unsqueeze(1).repeat(1, n_neighbors)  # (N2, k)
    edges_others = torch.stack([row_indices, knn_indices], dim=-1).reshape(-1, 2)  # (N2 * k, 2)
    dists_edges = dists.reshape(-1)  # (N2 * k,)

    return edges_others, dists_edges

# for(4) nonkey graph optimization
def optimize_others_with_orientation_edges4(
    edges,
    Transls, Transls_optimizable, Mask, #Dists_optimizable, #weight_distance_optimizable,
    Oris_xyzw, Oris_xyzw_optimizable, #Ori_distances_optimizable, #weight_ori_distance_optimizable, 
    Transls_key, Oris_xyzw_key,Transls_key_optimized, Oris_xyzw_key_optimized,
    weight_param_key_optimizable,sq_dists_key_nonkey_edges,
    ts_most_confident,per_nonkeyGaussian_dweight,
    W2Cs,Depths_info_perG_nonkey,
    device,
    lr=1e-1, max_epochs=10000, min_lr=1e-6, log_interval=100
):  
    nt_all = Transls.shape[0]
    nu=Transls.shape[1]
    nk=Transls_key_optimized.shape[1]
    ne=edges.shape[0]

    Oris_xyzw=Oris_xyzw/Oris_xyzw.norm(dim=-1,keepdim=True) # normalize
    Oris_xyzw_key_optimized=Oris_xyzw_key_optimized/Oris_xyzw_key_optimized.norm(dim=-1,keepdim=True) # normalize

    # Device setup
    Transls_optimizable=Transls_optimizable.to(device)
    Oris_xyzw_optimizable=Oris_xyzw_optimizable.tensor().to(device)

    Transls = Transls.to(device)
    Oris_xyzw = Oris_xyzw.to(device)
    Transls_key = Transls_key.to(device)
    Oris_xyzw_key = Oris_xyzw_key.to(device)
    Transls_key_optimized = Transls_key_optimized.to(device).detach()
    Oris_xyzw_key_optimized = Oris_xyzw_key_optimized.to(device).detach()
    weight_param_key_optimizable = nn.Parameter(weight_param_key_optimizable.to(device))
    sq_dists_key_nonkey_edges = sq_dists_key_nonkey_edges.to(device)
    per_nonkeyGaussian_dweight = nn.Parameter(per_nonkeyGaussian_dweight.to(device)) # don't use for now, just keep
    Depths_info_perG_nonkey = Depths_info_perG_nonkey.to(device)

    exp_sq_dists_key_nonkey_edges=torch.exp(-sq_dists_key_nonkey_edges) # (ne)
    W2Cs_R=torch.tensor(W2Cs[:,:3,:3],dtype=torch.float32,device=device) # [nt_all,3,3]

    Mask = Mask.to(device)

    @torch.no_grad() # check
    def cal_weight_edges2(weight_param_key_optimizable,exp_sq_dists_key_nonkey_edges,edges,nu):
        # weight_param_key_optimizable [nk]
        # exp_sq_dists_key_nonkey_edges [ne]
        device=weight_param_key_optimizable.device
        nk=weight_param_key_optimizable.shape[0]

        weight_param_key_optimizable_k=100*torch.abs(weight_param_key_optimizable) # [TODO]: 2000 # 100
        weight_param_edges=weight_param_key_optimizable_k[edges[:,1]] # (ne)
        weight_edges = exp_sq_dists_key_nonkey_edges**weight_param_edges # TODO: to unify scale #test: 20 for block

        weight_edges = torch.clamp(weight_edges, min=1e-10, max=1e10) # test

        weights_matrix=torch.zeros((nu,nk),device=device)
        weights_matrix[edges[:,0],edges[:,1]]=weight_edges
        sum_temp = torch.sum(weights_matrix,dim=-1,keepdim=True)
        if any(sum_temp==0):
            breakpoint()
        if torch.isnan(weight_param_key_optimizable).any():
            breakpoint()
        weights_matrix_normed=weights_matrix/sum_temp

        weight_edges_normed=weights_matrix_normed[edges[:,0],edges[:,1]] # (ne)

        return weight_edges_normed
    
    

    # prepare
    Oris_xyzw_key_norm=Oris_xyzw_key/Oris_xyzw_key.norm(dim=-1,keepdim=True)
    Oris_xyzw_key_optimized_norm=Oris_xyzw_key_optimized/Oris_xyzw_key_optimized.norm(dim=-1,keepdim=True)
    n_neighbors=(edges[:,0]==0).sum()
    if n_neighbors!=5:
        breakpoint()

    # new
    src_xyz_1=Transls[ts_most_confident,edges[::n_neighbors,0]] # (nu,3)
    src_xyzw_1=Oris_xyzw[ts_most_confident,edges[::n_neighbors,0]] # (nu,4)
    src_xyz_1 = nn.Parameter(src_xyz_1)
    src_xyzw_1 = nn.Parameter(src_xyzw_1)

    # get sk_src_node and sk_dst_node (key nodes)
    ts_most_confident_repeat=ts_most_confident.repeat(n_neighbors,1).T.reshape(-1) # (nt*5)
    sk_src_node_xyz_1=Transls_key[ts_most_confident_repeat,edges[:,1]] # (ne,3)
    sk_src_node_xyzw_1=Oris_xyzw_key_norm[ts_most_confident_repeat,edges[:,1]] # (ne,4)
    sk_src_node_wxyz_1=sk_src_node_xyzw_1[:,[3,0,1,2]]
    sk_src_node_xyz_2=sk_src_node_xyz_1.reshape(-1,n_neighbors,3) # (nu,n_neighbors,3)
    sk_src_node_quat_2=sk_src_node_wxyz_1.reshape(-1,n_neighbors,4) # (nu,n_neighbors,4)

    optimizer = torch.optim.Adam(
        [weight_param_key_optimizable,src_xyz_1,src_xyzw_1], lr=lr
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=min_lr
    )
    ######################################################3
    
    ts_all = torch.arange(nt_all, device=device)
    batch_size = 30
    steps_per_epoch = (nt_all + batch_size - 1) // batch_size
    total_steps = steps_per_epoch * max_epochs

    pbar = tqdm(range(total_steps), desc="Training")

    for step in pbar:
        epoch = step // steps_per_epoch
        inner_step = step % steps_per_epoch

        # Shuffle once per epoch
        if inner_step == 0:
            ts_shuffled = ts_all[torch.randperm(nt_all)]

        # Select current batch
        start = inner_step * batch_size
        ts_batch = ts_shuffled[start:start + batch_size]
        pbar.set_description(f"Epoch {epoch}, Step {inner_step}/{steps_per_epoch-1}")

        optimizer.zero_grad()        

        # print(f"ts_batch: {ts_batch}")
        nt=ts_batch.shape[0]
        Mask_DQB=torch.ones_like(Mask[ts_batch],device=Mask.device)
        # Mask_DQB=Mask[ts_batch]

        sk_nt_node_xyz_1=Transls_key[ts_batch][:,edges[:,1]] # (nt,ne,3)
        sk_nt_node_xyz_2=sk_nt_node_xyz_1.reshape(nt,-1,n_neighbors,3) # (nt,nu,n_neighbors,3)
        sk_nt_node_xyzw_1=Oris_xyzw_key_norm[ts_batch][:,edges[:,1]] # (nt,ne,4)
        sk_nt_node_wxyz_1=sk_nt_node_xyzw_1[:,:,[3,0,1,2]]
        sk_nt_node_quat_2=sk_nt_node_wxyz_1.reshape(nt,-1,n_neighbors,4) # (nt,nu,n_neighbors,4)

        sk_nt_src_node_xyz=sk_src_node_xyz_2.expand(nt, -1, -1, -1)[Mask_DQB] # (n_mask,n_neighbors,3)
        sk_nt_src_node_quat=sk_src_node_quat_2.expand(nt, -1, -1, -1)[Mask_DQB] # (n_mask,n_neighbors,4)
        sk_nt_dst_node_xyz=sk_nt_node_xyz_2[Mask_DQB] # (n_mask,n_neighbors,3)
        sk_nt_dst_node_quat=sk_nt_node_quat_2[Mask_DQB] # (n_mask,n_neighbors,4)

        # error propagation
        I0=torch.eye(3,device=device).expand(nt,nu,-1,-1).clone() # [nt,nu,3,3]
        I1=I0 # test # [nt,nu,3,3]
        I1_masked=I1[Mask_DQB] # [n_mask,3,3]

        if torch.isnan(weight_param_key_optimizable).any():
            print("weight_param_key_optimizable is nan")
            breakpoint()

        src_xyz=src_xyz_1.expand(nt, -1, -1)[Mask_DQB] # (n_mask,3)
        src_wxyz_1=src_xyzw_1[:,[3,0,1,2]]
        src_quat=src_wxyz_1.expand(nt, -1, -1)[Mask_DQB] # (n_mask<=nt*nu,4)

        weight_edges=cal_weight_edges2(weight_param_key_optimizable,exp_sq_dists_key_nonkey_edges,edges,nu) # [ne] # CHECK: 2000
        weight_per_nonkeyGaussian=weight_edges.reshape(-1,n_neighbors) # [nu,n_neighbors]
        sk_w=weight_per_nonkeyGaussian.expand(nt, -1, -1)[Mask_DQB] # (n_mask,n_neighbors)

        # __LBS_warp2__ __DQB_warp2__
        mu_dst, quat_dst=__DQB_warp2__( # [n_mask,3], [n_mask,4]
            sk_w, # [N,K]
            src_xyz, # [N,3]
            sk_nt_src_node_xyz,
            sk_nt_src_node_quat,
            sk_nt_dst_node_xyz, # [N,K,3]
            sk_nt_dst_node_quat, # [N,K,4]
            dyn_o=None,
            src_quat=src_quat, # [N,4] # wxyz
        )

        quat_dst_xyzw=quat_dst[:,[1,2,3,0]] # [n_mask,4] 
        if not torch.allclose(quat_dst_xyzw.norm(dim=-1),torch.ones(quat_dst_xyzw.shape[0],device=quat_dst_xyzw.device)):
            print("quat_dst_xyzw's norm != 1")
            breakpoint()
        Transls_interpolation_masked=mu_dst # [n_mask,3]
        Oris_interpolation_masked=quat_dst_xyzw # [n_mask,4]


        # error propagation
        a1=(Transls_interpolation_masked-Transls[ts_batch][Mask_DQB])[:,None,:]  # (n_mask, 1, 3)
        a1_T=a1.transpose(-1, -2) # (n_mask, 3, 1)
        squared_norm=(a1@I1_masked@a1_T).squeeze(-1,-2) # [n_mask]
        Dist_confident_pts_error = torch.sqrt(squared_norm + 1e-8) # [n_mask] # add epsilon for numerical stability
        loss_dist_confident_pts = torch.mean(Dist_confident_pts_error)

        # Loss : Confident points distance error
        Ori_confident_pts_error=torch.norm(Oris_interpolation_masked-Oris_xyzw[ts_batch][Mask_DQB],dim=-1) # [n_mask]
        loss_ori_confident_pts = torch.mean(Ori_confident_pts_error)


        a_loss_dist_confident_pts=1*loss_dist_confident_pts
        a_loss_ori_confident_pts=1*loss_ori_confident_pts

        
        total_loss = a_loss_dist_confident_pts + a_loss_ori_confident_pts

        # Backpropagation and optimization step
        total_loss.backward()
        optimizer.step()

        # Adjust learning rate with the scheduler
        scheduler.step(total_loss)

        # # Logging progress
        if epoch % log_interval == 0 or epoch == max_epochs - 1:
            print(f"Epoch {epoch}, L: {total_loss.item():.6f}, "
                f"ConfDist: {a_loss_dist_confident_pts.item():.6f}, "
                f"a_loss_ori_confident_pts: {a_loss_ori_confident_pts.item():.6f}, "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        pbar.set_description(f"L:{total_loss.item():.6f}, "+
                         f"c:{a_loss_dist_confident_pts.item():.6f}, "+
                         f"oc:{a_loss_ori_confident_pts.item():.6f}, "+
                         f"lr:{optimizer.param_groups[0]['lr']:.2e}")

        # Early stopping
        if optimizer.param_groups[0]['lr'] < min_lr * 2: #and a_loss_dist < 1e-3 and a_loss_oris_distance_confident_pts < 1e-1:
            print(f"Stopping early at epoch {epoch}.")
            break
    
    
    # ********************* new trying begin *******************
    # Step 1: calculate src_xyz_1_new and src_xyzw_1_new
    weight_param_key_optimizable=weight_param_key_optimizable.abs()
    weight_edges=cal_weight_edges2(weight_param_key_optimizable,exp_sq_dists_key_nonkey_edges,edges,nu) # [ne] # CHECK: 2000
    weight_per_nonkeyGaussian=weight_edges.reshape(-1,n_neighbors) # (nu,n_neighbors)

    weight_distance_optimizable= torch.zeros((nu,nk),dtype=torch.float32,device=device) # TODO
    weight_distance_optimizable[edges[:,0],edges[:,1]]=weight_edges
    weight_distance_optimizable = nn.Parameter(weight_distance_optimizable.to(device))
    
    # print(np.histogram((weight_per_nonkeyGaussian_adjusted>0.1).sum(dim=-1).cpu().detach().numpy(),bins=4))
    print("hist weight_edges:",np.histogram(weight_edges.cpu().detach().numpy(),bins=5))

    sk_w=weight_per_nonkeyGaussian # (nu,n_neighbors)
    
    sk_dst_node_xyz_1=Transls_key_optimized[ts_most_confident_repeat,edges[:,1]] # (ne,3)
    sk_dst_node_xyzw_1=Oris_xyzw_key_optimized_norm[ts_most_confident_repeat,edges[:,1]] # (ne,4)
    sk_dst_node_wxyz_1=sk_dst_node_xyzw_1[:,[3,0,1,2]]
    sk_dst_node_xyz_2=sk_dst_node_xyz_1.reshape(-1,n_neighbors,3) # (nu,n_neighbors,3)
    sk_dst_node_quat_2=sk_dst_node_wxyz_1.reshape(-1,n_neighbors,4) # (nu,n_neighbors,4)

    src_wxyz_1=src_xyzw_1[:,[3,0,1,2]]
    mu_dst, quat_dst=__DQB_warp2__( # [nu,3], [nu,4]
        sk_w, # [N,K]
        src_xyz_1, # [N,3]
        sk_src_node_xyz_2, # [nu,n_neighbors,3]
        sk_src_node_quat_2, # [nu,n_neighbors,4]
        sk_dst_node_xyz_2, # [N,K,3]
        sk_dst_node_quat_2, # [N,K,4]
        dyn_o=None,
        src_quat=src_wxyz_1, # [N,4] # wxyz
    )
    src_xyz_1_new=mu_dst
    src_xyzw_1_new=quat_dst[:,[1,2,3,0]] # [nu,4]
    # for further optimization
    blend_opt_dict = {
        'weight_param_key_optimizable': weight_param_key_optimizable, # (nu,n_neighbors)
        'src_xyz_1': src_xyz_1_new, # (nu,3) # CHECK
        'src_xyzw_1': src_xyzw_1_new, # (nu,4) # CHECK
        'per_nonkeyGaussian_dweight': per_nonkeyGaussian_dweight, # (nu,n_neighbors)
        'exp_sq_dists_key_nonkey_edges': exp_sq_dists_key_nonkey_edges, # (ne)
        'ts_most_confident': ts_most_confident, # (nu)
    }

    # Step 2: calculate Transls_optimized and Oris_xyzw_optimized
    Transls_optimized,Oris_xyzw_optimized=[],[]
    ts_batches = torch.split(ts_all, batch_size)
    for ts_batch in ts_batches:
        nt=ts_batch.shape[0]
        Mask_DQB=torch.ones_like(Mask[ts_batch],device=Mask.device)
        sk_w=weight_per_nonkeyGaussian.expand(nt, -1, -1)[Mask_DQB] # (n_mask,n_neighbors)

        sk_src_node_xyz_1=Transls_key[ts_most_confident_repeat,edges[:,1]] # (ne,3)
        sk_src_node_xyzw_1=Oris_xyzw_key_norm[ts_most_confident_repeat,edges[:,1]] # (ne,4)
        sk_src_node_wxyz_1=sk_src_node_xyzw_1[:,[3,0,1,2]]
        sk_src_node_xyz_2=sk_src_node_xyz_1.reshape(-1,n_neighbors,3) # (nu,n_neighbors,3)
        sk_src_node_quat_2=sk_src_node_wxyz_1.reshape(-1,n_neighbors,4) # (nu,n_neighbors,4)

        sk_nt_src_node_xyz=sk_src_node_xyz_2.expand(nt,-1,-1,-1)[Mask_DQB] # (n_mask,n_neighbors,3)
        sk_nt_src_node_quat=sk_src_node_quat_2.expand(nt,-1,-1,-1)[Mask_DQB] # (n_mask,n_neighbors,4)

        # key points dst needed to be updated; src is the same
        sk_nt_node_xyz_op_1=Transls_key_optimized[ts_batch][:,edges[:,1]] # (nt,ne,3)
        sk_nt_node_xyz_op_2=sk_nt_node_xyz_op_1.reshape(nt,-1,n_neighbors,3) # (nt,nu,n_neighbors,3)
        sk_nt_node_xyzw_op_1=Oris_xyzw_key_optimized_norm[ts_batch][:,edges[:,1]] # (nt,ne,4)
        sk_nt_node_wxyz_op_1=sk_nt_node_xyzw_op_1[:,:,[3,0,1,2]]
        sk_nt_node_quat_op_2=sk_nt_node_wxyz_op_1.reshape(nt,-1,n_neighbors,4) # (nt,nu,n_neighbors,4)

        sk_nt_dst_node_xyz_op=sk_nt_node_xyz_op_2[Mask_DQB] # (n_mask,n_neighbors,3)
        sk_nt_dst_node_quat_op=sk_nt_node_quat_op_2[Mask_DQB] # (n_mask,n_neighbors,4)

        # nonkey points update when src_xyz_1 and src_xyzw_1 are trainable
        src_xyz=src_xyz_1.expand(nt,-1,-1)[Mask_DQB] # (n_mask,3)
        src_wxyz_1=src_xyzw_1[:,[3,0,1,2]]
        src_quat=src_wxyz_1.expand(nt,-1,-1)[Mask_DQB] # (n_mask<=nt*nu,4)

        # __LBS_warp2__ __DQB_warp2__
        mu_dst, quat_dst=__DQB_warp2__( # [n_mask,3], [n_mask,4]
            sk_w, # [N,K]
            src_xyz, # [N,3]
            sk_nt_src_node_xyz,
            sk_nt_src_node_quat,
            sk_nt_dst_node_xyz_op, # [N,K,3]
            sk_nt_dst_node_quat_op, # [N,K,4]
            dyn_o=None,
            src_quat=src_quat, # [N,4] # wxyz
        )

        quat_dst_xyzw=quat_dst[:,[1,2,3,0]] # [n_mask,4] 
        if not torch.allclose(quat_dst_xyzw.norm(dim=-1),torch.ones(quat_dst_xyzw.shape[0],device=quat_dst_xyzw.device)):
            print("quat_dst_xyzw's norm != 1")
            breakpoint()
        Transls_optimized_batch=mu_dst.reshape(nt,nu,3) # [nt,nu,3]
        Oris_xyzw_optimized_batch=quat_dst_xyzw.reshape(nt,nu,4) # [nt,nu,4]
        Transls_optimized.append(Transls_optimized_batch)
        Oris_xyzw_optimized.append(Oris_xyzw_optimized_batch)
    Transls_optimized=torch.cat(Transls_optimized,dim=0) # [nt_all,nu,3]
    Oris_xyzw_optimized=torch.cat(Oris_xyzw_optimized,dim=0) # [nt_all,nu,4]

    # ********************* new trying  end *******************
    blend_dict={}
    return Transls_optimized, Oris_xyzw_optimized, weight_param_key_optimizable,weight_distance_optimizable,blend_dict,blend_opt_dict

def get_knn_edges_key(Transls,Mask,k):
    # group by variance of relative motion 
    # Transls [nt,nk,3]
    
    def nanstd(o, dim, keepdim=False):
        result = torch.sqrt(
                    torch.nanmean(
                        torch.pow( torch.abs(o-torch.nanmean(o,dim=dim).unsqueeze(dim)),2),
                        dim=dim
                    )
                )  
        if keepdim:
            result = result.unsqueeze(dim)
        return result

    nu=Transls.shape[1]
    
    # Dist=torch.cdist(Transls, Transls, p=2)
    Dist = torch.norm(Transls[:, :, None, :] - Transls[:, None, :, :], dim=-1) # [nt, nu, nk]
    Mask_dist=Mask[:, :, None] * Mask[:, None, :] # [nt, nu, nk]
    Dist_masked = torch.where(Mask_dist, Dist, torch.tensor(float('nan'), device=Dist.device))
    std_dist=nanstd(Dist_masked,dim=0) # [nu,nk]

    mean_dist_masked=Dist_masked.nanmean(dim=0) # [nu,nk] # may include nan
    Dist_masked2 = torch.where(Mask.unsqueeze(1).expand(-1, nu, -1), Dist, torch.tensor(float('nan'), device=Dist.device))
    mean_dist_masked2=Dist_masked2.nanmean(dim=0) # [nu,nk] # should not include nan # may still have nan
    if torch.any(mean_dist_masked2.isnan())==False:
        mean_dist_masked = torch.where(~torch.isnan(std_dist), mean_dist_masked, mean_dist_masked2)
    else:
        mean_dist_masked = Dist.mean(dim=0) # [nu,nk]
    Dist_masked2 = torch.where(Mask_dist, Dist_masked, torch.tensor(float('-inf'), device=std_dist.device))

    max_dist = torch.max(Dist_masked2, dim=0).values # has -inf
    max_dist2 = torch.where(~torch.isinf(max_dist), max_dist, torch.tensor(float('inf'), device=max_dist.device))

    values, indices = torch.topk(max_dist2, k=k+1, dim=1, largest=False)  # shape [nu, k+1] # Step 1: Get k+1 smallest distances (to exclude self)

    # Step 2: Drop the first column (self-distance)
    indices = indices[:, 1:]  # shape [nu, k]
    values = values[:, 1:]    # optional, only if you use values

    nu_indices = torch.arange(max_dist2.shape[0]).unsqueeze(1).repeat(1, k)  # shape [nu, k]
    edges = torch.stack([nu_indices, indices], dim=2).reshape(-1, 2)  # shape [nu * k, 2]

    return edges, mean_dist_masked

def get_knn_edges_others(Transls,Mask,Transls2,Mask2,k):
    # group by variance of relative motion
    # Transls [nt,nk,3]
    # Chunked over nu to avoid OOM from (nt,nu,nk,3) broadcast (e.g. 49 GB for nu=14740,nk=1012,nt=277)

    nu=Transls2.shape[1]
    nk=Transls.shape[1]

    chunk_size = 200  # (nt, chunk, nk, 3) intermediate ~675 MB per chunk

    mean_dist_masked_key = torch.zeros(nu, nk)
    max_dist = torch.full((nu, nk), float('-inf'))

    nan_val = torch.tensor(float('nan'))
    ninf_val = torch.tensor(float('-inf'))

    for start in range(0, nu, chunk_size):
        end = min(start + chunk_size, nu)
        c = end - start

        Dist_chunk = torch.norm(
            Transls2[:, start:end, None, :] - Transls[:, None, :, :],
            dim=-1
        )  # (nt, c, nk)

        Mask_expand_chunk = Mask.unsqueeze(1).expand(-1, c, -1)  # (nt, c, nk)

        Dist_masked_nan = torch.where(Mask_expand_chunk, Dist_chunk, nan_val)
        mean_dist_masked_key[start:end] = Dist_masked_nan.nanmean(dim=0)  # (c, nk)

        Dist_masked_ninf = torch.where(Mask_expand_chunk, Dist_chunk, ninf_val)
        max_dist[start:end] = torch.max(Dist_masked_ninf, dim=0).values  # (c, nk)

    max_dist2 = torch.where(~torch.isinf(max_dist), max_dist, torch.tensor(float('inf')))

    values, indices = torch.topk(max_dist2, k=k, dim=1, largest=False)  # smallest k distances
    nu_indices = torch.arange(nu).unsqueeze(1).repeat(1, k)  # shape [nu, k]
    edges = torch.stack([nu_indices, indices], dim=2).reshape(-1, 2)  # shape [nu * k, 2]
    return edges, mean_dist_masked_key

def run_batch(Transls2,Oris_xyzw2,Contribs2,
              Transls,Oris_xyzw,Transls_optimized,Oris_xyzw_optimized,
              Contribs,contrib_threshold,
              W2Cs,Depths_info_perG_nonkey,
              Mask):
    # Initializations
    # 1. vertices
    # 1.1 positions
    nu=Transls2.shape[1]
    nk=Transls.shape[1]
    nt=Transls.shape[0]
    Transls2=torch.tensor(Transls2,dtype=torch.float32)
    Transls_optimizable2=Transls2.clone().requires_grad_(True)
    if nt>40:
        conditiona=Contribs2>0.5*np.max(Contribs2,axis=0)
        condition2=conditiona
    else:
        conditionb=Contribs2>contrib_threshold
        condition2=conditionb
    Mask2=torch.tensor(condition2,dtype=torch.bool)

    # # 1.2 orientations
    Oris_xyzw_optimizable2=Oris_xyzw2.clone().requires_grad_(True) # LieTensor SO3

    # 2. edges
    # 2.1 position edges
    Dists_optimizable2=torch.rand(nu,nk,dtype=torch.float32,requires_grad=True)

    # 2.2 orientation edges
    Ori_distances_optimizable2=torch.rand(nu,nk,dtype=torch.float32,requires_grad=True)*2

    # 3. edge weight
    # 3.1 position stds
    weight_distance_optimizable2=torch.zeros((nu,nk),dtype=torch.float32,requires_grad=True)

    # knn graph (minmax distance + contrib version)
    n_neighbors=5
    edges_others,mean_dists=get_knn_edges_others(Transls,Mask,Transls2,Mask2,k=n_neighbors) 
    sq_dists_5=mean_dists[edges_others[:,0],edges_others[:,1]].reshape(-1,n_neighbors) # [nu,n_neighbors]
    ts_most_confident=np.argmax(Contribs2[:,:],axis=0) # [nu]
    ts_most_confident=torch.tensor(ts_most_confident,dtype=torch.long) # [nu]


    print(f"{torch.cuda.memory_allocated()/1024/1024/1024}GB")
    torch.cuda.empty_cache()
    print(f"{torch.cuda.memory_allocated()/1024/1024/1024}GB")

    # Method 3:
    # use Dual Quat Blending
    sq_dists_key_nonkey_edges=sq_dists_5.reshape(-1) # [ne]
    weight_param_key_optimizable=1*torch.ones(nk,dtype=torch.float32,requires_grad=True) # 2000 [TODO]
    per_nonkeyGaussian_dweight=torch.zeros(nu,n_neighbors,dtype=torch.float32,requires_grad=True)

    # batch training along t
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Transls_optimized2,Oris_xyzw_optimized2, weight_param_key_optimized2,weight_distance_optimized2,blend_dict,blend_opt_dict=\
        optimize_others_with_orientation_edges4(\
            edges_others,\
            Transls2, Transls_optimizable2.clone(), Mask2,  \
            Oris_xyzw2, Oris_xyzw_optimizable2.clone(),  \
            Transls,Oris_xyzw,Transls_optimized,Oris_xyzw_optimized,\
            weight_param_key_optimizable,sq_dists_key_nonkey_edges,\
            ts_most_confident,per_nonkeyGaussian_dweight,\
            W2Cs,Depths_info_perG_nonkey,
            device)

    # keep
    edges_others_optimized=edges_others
    Dists_optimized2=Dists_optimizable2
    Ori_distances_optimized2=Ori_distances_optimizable2

    return Transls_optimized2,Dists_optimized2,weight_distance_optimized2,Oris_xyzw_optimized2, Ori_distances_optimized2,edges_others_optimized,edges_others,weight_param_key_optimized2,ts_most_confident,blend_dict,blend_opt_dict

def main3(save_dir,file_effective,file_name_graph,depth_ratio,version_key_edges,threshold_min_set):
    import sys
    import os
    sys.path.append(os.path.dirname("../flow3d"))

    import torch
    import numpy as np
    from scipy.spatial.transform import Rotation
    import matplotlib.pyplot as plt
    import pickle
    import random
    import glob

    import pypose as pp
    from torch import nn

    from itertools import chain
    from scipy.spatial import KDTree,Delaunay,distance_matrix
    import open3d as o3d
    from collections import defaultdict
    import roma

    # set args
    work_dir=save_dir

    dict_dir=work_dir+"/out_dicts3" # has w2c and camera's K

    flag_draw_contribs_sample=False
    flag_visulize_sparisity=False
    flag_visulize_key_point=False
    flag_visulize_nonkey_point=False

    flag_auto_threshold=False
    contrib_threshold=2 # usually 2
    threshold=2e-0 ## TODO # for non-key points # fixed threshold


    # [1] load data
    nt=count_out_dict_files(dict_dir)

    with open(f"{dict_dir}/{file_effective}", 'rb') as file:
        effective_dict = pickle.load(file)
        inds_effective = effective_dict['ids_eff2'] #ids_eff1
        # Contribs_up = effective_dict['Contribs_up']
        threshold_min = effective_dict['threshold_min']
        threshold_mean = effective_dict['threshold_mean']
        Contribs_all = effective_dict['Contribs']
        print(f"threshold_min:{threshold_min}, threshold_mean:{threshold_mean}")

    if threshold_min_set!=-1:
        print(f"set threshold_min to {threshold_min_set}.")
        threshold_min=threshold_min_set
    else:
        print(f"use threshold_min from effective_dict: {threshold_min}.")

    if flag_auto_threshold:
        print(f"set auto threshold to {threshold_min}.")
        contrib_threshold=threshold_min
        threshold=threshold_min

    print(f"set threshold_min to {threshold_min}.")

    device='cuda'
    ids=inds_effective

    i_target=0
    i_show=10
    # Stream key-point data from pkl files (avoids loading all nt files into RAM simultaneously)
    Ng=Contribs_all.shape[1]
    _Transls_rows, _Quats_rows, _W2Cs_rows = [], [], []
    for t in range(nt):
        with open(f"{dict_dir}/out_dict_{t}.pkl", 'rb') as file:
            d = pickle.load(file)
        _Transls_rows.append(d["means"][ids].cpu().numpy())
        _Quats_rows.append(d["quats"][ids].cpu().numpy())
        _W2Cs_rows.append(d["w2c"].cpu().numpy())
        del d
    Transls=np.array(_Transls_rows); del _Transls_rows  #(nt,nk,3)
    Quats_wxyz=np.array(_Quats_rows); del _Quats_rows   # wxyz (nt,nk,4)
    Quats_xyzw=Quats_wxyz[...,[1,2,3,0]] # xyzw
    Contribs=Contribs_all[:,ids]
    Contribs_sort=np.array([get_order(Contribs[:,i]) for i in range(Contribs.shape[1])]).transpose(1,0)
    W2Cs=np.array(_W2Cs_rows); del _W2Cs_rows
    SE3s=np.concatenate([Transls,Quats_xyzw],axis=-1) # xyz xyzw (required by pypose)
    SE3s=torch.tensor(SE3s,dtype=torch.float32) #,device='cuda')
    Oris_xyzw=pp.LieTensor(SE3s[:,:,3:],ltype=pp.SO3_type)
    Oris=pp.LieTensor(SE3s[:,:,3:],ltype=pp.SO3_type).Log()
    lie_tensor_SE3s=pp.LieTensor(SE3s, ltype=pp.SE3_type)
    Depths_info_perG=depth_ratio*torch.ones((nt,Ng),dtype=torch.float32) # depths overwritten with constant anyway
    print(f"TEST: set Depths_info_perG={depth_ratio}*torch.ones_like(Depths_info_perG).") # test
    Depths_info_perG_key=Depths_info_perG[:,ids]
    nk=len(ids) # num of key points
    print("nk:",nk)
    print("num Gaussians:", Ng)

    # draw
    if flag_draw_contribs_sample:
        # i_show=20
        # means=Transls[:,i_show]
        # vars0=Vars[:,i_show]
        # # sc=ax.scatter(means[:,0], means[:,1], means[:,2],s=1,alpha=0.6,c=np.log(np.sum(vars0,axis=1)),cmap='viridis')

        # means=Transls[:,i_target]
        # vars1=Vars[:,i_target]
        # # sc=ax.scatter(means[:,0], means[:,1], means[:,2],s=1,alpha=0.6,c=np.log(np.sum(vars1,axis=1)),cmap='viridis')

        plt.figure(figsize=(10,4))
        # plt.subplot(1,3,1)
        # plt.plot(vars0,c='r')
        # plt.plot(vars1,c='b',alpha=0.6)
        # plt.xlabel("time frame")
        # plt.ylabel("uncertainty")
        # plt.ylim([0,1])
        i_show=123
        plt.subplot(1,3,2)
        plt.plot(Contribs[:,i_show],c='r')
        # plt.plot(Contribs[:,i_target],c='b',alpha=0.6)
        plt.ylabel("contribution")
        plt.xlabel("time frame")
        plt.subplot(1,3,3)
        plt.plot(Contribs_sort[:,i_show],c='r')
        # plt.plot(Contribs_sort[:,i_target],c='b',alpha=0.6)
        plt.ylabel("contribution sort")
        plt.xlabel("time frame")
        plt.show()


    # [2] Graph initialization for key Gaussians
    # Initializations
    # 0. naive ori distance
    # Chunked: avoids (nt,nk,nk,4) OOM (e.g. 46 GB for nt=427,nk=1500)
    Ori_distances_naive=compute_ori_distances_chunked(Oris_xyzw)

    # 1. vertices
    # 1.1 positions
    Transls=torch.tensor(Transls,dtype=torch.float32)
    Transls_optimizable=Transls.clone().requires_grad_(True)
    Mask=torch.tensor(Contribs>contrib_threshold,dtype=torch.bool)

    # # 1.2 orientations
    Oris_xyzw_optimizable=Oris_xyzw.clone().requires_grad_(True) # LieTensor SO3

    # 2. edge distance
    # Incremental mean avoids (nt,nk,nk) float64 array (e.g. 7.7 GB for nt=427,nk=1500)
    _Dists_sum = np.zeros((nk, nk), dtype=np.float64)
    for t in range(nt):
        _Dists_sum += distance_matrix(Transls[t], Transls[t])
    Dists_optimizable = _Dists_sum / nt
    del _Dists_sum
    Dists_optimizable=torch.tensor(Dists_optimizable,dtype=torch.float32,requires_grad=True)

    Ori_distances_optimizable=Ori_distances_naive.clone()
    Ori_distances_optimizable=torch.tensor(Ori_distances_optimizable,dtype=torch.float32,requires_grad=True)

    # 3. edge weight
    weight_distance_optimizable=1*torch.ones(Dists_optimizable.shape,dtype=torch.float32,requires_grad=True)
    weight_distance_optimizable=weight_distance_optimizable.fill_diagonal_(1e-10)#(-100) # 1e-10

    # use the different weight
    weight_ori_distance_optimizable=1*torch.ones(Ori_distances_optimizable.shape,dtype=torch.float32,requires_grad=True)
    weight_ori_distance_optimizable=weight_ori_distance_optimizable.fill_diagonal_(1e-10)#(-100) # 1e-10

    # 4. get local edges
    n_neighbors=120
    if version_key_edges=="max_contrib":
        edges_graph=get_knn_edges(n_neighbors,Transls,Contribs,Transls,Contribs,contrib_threshold,flag_remove_self=True) # edge version: max contrib version
    elif version_key_edges=="minmax_distance_contrib":
        # edges_graph,_=get_knn_edges_others(Transls,Mask,Transls,Mask,k=n_neighbors) # edge version: minmax distance + contrib mask version
        edges_graph,_=get_knn_edges_key(Transls,Mask,k=n_neighbors)
    elif version_key_edges=="max_native":
        edges_graph,_=get_knn_edges_naive(n_neighbors,Transls,Transls,flag_remove_self=True) # edge version: max contrib version
    else:
        breakpoint()
        raise ValueError("version_key_edges should be max_contrib or minmax_distance_contrib or max_native")
    weight_distance_optimizable=torch.zeros(Dists_optimizable.shape,dtype=torch.float32)
    weight_distance_optimizable[edges_graph[:,0],edges_graph[:,1]]=1 #1e-5
    weight_distance_optimizable.requires_grad=True

    # [3] optimization for key Guassians
    print("===========================================")
    print("Start Optimization for key Guassians")
    print("-------------------------------------------")
    
    torch.cuda.empty_cache()
    Transls_optimized,Dists_optimized,weight_distance_optimized,Oris_xyzw_optimized, Ori_distances_optimized=\
        optimize_overall_with_orientation_edges(\
            edges_graph,\
            Transls, Transls_optimizable.clone(), Mask, Dists_optimizable.clone(), weight_distance_optimizable.clone(),\
            Oris_xyzw, Oris_xyzw_optimizable.clone(), Ori_distances_optimizable.clone(), weight_ori_distance_optimizable, \
            None, None, None,None,\
            W2Cs,Depths_info_perG_key)
    
    print("-------------------------------------------")
    print("Finish Optimization for key Guassians")
    print("===========================================")

    if flag_visulize_sparisity:
        weight_distance_optimized[2,:10],weight_distance_optimized[:10,2] # check symetry
        # look at normed weight
        torch.abs(weight_distance_optimized)/torch.norm(weight_distance_optimized,dim=-1,p=1)[:,None]
        data = (torch.abs(weight_distance_optimized)/torch.norm(weight_distance_optimized,dim=-1,p=1)[:,None]).cpu().detach().numpy()
        print((data>1e-6).sum(axis=-1)[:50]) # check sparsity # lower than 1e-7 or 1e-6 mean no meaningful contribution
        np.histogram((data>1e-6).sum(axis=-1))


        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np

        # Example 2D array
        print((data>0.02).sum())

        # Create a meshgrid for the X and Y coordinates
        x = np.arange(data.shape[1])
        y = np.arange(data.shape[0])
        x, y = np.meshgrid(x, y)

        # Create a figure and a 3D axis
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the surface
        ax.plot_surface(x, y, np.flip(data,axis=1), cmap='viridis')

        # Add labels
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_yticks([0,100,200,300,400])
        ax.set_yticklabels(nk-np.arange(0,500,100))
        ax.set_zlabel('Weight Values')
        # ax.set_zlim(0,0.5)
        ax.set_title('Edge Weights')

        plt.show()


    if flag_visulize_key_point:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np

        t=Transls.shape[0]-1 
        incorrect_vertex_ids=np.array([i for i in range(nk) if Contribs[t,i]<contrib_threshold])
        correct_vertex_ids=np.array([i for i in range(nk) if Contribs[t,i]>contrib_threshold])

        pts1=Transls_optimized.cpu().detach().numpy()[t]
        pts2=Transls.cpu().numpy()[t]
        pts=pts1

        # Generate some random 3D points for two arrays
        x1 = pts[correct_vertex_ids][:,0]
        y1 = pts[correct_vertex_ids][:,1]
        z1 = pts[correct_vertex_ids][:,2]

        x2 = pts[incorrect_vertex_ids][:,0]
        y2 = pts[incorrect_vertex_ids][:,1]
        z2 = pts[incorrect_vertex_ids][:,2]

        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the first set of points in blue
        ax.scatter(x1, y1, z1, c='b', label='correct_vertex_ids Points',s=2)

        # Plot the second set of points in red
        ax.scatter(x2, y2, z2, c='r', label='incorrect_vertex_ids Points',s=2)

        # Add labels
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        # ax.set_box_aspect([1.0, 1.0, 1.0])
        ax.axis('equal')
        # Add legend
        ax.legend()

        # Show the plot
        plt.show()

    # save
    # get optimized graph from normed weights
    norm_weight_distance_optimized = (torch.abs(weight_distance_optimized)/torch.norm(weight_distance_optimized,dim=-1,p=1)[:,None]).cpu().detach().numpy()
    edges_graph_optimized=[]
    for i in range(nk):
        for j in range(nk):
            if norm_weight_distance_optimized[i,j]>1e-6:
                edges_graph_optimized.append([j,i])
    edges_graph_optimized=torch.tensor(edges_graph_optimized,dtype=torch.long)

    Ori_opt=Oris_xyzw_optimized
    Nodes_opt=torch.concatenate([Transls_optimized.transpose(1,0),Ori_opt.transpose(1,0)],dim=-1) # (nk,nt,7)
    Edges_keep=[[] for i in range(nk)]
    # for e in edges_opt:
    for e in edges_graph_optimized:
        # Edges_keep[e[0]].append([e[1],e[0]])
        Edges_keep[e[1]].append([e[0],e[1]])

    Edges_keep=[torch.tensor(Edges_keep[i],device=device) for i in range(nk)]
    num_Edges_keep=[Edges_keep[i].shape for i in range(nk)]
    # num_Edges_keep

    # check
    torch.norm(Oris_xyzw_optimized[0],dim=-1)[:10]

    # for save
    optimized_key_output={"Transls_optimized":Transls_optimized,\
                    "Dists_optimized":Dists_optimized,\
                    "weight_distance_optimized":weight_distance_optimized,\
                    "Oris_xyzw_optimized":Oris_xyzw_optimized, \
                    "Ori_distances_optimized":Ori_distances_optimized}

    # Save variables to a file
    # torch.save({
    #             'dicts': dicts,\
    #             'ids': ids,\
    #             'Contribs': Contribs,\
    #             'Transls': Transls,\
    #             'Oris_xyzw': Oris_xyzw,\
    #             'Transls_optimizable': Transls_optimizable,\
    #             'Dists_optimizable': Dists_optimizable,\
    #             'weight_distance_optimizable':weight_distance_optimizable,\
    #             'Ori_distances_optimizable':Ori_distances_optimizable,\
    #             'Transls_optimized': Transls_optimized,\
    #             'Dists_optimized': Dists_optimized,\
    #             "weight_distance_optimized":weight_distance_optimized,\
    #             "Oris_xyzw_optimized":Oris_xyzw_optimized, \
    #             "Ori_distances_optimized":Ori_distances_optimized,\
    #             "edges_graph":edges_graph,\
    #             "edges_graph_optimized":edges_graph_optimized,\
    #             "Nodes_opt":Nodes_opt,\
    #             "Edges_keep":Edges_keep,\
    #             "optimized_key_output":optimized_key_output,\
    #             # "Transls_optimized1":Transls_optimized1,\
    #             # "Dists_optimized1":Dists_optimized1,\
    #             # "weight_distance_optimized1":weight_distance_optimized1,\
    #             "device":device,\
    #             "work_dir":work_dir}\
                
    #             , 'variables_8.pth')
    
    flag_read=False
    if flag_read:
        # Read variables from the file
        checkpoint = torch.load('variables_8.pth', weights_only=False)
        dicts = checkpoint['dicts']
        ids = checkpoint['ids']
        Contribs = checkpoint['Contribs']
        Transls = checkpoint['Transls']
        Oris_xyzw = checkpoint['Oris_xyzw']
        Transls_optimized = checkpoint['Transls_optimized']
        Dists_optimized = checkpoint['Dists_optimized']
        weight_distance_optimized = checkpoint['weight_distance_optimized']
        Oris_xyzw_optimized = checkpoint['Oris_xyzw_optimized']
        Ori_distances_optimized = checkpoint['Ori_distances_optimized']
        edges_graph = checkpoint['edges_graph']
        edges_graph_optimized=checkpoint['edges_graph_optimized']
        Nodes_opt = checkpoint['Nodes_opt']
        Edges_keep = checkpoint['Edges_keep']
        optimized_key_output = checkpoint['optimized_key_output']
        device=checkpoint['device']
        work_dir=checkpoint['work_dir']
        nk=len(ids)


    # [4] Graph initialization for non-key Gaussians
    # 1. prepare data for non-key Gaussians
    Ng=Contribs_all.shape[1]  # total Gaussians (same as Contribs_all cols)
    ids2=[i for i in range(Ng) if i not in ids] # ids of non-target nodes
    Contribs2_temp=Contribs_all[:,ids2]
    Contribs2_temp_max=Contribs2_temp.max(axis=0)
    if nt>350:
        n_nonkey_max=36000 # 36000 # 150000 72000 too large memory for block teddy wheel
    elif nt>40:
        n_nonkey_max=72000
    else:
        n_nonkey_max=150000
    if Contribs2_temp_max.shape[0]>n_nonkey_max:
        contribs2_nth_largest = np.sort(Contribs2_temp_max)[-n_nonkey_max] # the nth (12000th/24000th) largest contrib value
    else:
        contribs2_nth_largest = np.sort(Contribs2_temp_max)[0]
    if nt>40: # iphone davis
        contribs2_nth_largest=max(contribs2_nth_largest,0.01) # 0.1
    else: # nvidia
        contribs2_nth_largest=max(contribs2_nth_largest,1e-4)
    threshold=contribs2_nth_largest # rewrite threshold
    ids2=np.array(ids2)[Contribs2_temp.max(axis=0)>threshold].tolist() # remove nodes with small contribs #15


    # Stream non-key data from pkl files
    _Transls2_rows, _Quats2_rows = [], []
    for t in range(nt):
        with open(f"{dict_dir}/out_dict_{t}.pkl", 'rb') as file:
            d = pickle.load(file)
        _Transls2_rows.append(d["means"][ids2].cpu().numpy())
        _Quats2_rows.append(d["quats"][ids2].cpu().numpy())
        del d
    Transls2=np.array(_Transls2_rows); del _Transls2_rows  #(nt,nu,3)
    Quats_wxyz2=np.array(_Quats2_rows); del _Quats2_rows   # wxyz (nt,nu,4)
    Quats_xyzw2=Quats_wxyz2[...,[1,2,3,0]] # xyzw
    Contribs2=Contribs_all[:,ids2]
    Contribs_sort2=np.array([get_order(Contribs2[:,i]) for i in range(Contribs2.shape[1])]).transpose(1,0)
    Oris_xyzw2=pp.LieTensor(Quats_xyzw2,ltype=pp.SO3_type)
    Oris2=pp.LieTensor(Quats_xyzw2,ltype=pp.SO3_type).Log()
    Depths_info_perG_nonkey=Depths_info_perG[:,ids2]
    nu=len(ids2) # num of non-target nodes
    print(f'Ng={Ng},nk=len(ids)={len(ids)},nu=len(ids2)={len(ids2)},threshold={threshold}')

    # 2. run optimizaiton for non-key Guassians
    print("===========================================")
    print("Start Optimization for nonkey Guassians")
    print("-------------------------------------------")

    Transls_optimized2,Dists_optimized2,weight_distance_optimized2,\
        Oris_xyzw_optimized2, Ori_distances_optimized2,edges_others_optimized,edges_others, \
        weight_param_key_optimized2,ts_most_confident,blend_dict,blend_opt_dict \
            =run_batch(Transls2,Oris_xyzw2,Contribs2,\
                Transls,Oris_xyzw,Transls_optimized,Oris_xyzw_optimized,\
                Contribs,contrib_threshold,\
                W2Cs,Depths_info_perG_nonkey,\
                Mask)

    print("weight_param_key_optimized2 hist:",np.histogram(weight_param_key_optimized2.cpu().detach().numpy(),bins=10))

    print("Finish Optimization for nonkey Guassians")
    print("===========================================")



    if flag_visulize_nonkey_point:
        # 3. check
        # just for Method 1
        data = (torch.abs(weight_distance_optimized2)/torch.norm(weight_distance_optimized2,dim=-1,p=1)[:,None]).cpu().detach().numpy()
        np.sort(data[4])[-5:] # check several largest weight
        print((data>1e-6).sum(axis=-1)[:50]) # check sparsity # lower than 1e-7 or 1e-6 mean no meaning contribution
        np.histogram((data>1e-6).sum(axis=-1)) # check sparsity

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np

        t=nt-1 
        incorrect_vertex_ids=np.array([i for i in range(nk) if Contribs[t,i]<contrib_threshold])
        correct_vertex_ids=np.array([i for i in range(nk) if Contribs[t,i]>contrib_threshold])

        pts1=Transls_optimized.cpu().detach().numpy()[t]
        pts2=Transls.cpu().numpy()[t]
        pts=pts1
        pts_others=Transls_optimized2.cpu().detach().numpy()[t]

        # Generate some random 3D points for two arrays
        x1 = pts[correct_vertex_ids][:,0]
        y1 = pts[correct_vertex_ids][:,1]
        z1 = pts[correct_vertex_ids][:,2]

        x2 = pts[incorrect_vertex_ids][:,0]
        y2 = pts[incorrect_vertex_ids][:,1]
        z2 = pts[incorrect_vertex_ids][:,2]

        x3 = pts_others[:,0]
        y3 = pts_others[:,1]
        z3 = pts_others[:,2]

        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the first set of points in blue
        ax.scatter(x1, y1, z1, c='b', label='correct_vertex_ids Points',s=4)

        # Plot the second set of points in red
        ax.scatter(x2, y2, z2, c='r', label='incorrect_vertex_ids Points',s=4)

        ax.scatter(x3, y3, z3, c='k', label='other Points',s=1)

        # Add labels
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        # ax.set_box_aspect([1.0, 1.0, 1.0])
        ax.axis('equal')
        # Add legend
        ax.legend()

        # Show the plot
        plt.show()

    Ori_opt2=torch.tensor(Quats_xyzw2,device=device) # tempory
    # Ori_opt2=Oris_xyzw_optimized2.tensor()
    Nodes_opt2=torch.concatenate([Transls_optimized2.transpose(1,0),Ori_opt2.transpose(1,0)],dim=-1) # (nk,nt,7)
    Edges_keep2=[[] for i in range(nu)]
    # for e in edges_opt:
    for e in edges_others_optimized:
        Edges_keep2[e[0]].append([e[1],e[0]+nk]) # [key id, other id + offset (nk)]
        # Edges_keep2[e[1]].append([e[0],e[1]])

    Edges_keep2=[torch.tensor(Edges_keep2[i],device=device) for i in range(nu)]
    num_Edges_keep2=[Edges_keep2[i].shape for i in range(nu)]

    # combine key and others
    Nodes_opt_combine=torch.cat([Nodes_opt,Nodes_opt2],dim=0)
    Edges_keep_combine=Edges_keep+Edges_keep2
    ids_combine=ids+ids2

    optimized_others_output={"Transls_optimized":Transls_optimized2,\
                  "Dists_optimized":Dists_optimized2,\
                  "weight_distance_optimized":weight_distance_optimized2,\
                  "Oris_xyzw_optimized":Oris_xyzw_optimized2, \
                  "Ori_distances_optimized":Ori_distances_optimized2}


    # [5] Save graph preprocessing data
    quats_xyzw=Nodes_opt[:,:,3:] # xyzw
    quats_wxyz=quats_xyzw[..., [3, 0, 1, 2]] # wxyz
    tracks_SE3_opt=[Nodes_opt[:,:,:3],quats_wxyz] # [[nk,nt,3], [nk,nt,4]]

    # combine
    quats_xyzw_combine=Nodes_opt_combine[:,:,3:] # xyzw
    quats_xyzw_combine=quats_xyzw_combine[..., [3, 0, 1, 2]] # wxyz
    tracks_SE3_opt_combine=[Nodes_opt_combine[:,:,:3],quats_xyzw_combine] # [[nk,nt,3], [nk,nt,4]]

    nk=len(Edges_keep)
    adjacent_matrix=torch.zeros((nk,nk))
    for i in range(nk):
        if Edges_keep[i].shape[0]==0:
            continue
        indices=Edges_keep[i][:,0]
        adjacent_matrix[i,indices]=1

    adjacent_matrix = torch.logical_or(adjacent_matrix.t(), adjacent_matrix).type(torch.int32)

    Edges_keep_bi=[]
    for i in range(nk):
        indices=torch.where(adjacent_matrix[i,:]==1)[0]
        edges_keep_bi=torch.cat((indices.unsqueeze(0),i*torch.ones_like(indices).unsqueeze(0))).t()
        Edges_keep_bi.append(edges_keep_bi)

    Edges_keep_bi_combine=Edges_keep_bi+Edges_keep2


    # Save 1:
    is_target=list(range(len(ids)))
    is_target_combine=list(range(len(ids_combine)))
    opt_dict={"Nodes_opt":Nodes_opt,"tracks_SE3_opt":tracks_SE3_opt,"ids":np.array(ids),\
            "is_target":is_target,"Edges_keep":Edges_keep,"Edges_keep_bi":Edges_keep_bi,\
            "optimized_key_output":optimized_key_output,\
            "Nodes_opt_combine":Nodes_opt_combine,"tracks_SE3_opt_combine":tracks_SE3_opt_combine,"ids_combine":np.array(ids_combine),\
            "is_target_combine":is_target_combine,"Edges_keep_combine":Edges_keep_combine,"Edges_keep_bi_combine":Edges_keep_bi_combine,\
            "optimized_others_output":optimized_others_output}
    import pickle
    dict_dir=f"{work_dir}/opt_dicts"
    os.makedirs(dict_dir, exist_ok=True)
    file_name=f"opt_v8_{len(ids)}_0.pkl"

    # Save 2:
    # Save variables to a file
    dict_dir_graph=f"{work_dir}/graph_model/"
    os.makedirs(dict_dir_graph, exist_ok=True)
    file_path_graph=dict_dir_graph+file_name_graph

    torch.save({
                'dicts': None,\
                'ids': ids,\
                'Contribs': Contribs,\
                'Transls': Transls,\
                'Oris_xyzw': Oris_xyzw,\
                'Transls_optimizable': Transls_optimizable,\
                'Dists_optimizable': Dists_optimizable,\
                'weight_distance_optimizable':weight_distance_optimizable,\
                'Ori_distances_optimizable':Ori_distances_optimizable,\
                'Transls_optimized': Transls_optimized,\
                'Dists_optimized': Dists_optimized,\
                "weight_distance_optimized":weight_distance_optimized,\
                "Oris_xyzw_optimized":Oris_xyzw_optimized, \
                "Ori_distances_optimized":Ori_distances_optimized,\
                "edges_graph": edges_graph,\
                "edges_graph_optimized": edges_graph_optimized,\
                "Nodes_opt":Nodes_opt,\
                "Edges_keep":Edges_keep,\
                "optimized_key_output":optimized_key_output,\
                "device":device,\
                "work_dir":work_dir,\
                "ids2":ids2,\
                "Contribs2": Contribs2,\
                "Transls_optimized2":Transls_optimized2,\
                "Dists_optimized2":Dists_optimized2,\
                "weight_distance_optimized2":weight_distance_optimized2,\
                "Oris_xyzw_optimized2":Oris_xyzw_optimized2,\
                "Ori_distances_optimized2":Ori_distances_optimized2,\
                "edges_others_optimized":edges_others_optimized,\
                "edges_others":edges_others,\
                "blend_opt_dict":blend_opt_dict,\
                "contrib_threshold":contrib_threshold,\
                }\
                , file_path_graph)

    print("file_path_graph:",file_path_graph)


def main2(save_dir,voxel_size=0.4,n_min_effective_frames=5,ratio_key=-1):
    flag_remove_outlier=False
    flag_show_sample=False
    flag_show_threshold=False
    flag_ablation_select_pts_wo_uncertainty=False
    flag_ablation_random_select=False
    # ratio_key=0.08 # ablation study of threshold and selected number of pts at each step.
    
    folder_dict=f"{save_dir}/out_dicts3"

    nt=count_out_dict_files(folder_dict)
    # Stream pkl files to avoid loading all into RAM simultaneously (can be 8+ GB for long sequences)
    _Contribs_rows = []
    means_by_t = []
    for t in range(nt):
        with open(f"{folder_dict}/out_dict_{t}.pkl", 'rb') as file:
            d = pickle.load(file)
        _Contribs_rows.append(d['contribs_strict'].cpu().numpy())
        means_by_t.append(d['means'].cpu().numpy())
        del d
    Contribs = np.array(_Contribs_rows)
    del _Contribs_rows
    
    # Safety check for empty Contribs
    if Contribs.size == 0:
        print("Warning: Contribs array is empty, using contribs_in_mask as fallback")

    if ratio_key>0:
        max_num_key_nodes=int(Contribs.shape[1]*ratio_key)
        num_stage_wise_select_key_nodes=int(max_num_key_nodes/1.5)
        max_num_key_nodes = min(max_num_key_nodes, 1500) # avoid used memory is too large.
        max_num_key_nodes = max(max_num_key_nodes, 5) # avoid no of key nodes too few.
    else:
        max_num_key_nodes=1500 # preivous setting
        num_stage_wise_select_key_nodes=1000 # previous setting

    print("******ratio_key: ",ratio_key, "max_num_key_nodes: ",max_num_key_nodes," num_stage_wise_select_key_nodes: ",num_stage_wise_select_key_nodes)

    
    c99=np.quantile(Contribs, 0.99)
    c80=np.quantile(Contribs, 0.80)
    contrib_ref=Contribs[(Contribs<c99)&(Contribs>c80)].mean()
    if contrib_ref<0.5:
        prod_num=0.5/contrib_ref
        Contribs=Contribs*prod_num
        Contribs = np.clip(Contribs, a_min=None, a_max=10)

    if flag_remove_outlier:
        def get_outliers_by_density(means):
            X=means[np.arange(0,means.shape[0],1),:]

            clusterer = DBSCAN(eps=0.02, min_samples=10) # hyperparameter
            labels=clusterer.fit(X).labels_
            outlier_map=labels==-1

            return outlier_map
        
        outlier_map=get_outliers_by_density(dicts[t]['means'].cpu().numpy())
        outlier_map.sum()

        Contribs_up=[]
        for t in range(nt):
            contribs=Contribs[t]
            means=dicts[t]['means'].cpu().numpy()

            if flag_remove_outlier:
                outlier_map=get_outliers_by_density(means)
                contribs[outlier_map]=1e-10 ##
            Contribs_up.append(contribs)
        Contribs_up=np.array(Contribs_up)
    else:
        Contribs_up=None

    print("num of gaussians: ",means_by_t[0].shape[0])
    # 1st Gaussian selection round
    # select according to contribs
    selected_ids_ts=[]
    for t in range(nt):
        if not flag_remove_outlier:
            contribs=Contribs[t]
            means=means_by_t[t]
        else:
            contribs=Contribs_up[t]
        
        sorted_indices=np.argsort(contribs) # ascending order w.r.t. ids
        n=contribs.shape[0]
        # candidate_ids_map=sorted_indices>=n-1000
        candidates_in_raw_ids=sorted_indices[-num_stage_wise_select_key_nodes:] #1000
        # candidates_in_raw_ids=[i for i in range(n) if candidate_ids_map[i]]

        candidate_means=means[candidates_in_raw_ids]
        # candidate_contribs=contribs[candidate_ids_map]
        
        voxel_grid = np.floor(candidate_means / voxel_size)
        _, unique_indices = np.unique(voxel_grid, axis=0, return_index=True)
        selected_candidates_in_raw_ids=[candidates_in_raw_ids[i] for i in unique_indices]
        if flag_ablation_random_select:
            n_random_select=len(selected_candidates_in_raw_ids)
            selected_candidates_in_raw_ids=np.random.choice(selected_candidates_in_raw_ids, n_random_select, replace=False).tolist()
        selected_ids_ts.append(selected_candidates_in_raw_ids)
        # print(f"t={t}, selected_candidates_in_raw_ids.shape={len(selected_candidates_in_raw_ids)}")

    counts=np.zeros(n)
    for t in range(nt):
        selected_candidates_in_raw_ids=selected_ids_ts[t]
        counts[selected_candidates_in_raw_ids]+=1

    selected_ids_effective_1frame=[i for i in range(n) if counts[i]>0]
    print("#pts left after 1st selection round: ", len(selected_ids_effective_1frame))

    # avoid too many point in the 1st selection round
    # select pts
    def select_pts(n,selected_ids_effective_1frame):
        arr1=np.ones(n)
        arr2=np.zeros(len(selected_ids_effective_1frame)-n)
        arr=np.concatenate((arr1,arr2),axis=0)
        np.random.shuffle(arr)
        selected_ids_effective_1frame=np.array(selected_ids_effective_1frame)[arr.astype(bool)].tolist()
        return selected_ids_effective_1frame

    # 2nd Gaussian selection round
    print("n_min_effective_frames: ",n_min_effective_frames)
    counts_effective_frames=np.zeros(len(selected_ids_effective_1frame))
    thresholds=[]
    for t in range(nt):
        if t%100==0:
            print(t)
        if not flag_remove_outlier:
            contribs=Contribs[t]
        else:
            contribs=Contribs_up[t]

        sorted_indices=np.argsort(contribs)
        # candidates_in_raw_ids=sorted_indices[-2000:]
        threshold=contribs[sorted_indices[-num_stage_wise_select_key_nodes]] # the 2000th largest contrib # 1000
        threshold=max(threshold,0.5) # 0.5
        counts_effective_frames+=(contribs[selected_ids_effective_1frame]>=threshold).astype(int)
        thresholds.append(threshold)

    threshold_min=np.min(np.array(thresholds))
    threshold_mean=np.mean(np.array(thresholds))
    print("min contrib threshold: ",threshold_min)
    print("mean contrib threshold: ",threshold_mean)
    print(np.histogram(counts_effective_frames))
    if flag_show_threshold:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(thresholds)
        plt.show()
    selected_ids_eff=[selected_ids_effective_1frame[i] for i in range(len(selected_ids_effective_1frame)) if counts_effective_frames[i]>=n_min_effective_frames]
    print("#pts left after 2nd selection round: ", len(selected_ids_eff))
    # counts_effective_frames[counts_effective_frames>30].shape

    # avoid too many point in the 2nd selection round
    if len(selected_ids_eff)>max_num_key_nodes: 
        selected_ids_eff=select_pts(max_num_key_nodes,selected_ids_eff) 
        print("#pts left after 2nd selection round (adjusted): ", len(selected_ids_eff))
    
    if flag_ablation_select_pts_wo_uncertainty:
        selected_ids_eff=np.arange(Contribs.shape[1])
        selected_ids_eff=select_pts(1000,selected_ids_eff) 
        print(f"flag_ablation_select_pts_wo_uncertainty={flag_ablation_select_pts_wo_uncertainty}")
        print("random select key pts:", len(selected_ids_eff))

    if flag_show_sample:
        import matplotlib.pyplot as plt
        # t=nt-1
        t=0
        contribs=Contribs[t]
        means=means_by_t[t]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x = means[selected_ids_eff][:,0]
        y = means[selected_ids_eff][:,1]
        z = means[selected_ids_eff][:,2]
        sc=ax.scatter(x, y, z,s=1,alpha=0.3,c=contribs[selected_ids_eff],cmap='viridis')

        cbar = plt.colorbar(sc)
        cbar.set_label('Contribs')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.axis('equal')
        plt.show()

    # save
    file_path=f"{folder_dict}/effective_dict_{len(selected_ids_eff)}.pkl"
    effective_dict={"ids_eff2":selected_ids_eff,"ids_eff1":selected_ids_effective_1frame,"Contribs_up":Contribs_up,
                    "threshold_min":threshold_min,"threshold_mean":threshold_mean,"Contribs":Contribs}
    file=open(file_path, 'wb')
    pickle.dump(effective_dict, file)
    file.close()
    print(file_path)
    file_effective=f"effective_dict_{len(selected_ids_eff)}.pkl"
    return file_effective
    
def compute_knn_indices(points, k, chunk_size=1024):
    """
    Memory-efficient KNN search: returns indices of k nearest neighbors for each point.
    points: [N, 3]
    returns: [N, k] LongTensor
    """
    N = points.shape[0]
    knn_indices = []

    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        chunk = points[i:end]  # [chunk, 3]
        dists = torch.cdist(chunk, points)  # [chunk, N]
        topk = dists.topk(k + 1, largest=False).indices[:, 1:]  # exclude self
        knn_indices.append(topk)

    return torch.cat(knn_indices, dim=0)  # [N, k]

def gaussian_kde_density_knn(points, inv_covariances, k=1000, eps=1e-6):
    """
    Vectorized version using k nearest neighbors for KDE density.
    points: [N, 3]
    covariances: [N, 3, 3]
    returns: [N] density using k-NN Mahalanobis distances
    """
    N, D = points.shape
    device = points.device

    # Invert covariances: [N, 3, 3]
    # cov_inv = torch.inverse(covariances + eps * torch.eye(D, device=device))
    cov_inv = inv_covariances

    # Memory-efficient KNN
    chunk_size= 10000
    knn_indices = compute_knn_indices(points, k, chunk_size=chunk_size)  # [N, k]

    # Gather neighbor points: [N, k, 3]
    knn_points = points[knn_indices]  # batched gather

    # Center point per row: [N, 1, 3] -> broadcast to [N, k, 3]
    diffs = knn_points - points[:, None, :]  # [N, k, 3]

    # Apply Mahalanobis: dᵢⱼ = (xᵢ - xⱼ)^T Σᵢ⁻¹ (xᵢ - xⱼ)
    # cov_inv: [N, 3, 3], diffs: [N, k, 3]
    # compute: diffᵢⱼ^T @ cov_invᵢ @ diffᵢⱼ -> result [N, k]

    diffs_unsq = diffs.unsqueeze(-1)              # [N, k, 3, 1]
    cov_inv_exp = cov_inv[:, None, :, :]          # [N, 1, 3, 3]
    m = torch.matmul(cov_inv_exp, diffs_unsq)     # [N, k, 3, 1]
    mahalanobis_sq = torch.matmul(diffs_unsq.transpose(-1, -2), m).squeeze(-1).squeeze(-1)  # [N, k]

    # Gaussian kernel weights
    weights = torch.exp(-0.5 * mahalanobis_sq)  # [N, k]

    density = weights.sum(dim=1)  # [N]
    return density


def plot_density_histogram_rgb(density, bins=20, height=256):
    """
    density: [N] tensor
    height: desired output image height in pixels
    Returns: [height, width, 3] RGB uint8 numpy image of the histogram with log-scale x-axis and labeled axes
    """
    density_np = density.cpu().numpy()
    density_np = density_np[density_np > 0]  # filter out non-positive for log scale

    dpi = 100  # Dots per inch
    fig_height_inches = height / dpi
    aspect_ratio = 1.5  # width / height
    fig_width_inches = fig_height_inches * aspect_ratio

    # Create figure with desired size
    fig, ax = plt.subplots(figsize=(fig_width_inches, fig_height_inches), dpi=dpi)
    ax.hist(density_np, bins=bins, color='skyblue', edgecolor='black')
    ax.set_xscale("log")
    ax.set_title("Density Histogram (Log X-axis)")
    ax.set_xlabel("Density (log scale)")
    ax.set_ylabel("Count")

    fig.tight_layout()
    fig.canvas.draw()

    width, height = fig.canvas.get_width_height()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)

    plt.close(fig)
    return image.astype(np.uint8)

def load_gt_masks(ws, gt_mask_dirname="mask"):
    import imageio.v2 as imageio
    """Load ground truth masks from directory"""
    gt_mask_dir = osp.join(ws, gt_mask_dirname)
    if not osp.exists(gt_mask_dir):
        logging.warning(f"GT mask directory {gt_mask_dir} not found")
        breakpoint()
        return None
    
    gt_mask_fns = [
        f for f in os.listdir(gt_mask_dir)
        if f.endswith(".png") or f.endswith(".jpg")
    ]
    gt_mask_fns.sort()
    
    gt_masks = []
    for gt_mask_fn in gt_mask_fns:
        gt_mask_path = osp.join(gt_mask_dir, gt_mask_fn)
        gt_mask = imageio.imread(gt_mask_path)
        if gt_mask.ndim == 3:
            gt_mask = gt_mask[..., 0]  # Take first channel if RGB
        gt_mask = (gt_mask > 128).astype(np.float32)  # Convert to binary
        gt_masks.append(gt_mask)
    
    gt_masks = torch.Tensor(np.stack(gt_masks)).float()  # T,H,W
    # self.register_gradfree_buffer("gt_masks", gt_masks)
    # self.has_gt_masks = True
    logging.info(f"Loaded GT masks from {gt_mask_dir}: {gt_masks.shape}, foreground ratio: {gt_masks.mean():.3f}")
    return gt_masks, gt_mask_fns

# @torch.inference_mode()
def main1(dataset_name,work_dir,save_dir,ws,relative_dir_saved_mosca_model=None, thr_rgb_diff=0.3):
    import torch
    import argparse
    import os
    import torch
    import time
    from lib_usplat4d_viewer.usplat4d_renderer import Renderer
    from lib_render.render_helper import render, render_gsplat
    import numpy as np
    from tqdm import tqdm
    import imageio.v3 as iio

    from datetime import datetime
    from loguru import logger as guru
    import torch.nn.functional as F
    import cv2
    import matplotlib.pyplot as plt
    import lpips

    from lib_ugraph.ugraph_utils import compute_ssim_map, compute_lpips_map

    

    MOSCA_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))  # USplat4d_vMoSca/
    MOSCA_PROFILE = osp.join(osp.dirname(MOSCA_ROOT), 'MoSca_mask', 'profile')
    if dataset_name=='iphone':
        cfg_fn=osp.join(MOSCA_PROFILE, 'iphone/iphone_fit.yaml')
    elif dataset_name=='davis' or dataset_name=='davis_mask':
        cfg_fn=osp.join(MOSCA_PROFILE, 'demo/demo_fit.yaml')
    elif dataset_name=='nvidia':
        cfg_fn=osp.join(MOSCA_PROFILE, 'nvidia/nvidia_fit.yaml')
    elif dataset_name=='diffusion4d':
        cfg_fn=osp.join(MOSCA_PROFILE, 'demo/demo_fit.yaml')
    elif dataset_name=='objaverse':
        cfg_fn=osp.join(MOSCA_PROFILE, 'objaverse/objaverse_fit.yaml')
    elif dataset_name=='hypernerfVrig':
        cfg_fn=osp.join(MOSCA_PROFILE, 'hypernerfVrig/hypernerfVrig_fit.yaml')
    else:
        cfg_fn=''
    port=9000

    device = torch.device("cuda") # if torch.cuda.is_available() else "cpu")
    renderer = Renderer(cfg_fn, device, work_dir, port=port, bg_flag=True, fg_flag=True,load_s2d=True, ws=ws, use_ugraph=False, relative_dir_saved_mosca_model=relative_dir_saved_mosca_model)
    
    nt=renderer.cams.T
    print("nt:",nt)

    W, H = renderer.cams.default_W, renderer.cams.default_H
    assert W == renderer.s2d.W and H == renderer.s2d.H
    K = renderer.cams.K(H=H,W=W)
    ts=torch.arange(nt,device=device)
    w2cs=torch.stack([renderer.cams.T_cw(t) for t in ts],dim=0)

    if save_dir=='':
        save_dir = work_dir

    video_dir = f"{save_dir}/videos/{datetime.now().strftime('%Y-%m-%d-%H%M%S')}"
    print("*"*20)
    print("*"*20)
    print(f"Saving video to {video_dir}")
    print("*"*20)
    print("*"*20)
    os.makedirs(video_dir, exist_ok=True)
    
    torch.backends.cuda.preferred_linalg_library("magma")


    if dataset_name=="davis" or dataset_name=="iphone" or dataset_name=="diffusion4d" or dataset_name=="objaverse" or dataset_name=="davis_mask" or dataset_name=="hypernerfVrig":
        thr_density_strict=0.1 #5
        thr_density_loose=0.1 #1
        # thr_rgb_diff=0.3 # from input
        thr_ssim=0.01
    elif dataset_name=="nvidia":
        thr_density_strict=0.0
        thr_density_loose=0.0
        thr_rgb_diff=0.6#0.6 # OVERWRITE
        thr_ssim=0.001 # 0.001
    else:
        breakpoint()
    
    try:
        gt_masks=renderer.s2d.gt_masks.to(device)
        gt_masks=gt_masks.to(dtype=torch.bool)
    except:
        gt_masks=torch.ones((nt,H,W),device=device,dtype=torch.bool)
    prior_depths=renderer.s2d.dep.to(device) # [nt,H,W]
    prior_depths_grad, _, _=compute_depth_gradient(prior_depths) # [nt,H,W]
    depths_info = 1e-3 * torch.ones_like(prior_depths_grad,device=device) # 1e-3 1e-5
    print("np.histogram:",np.histogram(prior_depths_grad[gt_masks].flatten().cpu().numpy(), bins=10))
    depth_min = prior_depths.min().item()
    def safe_quantile(tensor: torch.Tensor, q: float, max_samples: int = 100000):
        """Compute quantile safely by subsampling if needed."""
        if tensor.numel() > max_samples:
            indices = torch.randperm(tensor.numel(), device=tensor.device)[:max_samples]
            tensor = tensor.view(-1)[indices]
        return tensor.quantile(q)
    depth_max = safe_quantile(prior_depths, 0.99).item()
    prior_depths_grad_threshold = safe_quantile(prior_depths_grad[gt_masks], 0.90).item()
    print(f"set prior_depths_grad_threshold to {prior_depths_grad_threshold}.")
    print("fg Gaussian num:", renderer.d_model.N)
    print("bg Gaussian num:", renderer.s_model.N)
    print("generate out_dicts_t.pkl:")
    for i, (w2c, t) in enumerate(zip(tqdm(w2cs), ts)):
        assert i == t.item()
        with torch.inference_mode():
            gs5 = []
            if renderer.bg_flag:
                gs5.append(renderer.s_model())
            if renderer.fg_flag:
                d_list_t=renderer.d_model(t)
                gs5.append(d_list_t)
            n_fg = renderer.d_model(t)[0].shape[0]
            render_dict, out_dict = render_gsplat(
                gs5,
                H,
                W,
                K=K,
                T_cw=w2c,
                bg_color=[1.0, 1.0, 1.0],
                n_fg=n_fg,
            )
            n_all=len(gs5)

            # density filter begin
            mu, frame, scale, o, sph=d_list_t
            inv_actual_covariance = frame @ torch.diag_embed(1.0 / (scale ** 2)) @ frame.transpose(1, 2)

            density=gaussian_kde_density_knn(mu,inv_actual_covariance, k=100) # TODO: check scale factor
            large_density_Gaussian_strict=density>thr_density_strict # TODO: threshold # 5
            large_density_Gaussian_loose=density>thr_density_loose # TODO: threshold # 1
            # density filter end
            out_dict['w2c']=w2c.detach()
            out_dict['K']=K
            gt_mask = gt_masks[i]
            projected = project_points(out_dict['means'], K, w2c) # [N,2]
            in_mask_Gaussians=valid_and_visible(projected, gt_mask) # [N] bool
            out_dict['in_mask_Gaussians']=in_mask_Gaussians
            contribs=out_dict['contribs'].reshape(-1,3)[:,0].clone()
            
            img = out_dict["img"][0] # [H,W,3]
            gt_img = renderer.s2d.rgb[t].to(device) # [H,W,3]
            diff_img = F.l1_loss(img, gt_img, reduction="none").mean(dim=-1)  # [H,W]
            mask_diff_img = diff_img < thr_rgb_diff # [H,W] # TODO: threshold #0.1
            small_img_diff_Gaussians=valid_and_visible(projected, mask_diff_img) # [N] bool
            # ssim map begin
            ssim_map, ssim_loss=compute_ssim_map(img.cpu().numpy(),gt_img.cpu().numpy()) # [H,W,3]
            mask_ssim_img = torch.tensor(np.array(ssim_map > thr_ssim),device=device).all(dim=-1) # [H,W] # TODO
            large_ssim_Gaussians=valid_and_visible(projected, mask_ssim_img) # [N] bool
            # ssim map end
            # depth gradient mask
            mask_depth_grad_img = prior_depths_grad[i] < prior_depths_grad_threshold # [H,W]
            small_depth_grad_Gaussians=valid_and_visible(projected, mask_depth_grad_img) # [N] bool

            #  
            valid_gaussians_strict = small_depth_grad_Gaussians & large_ssim_Gaussians & small_img_diff_Gaussians & in_mask_Gaussians # & large_density_Gaussian_strict  #& small_lpips_Gaussians # & in_mask_Gaussians # 
            valid_gaussians_loose =  large_ssim_Gaussians & small_img_diff_Gaussians # & large_density_Gaussian_loose #& small_lpips_Gaussians # 
            contribs_in_mask=contribs.clone()
            contribs_in_mask[~in_mask_Gaussians]=1e-8 # out-mask Gaussians are set to small contribs
            contribs_strict=contribs.clone()
            contribs_loose=contribs.clone()
            contribs_strict[~valid_gaussians_strict]=1e-8
            contribs_loose[~valid_gaussians_loose]=1e-8

            out_dict['contribs_in_mask']=contribs_in_mask
            out_dict['contribs_strict']=contribs_strict
            out_dict['contribs_loose']=contribs_loose

            # depth uncertainty
            depths_info_perG=associate_pts_value(projected, depths_info[i]) # per Gaussian [N]
            out_dict['depths_info_perG']=depths_info_perG
            if True:
                import pickle
                dict_dir=f"{save_dir}/out_dicts3"
                os.makedirs(dict_dir, exist_ok=True)
                file=open(f'{dict_dir}/out_dict_{t}.pkl', 'wb')
                pickle.dump(out_dict, file)
                file.close()
            
            if True:
                def to_rgb(img: np.ndarray) -> np.ndarray:
                    # [H, W] or [H, W, 1] -> [H, W, 3]
                    if img.ndim == 2:
                        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    elif img.ndim == 3 and img.shape[2] == 1:
                        return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
                    return img  # Already RGB or unsupported format
                
                img_np = (img.cpu().numpy() * 255.0).astype(np.uint8)
                gt_img_np = (gt_img.cpu().numpy() * 255.0).astype(np.uint8)
                ssim_map_np = (np.array(ssim_map) * 255.0).clip(0, 255).astype(np.uint8)
                # lpips_map_np = to_rgb((np.array(lpips_map) * 255.0).clip(0, 255).astype(np.uint8))

                img_with_points = draw_projected_points_masked(img_np, projected, valid_gaussians_strict) # valid_gaussians, in_mask_Gaussians
                img_with_points_loose = draw_projected_points_masked(img_np, projected, valid_gaussians_loose)
                masked_img = overlay_mask_yellow(img_np, gt_mask.cpu().numpy())
                masked_gt_img = overlay_mask_yellow(gt_img_np, gt_mask.cpu().numpy())
                masked_diff_gt_img = overlay_mask_yellow(gt_img_np, mask_diff_img.cpu().numpy(), color= (255,0,0))
                
                diff_img_np = ((diff_img*5).clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8) # *10 to make it clear
                diff_img_np = to_rgb(diff_img_np)
                img_with_points_depth=draw_projected_points_colored(img_np, projected, depths_info_perG)
                depth_img=apply_depth_colormap(
                        prior_depths[i].unsqueeze(-1), near_plane=depth_min, far_plane=depth_max
                    )
                depth_img_np = (depth_img.clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
                prior_depths_grad_img=apply_depth_colormap(
                        prior_depths_grad[i].unsqueeze(-1), near_plane=depth_min, far_plane=depth_max
                    )
                prior_depths_grad_img_np = (prior_depths_grad_img.clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)

                combined = np.hstack([img_with_points, img_with_points_loose, depth_img_np, prior_depths_grad_img_np, ssim_map_np, masked_img, masked_gt_img, masked_diff_gt_img, diff_img_np, img_with_points_depth])
                iio.imwrite(f"{video_dir}/img_{i}.png", combined)
            
    # Delete large objects manually
    del renderer
    del render_dict
    del inv_actual_covariance
    # del lpips_model
    del prior_depths
    del prior_depths_grad
    out_dict = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in out_dict.items()}
    del out_dict
    del img, gt_img, diff_img, mask_diff_img
    del projected, depths_info_perG
    del contribs_strict, contribs_loose
    del ssim_map, ssim_loss#, lpips_map, lpips_loss
    

    # Optionally clear lists
    gs5.clear()

    # Force garbage collection
    import gc
    gc.collect()

    # Clear PyTorch CUDA cache
    torch.cuda.empty_cache()

def compare_images(img, img0, tolerance=1):
    if img.shape != img0.shape:
        print(f"Shape mismatch: {img.shape} vs {img0.shape}")
        return False

    diff = np.abs(img.astype(np.int32) - img0.astype(np.int32))
    max_diff = diff.max()

    if max_diff <= tolerance:
        print(f"Images are close. Max diff: {max_diff}")
        return True
    else:
        print(f"Images differ. Max diff: {max_diff}")
        return False
    
def backup_code(save_dir):
    from datetime import datetime
    backup_dir = osp.join(save_dir, "src_backup")
    os.makedirs(backup_dir, exist_ok=True)
    for path in [
        "run_preprocessing.sh",
        "usplat4d_prep.py",
    ]:
        os.system(f"cp -r {path} {backup_dir}")
    with open(osp.join(save_dir, f"commandline_args{datetime.now().strftime('%Y-%m-%d-%H%M%S')}.txt"), "w") as f:
        f.write(" ".join(sys.argv))

def run_from_outside():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_name", type=str, required=True)
    parser.add_argument("--func_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--depth_ratio", type=float, help="default=0.1 (iphone) or 0.001 (backpack or davis)", required=True) # default=0.1, 
    parser.add_argument("--threshold_min_set", type=float, help="default=0.5 or 0.1 (paper-windmill)", required=True) # default=0.5, 
    parser.add_argument("--version_key_edges", type=str, help="version_key_edges: max_contrib or minmax_distance_contrib", required=True) # default="max_contrib"
    parser.add_argument("--extra_save_str", type=str, help="extra_save_str", required=False, default="") 
    parser.add_argument("--ratio_key", type=float, help="ratio of key nodes to determine the selected number of nodes in the first stage and threshold in the 2nd stage", required=False, default=-1)
    parser.add_argument("--which_logs_folder", type=str, help="which_logs_folder", required=False, default="logs") # default="logs"
    parser.add_argument("--log_subfolder", type=str, help="log_subfolder", required=False, default="demo_fit_native_add3") # default="demo_fit_native_add3"
    parser.add_argument("--relative_dir_saved_mosca_model", type=str, help="relative_dir_saved_mosca_model. e.g.,", required=False)
    parser.add_argument("--n_min_effective_frames", type=int, help="n_min_effective_frames", required=False, default=5)
    parser.add_argument("--thr_rgb_diff", type=float, help="thr_rgb_diff", required=False, default=0.3)
    parser.add_argument("--ini_folder_path", type=str, help="root folder containing the trained MoSca model per sequence (e.g. /data/dataset/iphone_mosca_trained)", required=True)
    parser.add_argument("--usplat4d_folder_path", type=str, help="root folder for saving the USplat4D graph model per sequence (e.g. /data/dataset/iphone_mosca_graph_model)", required=True)
    args = parser.parse_args()

    seq_name=args.seq_name
    dataset_name=args.dataset_name
    depth_ratio=args.depth_ratio # 0.001
    threshold_min_set=args.threshold_min_set # 0.5 # paper-windmill 0.1
    version_key_edges=args.version_key_edges
    extra_save_str=args.extra_save_str
    ratio_key=args.ratio_key
    save_sub_folder=f'dr{depth_ratio}_thr{threshold_min_set}_v{version_key_edges}{extra_save_str}/' 
    logs_folder=args.which_logs_folder
    relative_dir_saved_mosca_model=args.relative_dir_saved_mosca_model
    log_subfolder=args.log_subfolder
    n_min_effective_frames=args.n_min_effective_frames
    thr_rgb_diff=args.thr_rgb_diff
    ini_folder_path=args.ini_folder_path
    usplat4d_folder_path=args.usplat4d_folder_path
    if dataset_name=="iphone":
        work_dir=f"{ini_folder_path}/{seq_name}/{logs_folder}/iphone_fit_native_add3"
        save_dir=f"{usplat4d_folder_path}/{seq_name}/"+save_sub_folder
        ws=f"{ini_folder_path}/{seq_name}/"
        voxel_size=0.2
    elif dataset_name=="davis_mask":
        work_dir=f"{ini_folder_path}/{seq_name}/{logs_folder}/demo_fit_native_add3"
        save_dir=f"{usplat4d_folder_path}/{seq_name}/"+save_sub_folder
        ws=f"{ini_folder_path}/{seq_name}/"
        voxel_size=0.05
    else:
        raise ValueError("dataset_name not supported")

    backup_code(save_dir)

    file_name_graph=f"v12_parameters9mee.pth"

    if args.func_name=="main1":
        main1(dataset_name,work_dir,save_dir,ws,relative_dir_saved_mosca_model,thr_rgb_diff)
    elif args.func_name=="main23":
        print('='*20)
        print(f"n_min_effective_frames: {n_min_effective_frames}")
        print('='*20)
        if args.depth_ratio is None or args.threshold_min_set is None or args.version_key_edges is None:
            parser.error("--depth_ratio, --threshold_min_set, and --version_key_edges are required when func_name is 'main23'")
        file_effective=main2(save_dir,voxel_size=voxel_size,n_min_effective_frames=n_min_effective_frames,ratio_key=ratio_key)
        main3(save_dir,file_effective,file_name_graph,depth_ratio,version_key_edges=version_key_edges,threshold_min_set=threshold_min_set) #"max_contrib" minmax_distance_contrib



if __name__ == "__main__":
    run_from_outside()