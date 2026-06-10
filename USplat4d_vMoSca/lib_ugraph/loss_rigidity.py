import torch
import numpy as np

import pypose as pp
from torch import nn


from ugraph_helper import Precompute_helper
import roma
import torch.nn.functional as F



def get_rigidity_loss(precompute: Precompute_helper, ts: torch.Tensor):
    # ts : [batch of time frames]
    # currently, we calculate all the loss for all the frames
    transls=precompute.params['transls'] # transls_som
    oris_wxyz=precompute.params['oris_wxyz'] # oris_wxyz_som

    edges_local=precompute.graph_para['edges_local'] # all
    sq_dists_local=precompute.graph_para['sq_dists_local'] # all
    weight_rigidity=precompute.graph_para['weight_rigidity'] # all

    n=transls.shape[1]
    nt=transls.shape[0]
    device=transls.device
    edges_local=edges_local.to(device)
    sq_dists_local=sq_dists_local.to(device)
    weight_rigidity=weight_rigidity.to(device)
    
    # stats={}


    oris_wxyz = F.normalize(oris_wxyz, p=2, dim=-1)
    # oris_xyzw=roma.quat_xyzw_to_wxyz(oris_wxyz) # [BUG]
    oris_xyzw=roma.quat_wxyz_to_xyzw(oris_wxyz)

    assert torch.norm(oris_xyzw[3,5],p=2)-1<1e-6

    # 1. get knn locals and weights
    #

    # 2. compute local ridigity loss for transls
    # 3. compute local ridigity loss for oris
    # 4. compute global rigidity loss
    # weights.shape # [num_nodes, num_neighbors]
    # sq_dists_local.shape # [num_nodes, num_neighbors]
    weight_edges=weight_rigidity.reshape(-1) # [num_edges]
    dis_local_canonical=sq_dists_local**0.5 # [num_nodes, n_neighbors]
    dis_local_canonical_edge=dis_local_canonical.reshape(-1) # [num_edges] 
    loss,stats=cal_rigidity_loss(transls,oris_xyzw,edges_local,weight_edges,dis_local_canonical_edge,"local/")

    # stats = {**stats_key, **stats_nonkey}
    
    return loss, stats



def cal_rigidity_loss(transls,oris_xyzw,edges_local,weight_edges,dis_local_canonical_edge,label=""):
    # transls.shape # [nt, num_nodes, 3]
    # oris_xyzw.shape # [nt, num_nodes, 4]
    # edges_local.shape # [num_edges, 2] # [...[center id, neighbor id]...]
    # weight_edges.shape # [num_edges]
    # dis_local_canonical_edge.shape # [num_edges]
    is_subsample=True # For block.... # False for ablation study
    if is_subsample:
        # Subsample edges where weight_edges > threshold
        # threshold = 1e-4 # 1e-4 # 1e-2 # 1e-3 # 1e-4 for ablation study
        threshold = torch.quantile(weight_edges,0.5)
        valid_mask = weight_edges > threshold
        valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)

        # Subsample from valid edges only
        # num_sample_edges = min(50000, valid_indices.shape[0])
        num_sample_edges = int(0.2*valid_indices.shape[0]) # 0.8
        sample_indices = valid_indices[torch.randperm(valid_indices.shape[0], device=valid_indices.device)[:num_sample_edges]]

        # Apply sampling
        edges_local = edges_local[sample_indices]
        weight_edges = weight_edges[sample_indices]
        dis_local_canonical_edge = dis_local_canonical_edge[sample_indices]

    # 1. compute local rigidity loss for transls
    # weight_edges=weights.reshape(-1) # [num_edges]
    transls_diff_edges=transls[:,edges_local[:,1]]-transls[:,edges_local[:,0]] # [nt, num_edges, 3]
    # oris_xyzw_i=oris_xyzw[:,edges_local[:,0]] # [:,num_edges, 4]

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
    # dis_local_canonical=sq_dists_local**0.5 # [num_nodes, n_neighbors]
    # dis_local_canonical_edge=dis_local_canonical.reshape(-1) # [num_edges] # 
    dis_local_edge=torch.norm(transls_diff_edges,dim=-1) # [nt, num_edges]
    dis_local_diff=torch.abs(dis_local_canonical_edge[None,:]-dis_local_edge) # [nt, num_edge]
    error_local_global=dis_local_diff*weight_edges[None,:] # [nt, num_edge]
    loss_local_global=error_local_global.mean() # scalar

    loss = loss_local_rigidity+loss_local_rigidity_ori+loss_local_global
    stats = {
        label+"loss_local_rigidity": loss_local_rigidity,
        label+"loss_local_rigidity_ori": loss_local_rigidity_ori,
        label+"loss_local_global": loss_local_global,
    }
    
    return loss, stats

def cal_dt_loss(delta_t,oris_xyzw,edges_local,weight_edges,transls_diff_edges):
    oris_xyzw_tdiff_dt=roma.quat_product(oris_xyzw[:-delta_t],roma.quat_conjugation(oris_xyzw[delta_t:])) # [nt-1, num_nodes, 4] # R{i,t-1} * R{i,t}^-1 # Conjugation of a unit quaternion is equal to its inverse. # quat_inverse(quat)
    oris_xyzw_tdiff_edges_dt=oris_xyzw_tdiff_dt[:,edges_local[:,0]] # [nt-1, num_edges, 4]
    transls_diff_cal_edges_dt=roma.utils.quat_action(oris_xyzw_tdiff_edges_dt,transls_diff_edges[delta_t:],is_normalized=True) # [nt-1, num_edges, 3] # = R*t
    transls_diff_current_dt=transls_diff_edges[:-delta_t] # [nt-1, num_edges, 3]
    dis_diff_dt=torch.norm(transls_diff_cal_edges_dt-transls_diff_current_dt,dim=-1) # [nt-1, num_edges]
    error_local_rigidity_dt=dis_diff_dt*weight_edges[None,:] # [nt-1, num_edges]
    loss_local_rigidity_dt=error_local_rigidity_dt.mean() # scalar

    # # 2. optionally, compute local rigidity loss for oris
    # qq_j_edges_dt=oris_xyzw_tdiff_dt[:,edges_local[:,1]] # [nt-1, num_edges, 4]
    # qq_i_edges_dt=oris_xyzw_tdiff_dt[:,edges_local[:,0]]
    # qq_diff_dt=torch.norm(qq_j_edges_dt-qq_i_edges_dt,dim=-1) # [nt-1, num_edges] #
    # error_local_rigidity_ori_dt=qq_diff_dt*weight_edges[None,:] # [nt-1, num_edges]
    # loss_local_rigidity_ori_dt=error_local_rigidity_ori_dt.mean() # scalar
    loss_local_rigidity_ori_dt=None

    if np.isnan([loss_local_rigidity_dt.item()]).any():
        print("loss_local_rigidity_dt.item(): ", loss_local_rigidity_dt.item())
        pass  # breakpoint() disabled for background runs

    return loss_local_rigidity_dt,loss_local_rigidity_ori_dt

def cal_rigidity_loss_extra(transls,oris_xyzw,edges_local,weight_edges,dis_local_canonical_edge,label=""):
    # transls.shape # [nt, num_nodes, 3]
    # oris_xyzw.shape # [nt, num_nodes, 4]
    # edges_local.shape # [num_edges, 2] # [...[center id, neighbor id]...]
    # weight_edges.shape # [num_edges]
    # dis_local_canonical_edge.shape # [num_edges]
    is_subsample=True # For block.... # False for ablation study
    if is_subsample:
        # Subsample edges where weight_edges > threshold
        threshold = 1e-4 # 1e-2 # 1e-3 # 1e-4 for ablation study
        valid_mask = weight_edges > threshold
        valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)

        # Subsample from valid edges only
        # num_sample_edges = min(50000, valid_indices.shape[0])
        num_sample_edges = int(0.8*valid_indices.shape[0]) # 0.8
        sample_indices = valid_indices[torch.randperm(valid_indices.shape[0], device=valid_indices.device)[:num_sample_edges]]

        # Apply sampling
        edges_local = edges_local[sample_indices]
        weight_edges = weight_edges[sample_indices]
        dis_local_canonical_edge = dis_local_canonical_edge[sample_indices]

    nt=transls.shape[0]

    # 1. compute local rigidity loss for transls
    # weight_edges=weights.reshape(-1) # [num_edges]
    transls_diff_edges=transls[:,edges_local[:,1]]-transls[:,edges_local[:,0]] # [nt, num_edges, 3]
    # oris_xyzw_i=oris_xyzw[:,edges_local[:,0]] # [:,num_edges, 4]

    # t, t-1
    oris_xyzw_tdiff=roma.quat_product(oris_xyzw[:-1],roma.quat_conjugation(oris_xyzw[1:])) # [nt-1, num_nodes, 4] # R{i,t-1} * R{i,t}^-1 # Conjugation of a unit quaternion is equal to its inverse. # quat_inverse(quat)
    oris_xyzw_tdiff_edges=oris_xyzw_tdiff[:,edges_local[:,0]] # [nt-1, num_edges, 4]
    transls_diff_cal_edges=roma.utils.quat_action(oris_xyzw_tdiff_edges,transls_diff_edges[1:],is_normalized=True) # [nt-1, num_edges, 3] # = R*t
    transls_diff_current=transls_diff_edges[:-1] # [nt-1, num_edges, 3]
    dis_diff=torch.norm(transls_diff_cal_edges-transls_diff_current,dim=-1) # [nt-1, num_edges]
    error_local_rigidity=dis_diff*weight_edges[None,:] # [nt-1, num_edges]
    loss_local_rigidity=error_local_rigidity.mean() # scalar

    # 2. compute local rigidity loss for oris
    qq_j_edges=oris_xyzw_tdiff[:,edges_local[:,1]] # [nt-1, num_edges, 4]
    qq_i_edges=oris_xyzw_tdiff[:,edges_local[:,0]]
    qq_diff=torch.norm(qq_j_edges-qq_i_edges,dim=-1) # [nt-1, num_edges] #
    error_local_rigidity_ori=qq_diff*weight_edges[None,:] # [nt-1, num_edges]
    loss_local_rigidity_ori=error_local_rigidity_ori.mean() # scalar

    # 3. compute global rigidity loss
    # dis_local_canonical=sq_dists_local**0.5 # [num_nodes, n_neighbors]
    # dis_local_canonical_edge=dis_local_canonical.reshape(-1) # [num_edges] # 
    dis_local_edge=torch.norm(transls_diff_edges,dim=-1) # [nt, num_edges]
    dis_local_diff=torch.abs(dis_local_canonical_edge[None,:]-dis_local_edge) # [nt, num_edge]
    error_local_global=dis_local_diff*weight_edges[None,:] # [nt, num_edge]
    loss_local_global=error_local_global.mean() # scalar

    # 4. compute dt rigidity loss according to the length of the sequence
    if nt<100:
        delta_t=int(nt/2)
    else:
        delta_t=50
    
    if nt>40:
        loss_local_rigidity_50,loss_local_rigidity_ori_50=cal_dt_loss(delta_t,oris_xyzw,edges_local,weight_edges,transls_diff_edges)
    else:
        loss_local_rigidity_50=torch.tensor(0.0)
        loss_local_rigidity_ori_50=torch.tensor(0.0)
    
    if nt>20:
        loss_local_rigidity_10,loss_local_rigidity_ori_10=cal_dt_loss(10,oris_xyzw,edges_local,weight_edges,transls_diff_edges) # we expected nt > 10
    else:
        loss_local_rigidity_10=torch.tensor(0.0)
        loss_local_rigidity_ori_10=torch.tensor(0.0)


    loss = loss_local_rigidity+loss_local_rigidity_ori+100*loss_local_global+100*loss_local_rigidity_50+100*loss_local_rigidity_10#+100*loss_local_rigidity_ori_dt
    stats = {
        label+"loss_local_rigidity": loss_local_rigidity.item(),
        label+"loss_local_rigidity_ori": loss_local_rigidity_ori.item(),
        label+"loss_local_global": loss_local_global.item(),
        label+"loss_local_rigidity_50": loss_local_rigidity_50.item(),
        label+"loss_local_rigidity_10": loss_local_rigidity_10.item(),
    }
    
    if np.isnan(list(stats.values())).any():
        print("Warning: NaN detected in stats!", stats)
        pass  # breakpoint() disabled for background runs
    
    return loss, stats

def cal_rigidity_loss2(transls,oris_xyzw,transls2,oris_xyzw2,edges_local,weight_edges,dis_local_canonical_edge,label=""):
    # transls.shape # [nt, num_nodes, 3]
    # oris_xyzw.shape # [nt, num_nodes, 4]
    # transls2.shape # [nt, num_nodes, 3]
    # oris_xyzw2.shape # [nt, num_nodes, 4]
    # edges_local.shape # [num_edges, 2] # [...[center id, neighbor2 id]...]
    # weight_edges.shape # [num_edges]
    # dis_local_canonical_edge.shape # [num_edges]
    

    # 1. compute local rigidity loss for transls
    # weight_edges=weights.reshape(-1) # [num_edges]
    transls_diff_edges=transls2[:,edges_local[:,1]]-transls[:,edges_local[:,0]] # [nt, num_edges, 3]
    # oris_xyzw_i=oris_xyzw[:,edges_local[:,0]] # [:,num_edges, 4]

    # t, t-1
    oris_xyzw_tdiff=roma.quat_product(oris_xyzw[:-1],roma.quat_conjugation(oris_xyzw[1:])) # [nt-1, num_nodes, 4] # R{i,t-1} * R{i,t}^-1 # Conjugation of a unit quaternion is equal to its inverse. # quat_inverse(quat)
    oris_xyzw_tdiff_edges=oris_xyzw_tdiff[:,edges_local[:,0]] # [nt-1, num_edges, 4]
    transls_diff_cal_edges=roma.utils.quat_action(oris_xyzw_tdiff_edges,transls_diff_edges[1:]) # [nt-1, num_edges, 3] # = R*t
    transls_diff_current=transls_diff_edges[:-1] # [nt-1, num_edges, 3]
    dis_diff=torch.norm(transls_diff_cal_edges-transls_diff_current,dim=-1) # [nt-1, num_edges]
    error_local_rigidity=dis_diff*weight_edges[None,:] # [nt-1, num_edges]
    loss_local_rigidity=error_local_rigidity.mean() # scalar

    # 2. compute local rigidity loss for oris
    oris_xyzw2_tdiff=roma.quat_product(oris_xyzw2[:-1],roma.quat_conjugation(oris_xyzw2[1:])) # [nt-1, num_nodes, 4] # R{i,t-1} * R{i,t}^-1 # Conjugation of a unit quaternion is equal to its inverse. # quat_inverse(quat)
    qq_j_edges=oris_xyzw2_tdiff[:,edges_local[:,1]] # [nt-1, num_edges, 4]
    qq_i_edges=oris_xyzw_tdiff[:,edges_local[:,0]]
    qq_diff=torch.norm(qq_j_edges-qq_i_edges,dim=-1) # [nt-1, num_edges] #
    error_local_rigidity_ori=qq_diff*weight_edges[None,:] # [nt-1, num_edges]
    loss_local_rigidity_ori=error_local_rigidity_ori.mean() # scalar

    # 3. compute global rigidity loss
    # dis_local_canonical=sq_dists_local**0.5 # [num_nodes, n_neighbors]
    # dis_local_canonical_edge=dis_local_canonical.reshape(-1) # [num_edges] # 
    dis_local_edge=torch.norm(transls_diff_edges,dim=-1) # [nt, num_edges]
    dis_local_diff=torch.abs(dis_local_canonical_edge[None,:]-dis_local_edge) # [nt, num_edge]
    error_local_global=dis_local_diff*weight_edges[None,:] # [nt, num_edge]
    loss_local_global=error_local_global.mean() # scalar

    loss = loss_local_rigidity+loss_local_rigidity_ori+loss_local_global
    stats = {
        label+"loss_local_rigidity": loss_local_rigidity,
        label+"loss_local_rigidity_ori": loss_local_rigidity_ori,
        label+"loss_local_global": loss_local_global,
    }
    
    return loss, stats
