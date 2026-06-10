# Mar 20, 2025
# compared with graph_model2.py, this file has the following changes:
# 1. use DQB inteorpolation for nonkey points to have a better interpolation.

import sys
import os
# sys.path.append(os.path.dirname("../lib_usplat4d_prep"))

import torch
import numpy as np

import pypose as pp
from torch import nn


from ugraph_helper import Precompute_helper
import roma
import torch.nn.functional as F

from loss_rigidity import cal_rigidity_loss,cal_rigidity_loss2,cal_rigidity_loss_extra, get_rigidity_loss
from lib_mosca.mosca import __DQB_warp2__, __LBS_warp2__

def get_local_ridigity_loss(precompute: Precompute_helper, ts: torch.Tensor):
    loss, stats=get_rigidity_loss(precompute, ts)
    return loss, stats

def get_graph_rigidity_interpolation_loss(precompute: Precompute_helper, ts: torch.Tensor, is_skip_key_graph: bool = False, is_skip_nonkey_graph: bool = False):
    # ts : [batch of time frames]

    transls=precompute.params['transls'] # all
    oris_wxyz=precompute.params['oris_wxyz'] # all
    dists_key=precompute.params['dists_key'] # key # [nk,nk]
    dists_oris_key=precompute.params['dists_oris_key'] # key #
    weight_dist_key=precompute.params['weight_dist_key']
    # dists_nonkey=precompute.params['dists_nonkey']
    # dists_oris_nonkey=precompute.params['dists_oris_nonkey']
    # weight_dist_nonkey=precompute.params['weight_dist_nonkey']
    weight_param_key=precompute.params['weight_param_key'] # TODO
    src_xyz_1_nonkey=precompute.params['src_xyz_1_nonkey']
    src_xyzw_1_nonkey=precompute.params['src_xyzw_1_nonkey']
    per_nonkeyGaussian_dweight_nonkey=precompute.params['per_nonkeyGaussian_dweight_nonkey']
    per_nonkeyGaussian_dxyz_dxyzw=precompute.params['per_nonkeyGaussian_dxyz_dxyzw']


    # contribs_key=precompute.graph_para['contribs_key']
    # contribs_nonkey=precompute.graph_para['contribs_nonkey']
    ids_key=precompute.graph_para['ids_key']
    ids_nonkey=precompute.graph_para['ids_nonkey']
    edges_graph_key=precompute.graph_para['edges_graph_key']
    edges_graph_nonkey=precompute.graph_para['edges_graph_nonkey']

    # exp_sq_dists_key_nonkey_edges=precompute.graph_para['exp_sq_dists_key_nonkey_edges'] # TODO
    ts_most_confident=precompute.graph_para['ts_most_confident']

    device=transls.device
    ids_key=ids_key.to(device)
    ids_nonkey=ids_nonkey.to(device)
    edges_graph_key=edges_graph_key.to(device)
    edges_graph_nonkey=edges_graph_nonkey.to(device)


    oris_wxyz = F.normalize(oris_wxyz, p=2, dim=-1)
    # oris_xyzw=roma.quat_xyzw_to_wxyz(oris_wxyz) # ??? [BUG]
    # oris_xyzw = oris_wxyz[:, :, [1, 2, 3, 0]]  # wxyz → xyzw
    oris_xyzw=roma.quat_wxyz_to_xyzw(oris_wxyz)

    assert torch.norm(oris_xyzw[3,5],p=2)-1<1e-6

    transls_key=transls[:,ids_key]
    transls_nonkey=transls[:,ids_nonkey]

    oris_xyzw_key=oris_xyzw[:,ids_key]
    oris_xyzw_nonkey=oris_xyzw[:,ids_nonkey]

    # calculate current exp_sq_dists_key_nonkey_edges
    # given ts_most_confident, transls_key, transls_nonkey
    # n_neighbors=5 # TODO
    n_neighbors=(edges_graph_nonkey[:,0]==0).sum()
    if n_neighbors!=5:
        breakpoint()
    nu=transls_nonkey.shape[1]
    transls_nonkey_ts_most_confident = transls_nonkey[ts_most_confident, torch.arange(nu,device=device)] # [nu,3]
    transls_nonkey_ts_most_confident_edges = transls_nonkey_ts_most_confident.repeat_interleave(n_neighbors, dim=0) # n_edges,3 = nu*n_neighbors,3
    transls_key_ts_most_confident_edges = transls_key[ts_most_confident.repeat_interleave(n_neighbors), edges_graph_nonkey[:,1]] # n_edges,3
    sq_dists_key_nonkey_edges = torch.sum((transls_nonkey_ts_most_confident_edges - transls_key_ts_most_confident_edges)**2, dim=-1) # n_edges
    exp_sq_dists_key_nonkey_edges = torch.exp(-sq_dists_key_nonkey_edges)

    # key edge loss
    if is_skip_key_graph==False:
        key_edge_loss, stats_key=get_key_edge_rigidity_loss(transls_key,dists_key,weight_dist_key,
                                        oris_xyzw_key,dists_oris_key,
                                        edges_graph_key,ts)
    else:
        key_edge_loss=torch.tensor(0.0).to(device)
        stats_key={
            'graph_key/loss':0.0,
            'graph_key/loss_rigidity':0.0,
            'graph_key/loss_weight_distance_regularizer':0.0,
            'graph_key/per_eff_weight_normed':0.0,
        }
    
    # nonkey edge loss
    if is_skip_nonkey_graph==False:
        nonkey_edge_loss, stats_nonkey, Transls_interpolation_ts_nonkey, Oris_xyzw_interpolation_ts_nonkey, ts_new=get_nonkey_edge_interpolation_loss(transls_nonkey,oris_xyzw_nonkey,
                                    transls_key,oris_xyzw_key,
                                    edges_graph_nonkey,ts,
                                    weight_param_key,src_xyz_1_nonkey,src_xyzw_1_nonkey,per_nonkeyGaussian_dweight_nonkey,
                                    exp_sq_dists_key_nonkey_edges,ts_most_confident,
                                    per_nonkeyGaussian_dxyz_dxyzw)
        
        # update interpolation results
        transls_interpolation=precompute.graph_para['transls_interpolation']
        oris_wxyz_interpolation=precompute.graph_para['oris_interpolation'] # wxyz
        transls_interpolation=transls_interpolation.to(device)
        oris_wxyz_interpolation=oris_wxyz_interpolation.to(device)
        transls_interpolation[:,ids_key]=transls_key
        oris_wxyz_interpolation[:,ids_key]=oris_wxyz[:,ids_key]
        transls_interpolation[ts][:,ids_nonkey]=Transls_interpolation_ts_nonkey
        Oris_wxyz_interpolation_ts_nonkey=roma.quat_xyzw_to_wxyz(Oris_xyzw_interpolation_ts_nonkey)
        oris_wxyz_interpolation[ts][:,ids_nonkey]=Oris_wxyz_interpolation_ts_nonkey #Oris_xyzw_interpolation_ts_nonkey[:,:,[3,0,1,2]] # xyzw -> wxyz
        precompute.graph_para['transls_interpolation']=transls_interpolation
        precompute.graph_para['oris_interpolation']=oris_wxyz_interpolation
    else: 
        nonkey_edge_loss=torch.tensor(0.0).to(device)
        stats_nonkey={
            'graph_nonkey/loss':0.0,
            'graph_nonkey/a_loss_dist_confident_pts':0.0,
            'graph_nonkey/a_loss_ori_confident_pts':0.0,
            'graph_nonkey/a_loss_per_nonkeyGaussian':0.0,
        }
    
    graph_loss = key_edge_loss + nonkey_edge_loss

    stats_graph = {**stats_key, **stats_nonkey}


    
    return graph_loss, stats_graph


def get_key_edge_rigidity_loss(Transls_optimizable, Dists_optimizable, weight_distance_optimizable, 
                      oris_xyzw_optimizable, Ori_distances_optimizable,
                      edges,ts):
    # Transls_optimizable [nt_all,nk,3]
    # Dists_optimizable [nk,nk]
    # weight_distance_optimizable [nk,nk]
    # oris_xyzw_optimizable [nt_all,nk,4]
    # Ori_distances_optimizable [nk,nk]
    # edges [ne=nk*nk,2]
    # ts : [nt_batch]
    flag_train_orientation=True
    nt=ts.shape[0]
    nk=Transls_optimizable.shape[1]
    ne=edges.shape[0]
    device=Transls_optimizable.device
    
    # oris_xyzw_optimizable_normed=oris_xyzw_optimizable/oris_xyzw_optimizable.norm(dim=-1,keepdim=True) 
    oris_xyzw_optimizable_normed=oris_xyzw_optimizable # alreadly normed
    weight_normed=torch.abs(weight_distance_optimizable)/torch.norm(weight_distance_optimizable,dim=-1,p=1)[:,None] # (nu,nk)
    weight_edges=weight_normed[edges[:,0],edges[:,1]] # (ne)
    dis_local_canonical_edge=Dists_optimizable[edges[:,0],edges[:,1]] # (ne) CHECK order
    # Slice to batch frames only: avoids [nt_all, ne, 3] OOM (427 frames × 50k edges × 3 = ~256 MB)
    # Stochastic sampling of frames is valid — over training, all frames are covered
    # loss_rigidity,stats_rigidity=cal_rigidity_loss(Transls_optimizable,oris_xyzw_optimizable_normed,edges,weight_edges,dis_local_canonical_edge,"graph_key/")
    loss_rigidity,stats_rigidity=cal_rigidity_loss_extra(Transls_optimizable[ts],oris_xyzw_optimizable_normed[ts],edges,weight_edges,dis_local_canonical_edge,"graph_key/")
    
    if flag_train_orientation:
        pass
    
    # Loss 7: symmetry loss for edge weights
    # TODO
    # loss_symmetry=torch.mean(torch.abs(weight_distance_optimizable-weight_distance_optimizable.transpose(0,1)))

    # Loss: avoid weight_distance_optimizable too large
    loss_weight_distance_regularizer=torch.mean(torch.abs(weight_distance_optimizable))


    a_loss_rigidity=100*loss_rigidity
    a_loss_weight_distance_regularizer=0.01*loss_weight_distance_regularizer

    key_edge_loss = a_loss_rigidity#+a_loss_weight_distance_regularizer
    
    stats={
        'graph_key/loss':key_edge_loss.item(),
        'graph_key/loss_rigidity':loss_rigidity.item(),
        'graph_key/loss_weight_distance_regularizer':loss_weight_distance_regularizer.item(),
        'graph_key/per_eff_weight_normed':(weight_normed>1e-2).sum().item()/(nk),
    }

    stats.update(stats_rigidity)

    return key_edge_loss, stats


def get_nonkey_edge_interpolation_loss(transls_nonkey_optimizable, oris_xyzw_nonkey_optimizable, 
                          Transls_key_optimizable, oris_xyzw_key_optimizable,
                          edges,ts,
                          weight_param_key_optimizable,src_xyz_1_nonkey,src_xyzw_1_nonkey,per_nonkeyGaussian_dweight_nonkey,
                          exp_sq_dists_key_nonkey_edges,ts_most_confident,
                          per_nonkeyGaussian_dxyz_dxyzw=None):

    nt_all = transls_nonkey_optimizable.shape[0]
    nu=transls_nonkey_optimizable.shape[1]
    nk=Transls_key_optimizable.shape[1]
    ne=edges.shape[0]
    device=transls_nonkey_optimizable.device
    # ts = torch.arange(0, nt_all, device=device) # test
    nt = ts.shape[0]

    # Transls_interpolation # [nt,nu,3]
    # Oris_interpolation # [nt,nu,4]
    # per_nonkeyGaussian_dxyz_dxyzw [nu,7]
    Transls_interpolation_ts_nonkey, Oris_xyzw_interpolation_ts_nonkey=cal_DQB_nonkeyGaussian(Transls_key_optimizable, oris_xyzw_key_optimizable,
                          edges,ts,
                          weight_param_key_optimizable,src_xyz_1_nonkey,src_xyzw_1_nonkey,per_nonkeyGaussian_dweight_nonkey,
                          exp_sq_dists_key_nonkey_edges,ts_most_confident)
    if per_nonkeyGaussian_dxyz_dxyzw is not None:
        Transls_interpolation_ts_nonkey=Transls_interpolation_ts_nonkey+per_nonkeyGaussian_dxyz_dxyzw[:,:3][None,:,:] # [nt,nu,3]
        Oris_xyzw_interpolation_ts_nonkey=Oris_xyzw_interpolation_ts_nonkey+per_nonkeyGaussian_dxyz_dxyzw[:,3:][None,:,:] # [nt,nu,4]

    # Loss 1: Confident points distance error
    Dist_confident_pts_error=torch.norm(Transls_interpolation_ts_nonkey-transls_nonkey_optimizable[ts],dim=-1) # [nt,nu]
    loss_dist_confident_pts = torch.mean(Dist_confident_pts_error)

    Ori_confident_pts_error=torch.norm(Oris_xyzw_interpolation_ts_nonkey-oris_xyzw_nonkey_optimizable[ts],dim=-1) # [nt,nu]
    loss_ori_confident_pts = torch.mean(Ori_confident_pts_error)

    # Loss 6: regularize per nonkey Gaussian delta weight
    # loss_per_nonkeyGaussian=torch.mean(torch.abs(per_nonkeyGaussian_dweight_nonkey))
    loss_per_nonkeyGaussian = torch.mean(per_nonkeyGaussian_dweight_nonkey ** 2) # TODO

    if per_nonkeyGaussian_dxyz_dxyzw is not None:
        loss_per_nonkeyGaussian_dxyz_dxyzw=torch.mean(per_nonkeyGaussian_dxyz_dxyzw ** 2) # TODO
    else:
        loss_per_nonkeyGaussian_dxyz_dxyzw=torch.tensor(0,device=device)

    a_loss_dist_confident_pts=1*loss_dist_confident_pts
    a_loss_ori_confident_pts=1*loss_ori_confident_pts
    a_loss_per_nonkeyGaussian=1e-5*loss_per_nonkeyGaussian
    a_loss_per_nonkeyGaussian_dxyz_dxyzw=1e-3*loss_per_nonkeyGaussian_dxyz_dxyzw
    
    nonkey_edge_loss = a_loss_dist_confident_pts +a_loss_ori_confident_pts+a_loss_per_nonkeyGaussian+a_loss_per_nonkeyGaussian_dxyz_dxyzw
 
    stats={
        'graph_nonkey/loss':nonkey_edge_loss.item(),
        'graph_nonkey/a_loss_dist_confident_pts':a_loss_dist_confident_pts.item(),
        'graph_nonkey/a_loss_ori_confident_pts':a_loss_ori_confident_pts.item(),
        'graph_nonkey/a_loss_per_nonkeyGaussian':a_loss_per_nonkeyGaussian.item(),
        'graph_nonkey/a_loss_per_nonkeyGaussian_dxyz_dxyzw':a_loss_per_nonkeyGaussian_dxyz_dxyzw.item(),
    }

    # stats.update(stats_rigidity)

    return nonkey_edge_loss, stats, Transls_interpolation_ts_nonkey, Oris_xyzw_interpolation_ts_nonkey, ts

def cal_weight_edges2(weight_param_key_optimizable,exp_sq_dists_key_nonkey_edges,edges,nu):
    # weight_param_key_optimizable [nk]
    # exp_sq_dists_key_nonkey_edges [ne]
    device=weight_param_key_optimizable.device
    nk=weight_param_key_optimizable.shape[0]

    weight_param_key_optimizable_k=100*torch.abs(weight_param_key_optimizable) # [TODO]: 2000
    weight_param_edges=weight_param_key_optimizable_k[edges[:,1]] # (ne)
    # assert all(weight_param_edges>0)
    # assert all(sq_dists_key_nonkey_edges>0)
    weight_edges = exp_sq_dists_key_nonkey_edges**weight_param_edges

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
    
def compute_accel_loss(transls):
    # transls [nt,n,3] or oris [nt,n,4]
    accel = 2 * transls[1:-1] - transls[:-2] - transls[2:] # [nt-2,n,3/4]
    loss = accel.norm(dim=-1).mean() # [nt-2,n,3/4] => [nt-2,n]=> [1]
    return loss


def cal_DQB_nonkeyGaussian(Transls_key_optimizable, oris_xyzw_key_optimizable,
                          edges,ts,
                          weight_param_key_optimizable,src_xyz_1_nonkey,src_xyzw_1_nonkey,per_nonkeyGaussian_dweight_nonkey,
                          exp_sq_dists_key_nonkey_edges,ts_most_confident):
    use_per_nonkeyGaussian_dweight=True

    nu = src_xyz_1_nonkey.shape[0]
    assert ts_most_confident.shape[0]==nu
    nt = ts.shape[0]

    # prepare
    n_neighbors=(edges[:,0]==0).sum()
    if n_neighbors!=5:
        breakpoint()
    # get sk_src_node and sk_dst_node (key nodes)
    ts_most_confident_repeat=ts_most_confident.repeat(n_neighbors,1).T.reshape(-1) # (nu*n_neighbors=ne)
    sk_src_node_xyz_1=Transls_key_optimizable[ts_most_confident_repeat,edges[:,1]] # (ne,3)
    sk_src_node_xyzw_1=oris_xyzw_key_optimizable[ts_most_confident_repeat,edges[:,1]] # (ne,4)
    sk_src_node_wxyz_1=sk_src_node_xyzw_1[:,[3,0,1,2]] # (ne,4)
    sk_src_node_xyz_2=sk_src_node_xyz_1.reshape(-1,n_neighbors,3) # (nu,n_neighbors,3)
    sk_src_node_quat_2=sk_src_node_wxyz_1.reshape(-1,n_neighbors,4) # (nu,n_neighbors,4)

    sk_nt_node_xyz_1=Transls_key_optimizable[ts][:,edges[:,1]] # (nt,ne,3)
    sk_nt_node_xyz_2=sk_nt_node_xyz_1.reshape(nt,-1,n_neighbors,3) # (nt,nu,n_neighbors,3)
    sk_nt_node_xyzw_1=oris_xyzw_key_optimizable[ts][:,edges[:,1]] # (nt,ne,4)
    sk_nt_node_wxyz_1=sk_nt_node_xyzw_1[:,:,[3,0,1,2]] # (nt,ne,4)
    sk_nt_node_quat_2=sk_nt_node_wxyz_1.reshape(nt,-1,n_neighbors,4) # (nt,nu,n_neighbors,4)

    sk_nt_src_node_xyz=sk_src_node_xyz_2[None].repeat(nt,1,1,1).reshape(-1,n_neighbors,3) # (nt*nu,n_neighbors,3)
    sk_nt_src_node_quat=sk_src_node_quat_2[None].repeat(nt,1,1,1).reshape(-1,n_neighbors,4) # (nt*nu,n_neighbors,4)
    sk_nt_dst_node_xyz=sk_nt_node_xyz_2.reshape(-1,n_neighbors,3) # (nt*nu,n_neighbors,3)
    sk_nt_dst_node_quat=sk_nt_node_quat_2.reshape(-1,n_neighbors,4) # (nt*nu,n_neighbors,4)


    if torch.isnan(weight_param_key_optimizable).any():
        print("weight_param_key_optimizable is nan")
        breakpoint()

    
    src_xyz=src_xyz_1_nonkey[None].repeat(nt,1,1).reshape(-1,3) # (nt*nu,3)
    src_wxyz_1=src_xyzw_1_nonkey[:,[3,0,1,2]]
    src_quat=src_wxyz_1[None].repeat(nt,1,1).reshape(-1,4) # (nt*nu,4)

    weight_edges=cal_weight_edges2(weight_param_key_optimizable,exp_sq_dists_key_nonkey_edges,edges,nu) # [ne] # CHECK: 2000
    weight_per_nonkeyGaussian=weight_edges.reshape(-1,n_neighbors) # (nu,n_neighbors)
    if use_per_nonkeyGaussian_dweight:
        weight_per_nonkeyGaussian_adjusted=torch.abs(weight_per_nonkeyGaussian+per_nonkeyGaussian_dweight_nonkey) # (nu,n_neighbors)
        weight_per_nonkeyGaussian_adjusted=weight_per_nonkeyGaussian_adjusted/torch.sum(weight_per_nonkeyGaussian_adjusted,dim=-1,keepdim=True) # (nu,n_neighbors)
        sk_w=weight_per_nonkeyGaussian_adjusted[None,:,:].repeat(nt,1,1).reshape(-1,n_neighbors) # (nt*nu,n_neighbors)
    else:
        sk_w=weight_per_nonkeyGaussian[None,:,:].repeat(nt,1,1).reshape(-1,n_neighbors) # (nt*nu,n_neighbors)

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
    Transls_interpolation=mu_dst.reshape(nt,nu,3) # [nt,nu,3]
    Oris_xyzw_interpolation=quat_dst_xyzw.reshape(nt,nu,4) # [nt,nu,4]

    return Transls_interpolation,Oris_xyzw_interpolation

def get_DQB_nonkeyGaussian(precompute: Precompute_helper, ts: torch.Tensor):
    # ts : [batch of time frames]

    transls=precompute.params['transls'] # all
    oris_wxyz=precompute.params['oris_wxyz'] # all

    weight_param_key=precompute.params['weight_param_key'] # TODO
    src_xyz_1_nonkey=precompute.params['src_xyz_1_nonkey']
    src_xyzw_1_nonkey=precompute.params['src_xyzw_1_nonkey']
    per_nonkeyGaussian_dweight_nonkey=precompute.params['per_nonkeyGaussian_dweight_nonkey']


    ids_key=precompute.graph_para['ids_key']
    ids_nonkey=precompute.graph_para['ids_nonkey']
    edges_graph_key=precompute.graph_para['edges_graph_key']
    edges_graph_nonkey=precompute.graph_para['edges_graph_nonkey']

    # exp_sq_dists_key_nonkey_edges=precompute.graph_para['exp_sq_dists_key_nonkey_edges'] # TODO
    ts_most_confident=precompute.graph_para['ts_most_confident']

    device=transls.device
    ids_key=ids_key.to(device)
    ids_nonkey=ids_nonkey.to(device)
    edges_graph_key=edges_graph_key.to(device)
    edges_graph_nonkey=edges_graph_nonkey.to(device)


    oris_wxyz = F.normalize(oris_wxyz, p=2, dim=-1)
    # oris_xyzw=roma.quat_xyzw_to_wxyz(oris_wxyz) # ??? [BUG]
    # oris_xyzw = oris_wxyz[:, :, [1, 2, 3, 0]]  # wxyz → xyzw
    oris_xyzw=roma.quat_wxyz_to_xyzw(oris_wxyz)

    assert torch.norm(oris_xyzw[3,5],p=2)-1<1e-6

    transls_key=transls[:,ids_key]
    transls_nonkey=transls[:,ids_nonkey]

    oris_xyzw_key=oris_xyzw[:,ids_key]
    # oris_xyzw_nonkey=oris_xyzw[:,ids_nonkey]

    # calculate current exp_sq_dists_key_nonkey_edges
    # given ts_most_confident, transls_key, transls_nonkey
    n_neighbors=(edges_graph_nonkey[:,0]==0).sum()
    if n_neighbors!=5:
        breakpoint()
    nu=transls_nonkey.shape[1]
    transls_nonkey_ts_most_confident = transls_nonkey[ts_most_confident, torch.arange(nu,device=device)] # [nu,3]
    transls_nonkey_ts_most_confident_edges = transls_nonkey_ts_most_confident.repeat_interleave(n_neighbors, dim=0) # n_edges,3 = nu*n_neighbors,3
    transls_key_ts_most_confident_edges = transls_key[ts_most_confident.repeat_interleave(n_neighbors), edges_graph_nonkey[:,1]] # n_edges,3
    sq_dists_key_nonkey_edges = torch.sum((transls_nonkey_ts_most_confident_edges - transls_key_ts_most_confident_edges)**2, dim=-1) # n_edges
    exp_sq_dists_key_nonkey_edges = torch.exp(-sq_dists_key_nonkey_edges)
    
    Transls_interpolation_ts_nonkey, Oris_xyzw_interpolation_ts_nonkey=cal_DQB_nonkeyGaussian(transls_key,oris_xyzw_key,
                                edges_graph_nonkey,ts,
                                weight_param_key,src_xyz_1_nonkey,src_xyzw_1_nonkey,per_nonkeyGaussian_dweight_nonkey,
                                exp_sq_dists_key_nonkey_edges,ts_most_confident)
    
    # update interpolation results
    transls_interpolation=precompute.graph_para['transls_interpolation']
    oris_wxyz_interpolation=precompute.graph_para['oris_interpolation'] # wxyz
    transls_interpolation=transls_interpolation.to(device)
    oris_wxyz_interpolation=oris_wxyz_interpolation.to(device)
    transls_interpolation[:,ids_key]=transls_key
    oris_wxyz_interpolation[:,ids_key]=oris_wxyz[:,ids_key]
    transls_interpolation[ts][:,ids_nonkey]=Transls_interpolation_ts_nonkey
    Oris_wxyz_interpolation_ts_nonkey=roma.quat_xyzw_to_wxyz(Oris_xyzw_interpolation_ts_nonkey)
    oris_wxyz_interpolation[ts][:,ids_nonkey]=Oris_wxyz_interpolation_ts_nonkey #Oris_xyzw_interpolation_ts_nonkey[:,:,[3,0,1,2]] # xyzw -> wxyz
    precompute.graph_para['transls_interpolation']=transls_interpolation
    precompute.graph_para['oris_interpolation']=oris_wxyz_interpolation

    #
    # Create empty tensor for oris
    
    nt = ts.shape[0]
    n_total = len(ids_key) + len(ids_nonkey)

    transls_ts = torch.zeros((nt, n_total, 3), device=transls.device, dtype=transls.dtype)
    transls_key_ts=transls[ts][:,ids_key]
    transls_ts[:, ids_key] = transls_key_ts
    transls_ts[:, ids_nonkey] = Transls_interpolation_ts_nonkey

    oris_wxyz_ts = torch.zeros((nt, n_total, 4), device=oris_wxyz.device, dtype=oris_wxyz.dtype)
    oris_wxyz_key_ts=oris_wxyz[ts][:,ids_key]
    oris_wxyz_ts[:, ids_key] = oris_wxyz_key_ts
    oris_wxyz_ts[:, ids_nonkey] = Oris_wxyz_interpolation_ts_nonkey
    
    return transls_ts, oris_wxyz_ts
