import torch
import os
import glob
import pickle
import numpy as np
import pypose as pp
import torch.nn as nn
import roma
import open3d as o3d
from tqdm import tqdm

class Precompute_helper(nn.Module):
    def __init__(
        self,
        params=None,
        graph_para=None,
        graph_dir=""
    ):
        super().__init__()
        if params==None or graph_para==None:
            if graph_dir=="":
                # self.work_dir="/data/dataset/used_ds/som/sriracha-tree/work-dir" # backpack,haru-sit, sriracha-tree 
                self.work_dir="/data/dataset/custom_ds_graph_model/rhino/" # rhino, schoolgirls
                # self.work_dir="/data/dataset/iphone_som_graph_model/block" # 
            else:
                print(f"graph_dir:{graph_dir}")
                self.work_dir=graph_dir
        

            params,graph_para=self.combine()
            self.params = nn.ParameterDict(params)
            self.graph_para = graph_para
            for name, tensor in graph_para.items():
                self.register_buffer(name, tensor)
            # self.graph_para = nn.ParameterDict(graph_para) # CHECK
        else:
            self.params=nn.ParameterDict(params)
            self.graph_para=graph_para
            for name, tensor in graph_para.items():
                self.register_buffer(name, tensor)

    def set_para_device(self,device):
        # for name, tensor in self.params.items():
        #     self.params[name]=tensor.to(device)
        for name, tensor in self.graph_para.items():
            self.graph_para[name]=tensor.to(device)
        # for name, tensor in self.graph_para.items(): # maybe not necessary
        #     getattr(self, name) = tensor # incorrect

    def init_from_means_quats(self,means,quats):
        params = nn.ParameterDict()
        params['transls'] = nn.Parameter(means)
        params['oris_wxyz'] = nn.Parameter(quats)
        graph_para = {}

        return Precompute_helper(params,graph_para)

    @staticmethod
    def init_from_state_dict(state_dict, prefix="precompute."): # TODO
        req_keys = ["dir_matrix_key", "dir_matrix_nonkey", "dists_key", "dists_nonkey", "dists_oris_key", "dists_oris_nonkey", "oris_wxyz", "transls", "weight_dir_key", "weight_dir_nonkey", "weight_dist_key", "weight_dist_nonkey",
                    'weight_param_key','src_xyz_1_nonkey','src_xyzw_1_nonkey','per_nonkeyGaussian_dweight_nonkey']
        if not all(f"{prefix}params.{k}" in state_dict for k in req_keys):
            print("Precompute: Missing keys in state_dict for [params]")
            return None
        params={}
        for k in req_keys:
            if f"{prefix}params.{k}" in state_dict:
                params[k] = state_dict[f"{prefix}params.{k}"]

        

        req_keys2 = ["contribs_key", "contribs_nonkey", "ids_key", "ids_nonkey", "ids_key_nonkey", "edges_graph_key", "edges_graph_nonkey", "edges_local", "sq_dists_local", "weight_rigidity",
                     'transls_som','oris_wxyz_som',
                     'exp_sq_dists_key_nonkey_edges','ts_most_confident','transls_interpolation','oris_interpolation',
                     'contrib_threshold_key']
        if not all(f"{prefix}{k}" in state_dict for k in req_keys2):
            print("Precompute: Missing keys in state_dict for [graph_para]")
            return None
        graph_para={}
        for k in req_keys2:
            if f"{prefix}{k}" in state_dict:
                graph_para[k] = state_dict[f"{prefix}{k}"]

        
        req_key_extra="per_nonkeyGaussian_dxyz_dxyzw"
        if f"{prefix}params.{req_key_extra}" in state_dict:
            params[req_key_extra] = state_dict[f"{prefix}params.{req_key_extra}"]
        else:
            nu=graph_para["ids_nonkey"].shape[0]
            params[req_key_extra] = torch.zeros((nu,7)).to(params['transls'].device) # dummy

        # params_dict={
        #     # 'transls_som':nn.Parameter(Transls_SoM),
        #     # 'oris_wxyz_som':nn.Parameter(Oris_wxyz_SoM),
        #     'transls':nn.Parameter(Transls), # Transls # Transls_SoM
        #     'oris_wxyz':nn.Parameter(Oris_wxyz), # Oris_wxyz # Oris_wxyz_SoM
        #     'dists_key':nn.Parameter(Dists_optimized),
        #     'dists_oris_key': nn.Parameter(Ori_distances_optimized),
        #     'weight_dist_key': nn.Parameter(weight_distance_optimized),
        #     'dists_nonkey': nn.Parameter(Dists_optimized2),
        #     'dists_oris_nonkey': nn.Parameter(Ori_distances_optimized2),
        #     'weight_dist_nonkey': nn.Parameter(weight_distance_optimized2),  
        #     'dir_matrix_key': nn.Parameter(dir_matrix_key),
        #     'weight_dir_key': nn.Parameter(weight_dir_key),
        #     'dir_matrix_nonkey': nn.Parameter(dir_matrix_nonkey),
        #     'weight_dir_nonkey': nn.Parameter(weight_dir_nonkey), 
        # }
        # graph_para={
        #     'contribs_key':contribs_key,
        #     'contribs_nonkey': contribs_nonkey,
        #     'ids_key':ids_key,
        #     'ids_nonkey':ids_nonkey,
        #     'ids_key_nonkey': ids_key_nonkey,
        #     'edges_graph_key':edges_graph_key, # [...[key id, key id]...]
        #     'edges_graph_nonkey':edges_graph_nonkey, # [...[key id, non key id]...]
        #     'edges_local':edges_local, # [num_edges_local,2]
        #     'sq_dists_local': sq_dists_local, # [num_nodes, num_neighbors] (num_edges_local=num_nodes*num_neighbors)
        #     'weight_rigidity':weight_rigidity, # [num_edges_local]
        # }
        args={"params":params,"graph_para":graph_para}
        return Precompute_helper(**args)
    
    @staticmethod
    def init_from_ini_ckpt(): # TODO
        work_dir="/data/dataset/used_ds/som/backpack/work-dir1" # backpack work-dir1 ini.ckpt
        Transls,Oris_xyzw,Contribs=read_precomputed_pose_som(work_dir)
        Transls=torch.tensor(Transls)
        Oris_xyzw=Oris_xyzw.tensor()
        nt=Transls.shape[0]

        Transls_SoM=Transls
        Oris_xyzw_SoM=Oris_xyzw
        Oris_wxyz_SoM=roma.quat_xyzw_to_wxyz(Oris_xyzw_SoM)
        contribs=Contribs

        n_neighbors=20
        edges_local,sq_dists_local=get_knn_neighbors_by_contribs(Transls_SoM,contribs,n_neighbors,contrib_threshold=0.4) # [num_nodes, num_neighbors]
        # sq_dists_local is canonical square distances!
        lambda_rigidity=2000 # TODO: CHECK!
        weight_rigidity=torch.exp(-lambda_rigidity*sq_dists_local) # [num_nodes, num_neighbors] # not learnable

        params={
            'transls':nn.Parameter(Transls_SoM), # Transls # Transls_SoM
            'oris_wxyz':nn.Parameter(Oris_wxyz_SoM), # Oris_wxyz # Oris_wxyz_SoM
        }
        graph_para={
            'edges_local':edges_local, # [num_edges_local,2]
            'sq_dists_local': sq_dists_local, # [num_nodes, num_neighbors] (num_edges_local=num_nodes*num_neighbors)
            'weight_rigidity':weight_rigidity, # [num_edges_local]
        }

        args={"params":params,"graph_para":graph_para}
        return Precompute_helper(**args)

    def combine(self):
        ## 
        Transls,Oris_xyzw,Contribs=self.read_precomputed_pose_som()
        Transls=torch.tensor(Transls)
        Oris_xyzw=Oris_xyzw.tensor()
        nt=Transls.shape[0]

        parameters=self.read_precomputed_pose_graph()

        if True:
            ids=parameters['ids']
            Transls_optimized=parameters['Transls_optimized']
            Dists_optimized=parameters['Dists_optimized']
            weight_distance_optimized=parameters['weight_distance_optimized']

            Oris_xyzw_optimized=parameters['Oris_xyzw_optimized']
            Ori_distances_optimized=parameters['Ori_distances_optimized']
            edges_graph_key=parameters['edges_graph']
            
            ids2=parameters['ids2']
            Transls_optimized2=parameters['Transls_optimized2']
            Dists_optimized2=parameters['Dists_optimized2']
            Oris_xyzw_optimized2=parameters['Oris_xyzw_optimized2']
            Ori_distances_optimized2=parameters['Ori_distances_optimized2']
            weight_distance_optimized2=parameters['weight_distance_optimized2']
            # edges_others_optimized=parameters['edges_others_optimized'] # [nonkey_is,key_is]
            edges_graph_nonkey=parameters['edges_others'] # [nonkey_is,key_is]

            contribs_key=parameters['Contribs']
            contribs_nonkey=parameters['Contribs2']

            # for DQB interpolation
            blend_opt_dict=parameters['blend_opt_dict']
            weight_param_key=blend_opt_dict['weight_param_key_optimizable']
            src_xyz_1_nonkey=blend_opt_dict['src_xyz_1']
            src_xyzw_1_nonkey=blend_opt_dict['src_xyzw_1']
            per_nonkeyGaussian_dweight_nonkey=blend_opt_dict['per_nonkeyGaussian_dweight']
            exp_sq_dists_key_nonkey_edges=blend_opt_dict['exp_sq_dists_key_nonkey_edges']
            ts_most_confident=blend_opt_dict['ts_most_confident']
    
        nu=per_nonkeyGaussian_dweight_nonkey.shape[0]
        per_nonkeyGaussian_dxyz_dxyzw = torch.zeros((nu,7)) # dummy

        contrib_threshold_key=torch.tensor(parameters['contrib_threshold'])
        print("contrib_threshold_key",contrib_threshold_key)
        

        assert len(ids)==Transls_optimized.shape[1]
        assert len(ids2)==Transls_optimized2.shape[1]

        ids_key_nonkey=np.array(ids+ids2)
        ids_key_nonkey=np.sort(ids_key_nonkey)

        Transls_SoM=Transls[:,ids_key_nonkey].clone()
        Oris_xyzw_SoM=Oris_xyzw[:,ids_key_nonkey].clone()
        Oris_wxyz_SoM=roma.quat_xyzw_to_wxyz(Oris_xyzw_SoM)

        Transls[:,np.array(ids)]=Transls_optimized.cpu() # (nt, nk, 3)
        Transls[:,np.array(ids2)]=Transls_optimized2.cpu() # (nt, nu, 3)

        # Oris_xyzw=Oris_xyzw.tensor()
        if torch.is_tensor(Oris_xyzw_optimized):
            Oris_xyzw[:,np.array(ids)]=Oris_xyzw_optimized.cpu()
            Oris_xyzw[:,np.array(ids2)]=Oris_xyzw_optimized2.cpu()
        else:
            Oris_xyzw[:,np.array(ids)]=Oris_xyzw_optimized.tensor().cpu()
            Oris_xyzw[:,np.array(ids2)]=Oris_xyzw_optimized2.tensor().cpu()
        Oris_wxyz=roma.quat_xyzw_to_wxyz(Oris_xyzw)
        Transls=Transls[:,ids_key_nonkey] ######### TODO # test
        Oris_wxyz=Oris_wxyz[:,ids_key_nonkey]
        ids_key,ids_nonkey=[],[]
        i,j,k=0,0,0
        while True:
            n1=len(ids)
            n2=len(ids2)
            assert i+j==k
            if i<n1 and j<n2:
                if ids[i]<ids2[j]:
                    ids_key.append(k)
                    i+=1
                else:
                    ids_nonkey.append(k)
                    j+=1
            elif i<n1:
                ids_key.append(k)
                i+=1
            elif j<n2:
                ids_nonkey.append(k)
                j+=1
            else:
                break
            k+=1
        ids_key=np.array(ids_key)
        ids_nonkey=np.array(ids_nonkey)

        dir_matrix_key=torch.rand((len(ids),len(ids),3)) # (nk,nk,3)
        dir_matrix_key=dir_matrix_key/torch.norm(dir_matrix_key,dim=-1,keepdim=True)
        weight_dir_key=torch.zeros((len(ids),len(ids)))
        weight_dir_key[edges_graph_key[:,0],edges_graph_key[:,1]]=1

        dir_matrix_nonkey=torch.rand((len(ids2),len(ids),3)) # (nu,nk,3)
        dir_matrix_nonkey=dir_matrix_nonkey/torch.norm(dir_matrix_nonkey,dim=-1,keepdim=True)
        weight_dir_nonkey=torch.zeros((len(ids2),len(ids))) # (nu,nk)
        weight_dir_nonkey[edges_graph_nonkey[:,0],edges_graph_nonkey[:,1]]=1


        # combine contribs
        contribs=np.zeros((nt,ids_key_nonkey.shape[0])) # nt, num_nodes
        contribs[:,ids_key]=contribs_key
        contribs[:,ids_nonkey]=contribs_nonkey

        # for local and gloabl rigidity
        # TODO: adjust contrib_threshold
        n_neighbors=20
        edges_local,sq_dists_local=get_knn_neighbors_by_contribs(Transls_SoM,contribs,n_neighbors,contrib_threshold=0.4) # [num_nodes, num_neighbors]
        # sq_dists_local is canonical square distances!
        lambda_rigidity=2000 # TODO: CHECK!
        weight_rigidity=torch.exp(-lambda_rigidity*sq_dists_local) # [num_nodes, num_neighbors] # not learnable


        params_dict={
            'transls':nn.Parameter(Transls), #  
            'oris_wxyz':nn.Parameter(Oris_wxyz), #  
            'dists_key':nn.Parameter(Dists_optimized),
            'dists_oris_key': nn.Parameter(Ori_distances_optimized), #
            'weight_dist_key': nn.Parameter(weight_distance_optimized),
            'dists_nonkey': nn.Parameter(Dists_optimized2),
            'dists_oris_nonkey': nn.Parameter(Ori_distances_optimized2), #
            'weight_dist_nonkey': nn.Parameter(weight_distance_optimized2),  
            'dir_matrix_key': nn.Parameter(dir_matrix_key), #
            'weight_dir_key': nn.Parameter(weight_dir_key), #
            'dir_matrix_nonkey': nn.Parameter(dir_matrix_nonkey), #
            'weight_dir_nonkey': nn.Parameter(weight_dir_nonkey),  #
            'weight_param_key': nn.Parameter(weight_param_key),
            'src_xyz_1_nonkey': nn.Parameter(src_xyz_1_nonkey), # (nu,3)
            'src_xyzw_1_nonkey': nn.Parameter(src_xyzw_1_nonkey), # (nu,4)
            'per_nonkeyGaussian_dweight_nonkey': nn.Parameter(per_nonkeyGaussian_dweight_nonkey), # (nu,n_neighbors)
            'per_nonkeyGaussian_dxyz_dxyzw': nn.Parameter(per_nonkeyGaussian_dxyz_dxyzw), # (nu,7)
        }
        graph_para={
            'contribs_key':torch.tensor(contribs_key),
            'contribs_nonkey': torch.tensor(contribs_nonkey),
            'ids_key':torch.tensor(ids_key),
            'ids_nonkey':torch.tensor(ids_nonkey),
            'ids_key_nonkey': torch.tensor(ids_key_nonkey),
            'edges_graph_key':edges_graph_key, # [...[key id, key id]...]
            'edges_graph_nonkey':edges_graph_nonkey, # [...[non key id, key id]...]
            'edges_local':edges_local, # [num_edges_local,2]
            'sq_dists_local': sq_dists_local, # [num_nodes, num_neighbors] (num_edges_local=num_nodes*num_neighbors)
            'weight_rigidity':weight_rigidity, # [num_edges_local]
            'transls_som':Transls_SoM,
            'oris_wxyz_som':Oris_wxyz_SoM,
            'exp_sq_dists_key_nonkey_edges':exp_sq_dists_key_nonkey_edges, # (ne)
            'ts_most_confident':ts_most_confident, # (nu)
            'transls_interpolation': Transls.clone().detach(),
            'oris_interpolation': Oris_wxyz.clone().detach(),
            'contrib_threshold_key': contrib_threshold_key, # scalar
        }
            
        return params_dict,graph_para
    
    def cull_params(self, should_cull):
        """
        cull gaussians
        """
        updated_params = {}
        should_cull_key=should_cull[self.ids_key] # Check
        assert should_cull_key.any() == False
        should_cull_nonkey=should_cull[self.ids_nonkey] # Check
        for name, x in self.params.items():
            if name in ['transls','oris_wxyz']:
                x_new = nn.Parameter(x[:,~should_cull]) # check
            elif "_nonkey" in name:
                x_new = nn.Parameter(x[~should_cull_nonkey]) # check
            elif "_key" in name:
                continue
                # x_new = x
            else:
                breakpoint()
            updated_params[name] = x_new
            self.params[name] = x_new

        updated_graph_para = {}
        graph_para_last=self.graph_para.copy()
        for name, x in self.graph_para.items():
            if "_key" in name and "_nonkey" not in name:
                # x_new = x
                continue
            elif name in ['contribs_nonkey']:
                x_new = x[:,~should_cull_nonkey]
            elif name in ['ids_nonkey']:
                x_new = x[~should_cull_nonkey]
            elif name in ['ids_key_nonkey']:
                x_new = x[~should_cull]
            elif name in ['edges_graph_nonkey']:
                nk=graph_para_last['ids_key'].shape[0]
                nu=graph_para_last['ids_nonkey'].shape[0]
                device=x.device
                adjacent_matrix=torch.zeros((nu,nk),device=device)                
                for e in x:
                    adjacent_matrix[e[0],e[1]]=1
                adjacent_matrix_new = adjacent_matrix[~should_cull_nonkey,:]
                mask = adjacent_matrix_new>0
                flat_Edges_keep=np.argwhere(mask.cpu().numpy())
                flat_Edges_keep = torch.tensor(flat_Edges_keep, dtype=torch.long, device=device)
                x_new = flat_Edges_keep
            elif name in ['edges_local','sq_dists_local','weight_rigidity']:
                continue
            elif name in ['transls_som','oris_wxyz_som']:
                x_new = x[:,~should_cull]
            elif name in ['exp_sq_dists_key_nonkey_edges']:
                n_neighbors=5 ### TODO
                x_new = x.reshape(-1,n_neighbors)[~should_cull_nonkey].reshape(-1) # CHECK
            elif name in ['ts_most_confident']:
                x_new = x[~should_cull_nonkey]
            elif name in ['transls_interpolation','oris_interpolation']:
                x_new = x[:,~should_cull]
            else:
                breakpoint()
            updated_graph_para[name] = x_new
            self.graph_para[name] = x_new

        # update ['edges_local','sq_dists_local','weight_rigidity'] in self.graph_para
        # reculate uncertainty # TODO # Currently, just use previous uncertainty for easy implementation
        contribs_key=self.graph_para['contribs_key']
        contribs_nonkey=self.graph_para['contribs_nonkey']
        ids_key=self.graph_para['ids_key']
        ids_nonkey=self.graph_para['ids_nonkey']
        # resort. [0,2,3,5]=>[0,1,2,3]
        ids_key,ids_nonkey=resort_ids(ids_key,ids_nonkey) # Check
        ids_key=torch.tensor(ids_key,device=contribs_key.device)
        ids_nonkey=torch.tensor(ids_nonkey,device=contribs_key.device)
        Transls_SoM=self.graph_para['transls_som']

        assert Transls_SoM.shape[1]==ids_key.shape[0]+ids_nonkey.shape[0]
        assert torch.max(ids_key.max(),ids_nonkey.max())==ids_key.shape[0]+ids_nonkey.shape[0]-1

        nt=contribs_key.shape[0]
        contribs=torch.zeros((nt,ids_key.shape[0]+ids_nonkey.shape[0]),device=contribs_key.device) # nt, num_nodes
        contribs[:,ids_key]=contribs_key
        contribs[:,ids_nonkey]=contribs_nonkey
        # 
        # for local and gloabl rigidity
        n_neighbors=20
        edges_local,sq_dists_local=get_knn_neighbors_by_contribs(Transls_SoM,contribs,n_neighbors,contrib_threshold=0.4) # [num_nodes, num_neighbors]
        # sq_dists_local is canonical square distances!
        lambda_rigidity=2000 # TODO: CHECK!
        weight_rigidity=torch.exp(-lambda_rigidity*sq_dists_local) # [num_nodes, num_neighbors] # not learnable
        updated_graph_para['ids_key']=ids_key
        updated_graph_para['ids_nonkey']=ids_nonkey
        updated_graph_para['edges_local']=edges_local
        updated_graph_para['sq_dists_local']=sq_dists_local
        updated_graph_para['weight_rigidity']=weight_rigidity
        self.graph_para['ids_key']=ids_key
        self.graph_para['ids_nonkey']=ids_nonkey
        self.graph_para['edges_local']=edges_local
        self.graph_para['sq_dists_local']=sq_dists_local
        self.graph_para['weight_rigidity']=weight_rigidity

        self.update_graph_para_buffer(updated_graph_para)

        return updated_params, updated_graph_para, should_cull_nonkey
    
    def densify_params(self, should_split, should_dup):
        """
        densify gaussians
        """
        if should_split is not None:
            assert should_split.sum()==0 # currently, no gaussian is split.

        updated_params = {}
        should_dup_key=should_dup[self.ids_key] # Check
        assert should_dup_key.any()==False
        should_dup_nonkey=should_dup[self.ids_nonkey] # Check
        for name, x in self.params.items():
            if name in ['transls','oris_wxyz']: # (nt, ng, 3) (nt, ng, 4)
                x_dup = x[:,should_dup]
                x_new = nn.Parameter(torch.cat([x, x_dup], dim=1)) # check
            elif name in ['src_xyz_1_nonkey','src_xyzw_1_nonkey','per_nonkeyGaussian_dweight_nonkey','per_nonkeyGaussian_dxyz_dxyzw']:
                x_dup = x[should_dup_nonkey]
                x_new = nn.Parameter(torch.cat([x, x_dup], dim=0))
            elif "_nonkey" in name: # (nu, ...) (nu)
                name_key=name.replace("_nonkey","_key")
                x_key=self.params[name_key]
                ng=x_key.shape[0]+x.shape[0]
                x_all = torch.zeros((ng,*x.shape[1:]),device=x.device)
                x_all[self.ids_key]=x_key
                x_all[self.ids_nonkey]=x
                x_dup = x_all[should_dup]
                x_new = nn.Parameter(torch.cat([x, x_dup], dim=0)) 
            elif "_key" in name:
                continue
            else:
                breakpoint()
            updated_params[name] = x_new
            self.params[name] = x_new

        
        updated_graph_para = {}
        graph_para_last=self.graph_para.copy()
        for name, x in self.graph_para.items():
            if "_key" in name and "_nonkey" not in name:
                continue
            elif name in ['contribs_nonkey']:
                name_key=name.replace("_nonkey","_key")
                x_key=graph_para_last[name_key]
                ng=x_key.shape[1]+x.shape[1]
                x_all = torch.zeros((x.shape[0],ng),device=x.device) # check
                x_all[:,self.ids_key]=x_key
                x_all[:,self.ids_nonkey]=x
                x_dup = x_all[:,should_dup]
                x_new = torch.cat([x,x_dup],dim=1)
            elif name in ['ids_nonkey']:
                name_key=name.replace("_nonkey","_key")
                x_key=graph_para_last[name_key]
                ng=x_key.shape[0]+x.shape[0]
                x_dup = torch.arange(ng,ng+should_dup.sum(),device=x.device)
                x_new = torch.cat([x,x_dup],dim=0) # check
            elif name in ['ids_key_nonkey']: # should not be used after initialization
                x_dup = x[should_dup]
                x_new = torch.cat([x, x_dup], dim=0) # check
            elif name in ['edges_graph_nonkey']:
                device=x.device
                # get edges
                weight_distance_optimized2=self.params['weight_dist_nonkey']
                nk=weight_distance_optimized2.shape[1]
                nu=weight_distance_optimized2.shape[0]
                norm_weight_distance_optimized2 = (torch.abs(weight_distance_optimized2)/torch.norm(weight_distance_optimized2,dim=-1,p=1)[:,None]).cpu().detach().numpy()
                edges_others_optimized=[]
                mask = norm_weight_distance_optimized2 > 0
                edges_others_optimized = np.argwhere(mask)
                edges_others_optimized = torch.tensor(edges_others_optimized, dtype=torch.long, device=device)
                x_new = edges_others_optimized # CHECK: for dup_key, they has random number of edges
                if x_new.shape[0]>2000000:
                    breakpoint()
            elif name in ['edges_local','sq_dists_local','weight_rigidity']:
                continue
            elif name in ['transls_som','oris_wxyz_som']:
                x_dup = x[:,should_dup]
                x_new = torch.cat([x, x_dup], dim=1) # check
            elif name in ['exp_sq_dists_key_nonkey_edges']:
                n_neighbors=5 ### TODO
                x_dup = x.reshape(-1,n_neighbors)[should_dup_nonkey].reshape(-1) # CHECK
                x_new = torch.cat([x,x_dup],dim=0)
            elif name in ['ts_most_confident']:
                x_dup = x[should_dup_nonkey]
                x_new = torch.cat([x,x_dup],dim=0)
            elif name in ['transls_interpolation','oris_interpolation']:
                x_dup = x[:,should_dup]
                x_new = torch.cat([x,x_dup],dim=1)
            else:
                breakpoint()
            updated_graph_para[name] = x_new
            self.graph_para[name] = x_new
        
        # update ['edges_local','sq_dists_local','weight_rigidity'] in self.graph_para
        # reculate uncertainty # TODO # Currently, just use initial uncertainty for faster training
        contribs_key=self.graph_para['contribs_key']
        contribs_nonkey=self.graph_para['contribs_nonkey']
        ids_key=self.graph_para['ids_key']
        ids_nonkey=self.graph_para['ids_nonkey']
        Transls_SoM=self.graph_para['transls_som']

        assert Transls_SoM.shape[1]==ids_key.shape[0]+ids_nonkey.shape[0]
        assert torch.max(ids_key.max(),ids_nonkey.max())==ids_key.shape[0]+ids_nonkey.shape[0]-1

        nt=contribs_key.shape[0]
        contribs=torch.zeros((nt,ids_key.shape[0]+ids_nonkey.shape[0]),device=contribs_key.device) # nt, num_nodes
        contribs[:,ids_key]=contribs_key
        contribs[:,ids_nonkey]=contribs_nonkey
        # 
        # for local and gloabl rigidity
        n_neighbors=20
        edges_local,sq_dists_local=get_knn_neighbors_by_contribs(Transls_SoM,contribs,n_neighbors,contrib_threshold=0.4) # [num_nodes, num_neighbors]
        # sq_dists_local is canonical square distances!
        lambda_rigidity=2000 # TODO: CHECK!
        weight_rigidity=torch.exp(-lambda_rigidity*sq_dists_local) # [num_nodes, num_neighbors] # not learnable
        updated_graph_para['edges_local']=edges_local
        updated_graph_para['sq_dists_local']=sq_dists_local
        updated_graph_para['weight_rigidity']=weight_rigidity
        self.graph_para['edges_local']=edges_local
        self.graph_para['sq_dists_local']=sq_dists_local
        self.graph_para['weight_rigidity']=weight_rigidity

        self.update_graph_para_buffer(updated_graph_para)

        return updated_params, updated_graph_para

    def update_graph_para_buffer(self,graph_para):
        for name, tensor in graph_para.items():
            self._buffers[name] = tensor 

    def read_precomputed_pose_som(self): 

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
        ## get traj for target nodes
        dicts=[]

        # dict_dir=self.work_dir+"/out_dicts2" # has w2c and camera's K
        dict_dir=self.work_dir+"/out_dicts3" # has w2c and camera's K

        def count_out_dict_files(folder_path):
            # Use glob to find files matching the pattern
            file_pattern = os.path.join(folder_path, 'out_dict_*.pkl')
            files = glob.glob(file_pattern)
            return len(files)

        nt=count_out_dict_files(dict_dir)
        for t in range(nt):
            with open(f"{dict_dir}/out_dict_{t}.pkl", 'rb') as file:
                loaded_dict = pickle.load(file)
                dicts.append(loaded_dict)

        ids=np.arange(0,dicts[0]["means"].shape[0]).tolist()
        

        Transls=np.array([dicts[t]["means"][ids].cpu().numpy() for t in range(nt)]) #(t,|ids|,3)
        Quats_wxyz=np.array([dicts[t]["quats"][ids].cpu().numpy() for t in range(nt)]) # wxyz (t,|ids|,4)
        Quats_xyzw=Quats_wxyz[...,[1,2,3,0]] # xyzw
        Contribs=np.array([dicts[t]['contribs'].reshape(-1,3).cpu().numpy()[:,0][ids] for t in range(nt)]) #### 
        SE3s=np.concatenate([Transls,Quats_xyzw],axis=-1) # xyz xyzw (required by pypose)
        SE3s=torch.tensor(SE3s,dtype=torch.float32) #,device='cuda')
        Oris_xyzw=pp.LieTensor(SE3s[:,:,3:],ltype=pp.SO3_type)

        return Transls,Oris_xyzw,Contribs

    def read_precomputed_pose_graph(self): 
        file_path_parameters=self.work_dir+'/graph_model/v12_parameters9mee.pth'
        parameters=torch.load(file_path_parameters, weights_only=False)
        return parameters
    
    def get_means_quats_by_ts(self,ts):
        return self.Transls[ts],self.quat_activation(self.Oris_xyzw[ts])
    
    def get_optimizable_list(
        self,
        lr_transls: float = 1.6e-4,
        lr_oris_wxyz: float = 5e-2,
        lr_dists_key: float = 1.6e-4,
        lr_dists_oris_key: float = 5e-2,
        lr_weight_dist_key: float = 1e-4,
        lr_dists_nonkey: float = 1.6e-4,
        lr_dists_oris_nonkey: float = 5e-2,
        lr_weight_dist_nonkey: float = 1e-4,
        lr_dir_matrix_key: float = 1e-4,
        lr_weight_dir_key: float = 1e-4,
        lr_dir_matrix_nonkey: float = 1e-4,
        lr_weight_dir_nonkey: float = 1e-4,
        lr_weight_param_key: float = 1e-4,
        lr_src_xyz_1_nonkey: float = 1e-4,
        lr_src_xyzw_1_nonkey: float = 1e-4,
        lr_per_nonkeyGaussian_dweight_nonkey: float = 1e-4,
        lr_per_nonkeyGaussian_dxyz_dxyzw: float = 1e-4,
    ):
        ret = []

        def try_add(name, lr):
            if name in self.params:
                ret.append({"params": [self.params[name]], "lr": lr, "name": name})

        try_add("transls", lr_transls)
        try_add("oris_wxyz", lr_oris_wxyz)
        try_add("dists_key", lr_dists_key)
        try_add("dists_oris_key", lr_dists_oris_key)
        try_add("weight_dist_key", lr_weight_dist_key)
        try_add("dists_nonkey", lr_dists_nonkey)
        try_add("dists_oris_nonkey", lr_dists_oris_nonkey)
        try_add("weight_dist_nonkey", lr_weight_dist_nonkey)
        try_add("dir_matrix_key", lr_dir_matrix_key)
        try_add("weight_dir_key", lr_weight_dir_key)
        try_add("dir_matrix_nonkey", lr_dir_matrix_nonkey)
        try_add("weight_dir_nonkey", lr_weight_dir_nonkey)
        try_add("weight_param_key", lr_weight_param_key)
        try_add("src_xyz_1_nonkey", lr_src_xyz_1_nonkey)
        try_add("src_xyzw_1_nonkey", lr_src_xyzw_1_nonkey)
        try_add("per_nonkeyGaussian_dweight_nonkey", lr_per_nonkeyGaussian_dweight_nonkey)
        try_add("per_nonkeyGaussian_dxyz_dxyzw", lr_per_nonkeyGaussian_dxyz_dxyzw)

        return ret





def o3d_knn_p(pts2,pts, num_knn, flag_remove_self=False):
    indices = []
    sq_dists = []
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, np.float64))
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    for p in pts2:
        if not flag_remove_self:
            [n_num, i, d2] = pcd_tree.search_knn_vector_3d(p, num_knn)
            # d=np.array(d2)**0.5
            indices.append(i[0:])
            sq_dists.append(d2[0:])
        else:
            [n_num, i, d2] = pcd_tree.search_knn_vector_3d(p, num_knn + 1)
            # d=np.array(d2)**0.5
            indices.append(i[1:])
            sq_dists.append(d2[1:])
        if n_num < num_knn:
            print("Warning: not enough neighbors found")
            breakpoint()
    return np.array(sq_dists), np.array(indices)

    
def get_knn_edges(n_neighbors,Transls,Contribs,Transls2,Contribs2,contrib_threshold, flag_remove_self=False):

    nk=Transls.shape[1]
    nu=Transls2.shape[1]
    if torch.is_tensor(Transls):
        Transls=Transls.cpu()
    if torch.is_tensor(Transls2):
        Transls2=Transls2.cpu()
    if torch.is_tensor(Contribs):
        Contribs=Contribs.cpu().numpy()
    if torch.is_tensor(Contribs2):
        Contribs2=Contribs2.cpu().numpy()
    i_s_knn_others=[]
    sq_dists_others=[]
    contribs_max=Contribs.max(axis=0)
    pbar = tqdm(range(0, nu))
    for i in pbar:
        t_most_confident=np.argmax(Contribs2[:,i]) # np.argsort(Contribs2[:,i])[-1]
        # eff_map=Contribs[t_most_confident]>contrib_threshold # key points map at t_most_confident
        eff_map=Contribs[t_most_confident]>contrib_threshold*contribs_max # key points map at t_most_confident # TODO: CHECK
        pts=Transls[t_most_confident][eff_map] # confident key points at t_most_confident
        i_s=torch.arange(nk)[eff_map] # confident key points indices at t_most_confident
        pt2=Transls2[t_most_confident,i] # i-th other points at t_most_confident
        pts2=[pt2] # just one point
        sq_dists, indices = o3d_knn_p(pts2, pts, n_neighbors, flag_remove_self)
        sq_dists,indices= sq_dists[0],indices[0]
        i_s_knn=i_s[indices] # usage: ids[i_s_knn]
        i_s_knn_others.append(i_s_knn)
        sq_dists_others.append(torch.tensor(sq_dists))
        pbar.set_description(f"get_knn_edges nu: {i}")
    i_s_knn_others=torch.stack(i_s_knn_others) # [nu, n_neighbors]
    sq_dists_others=torch.stack(sq_dists_others) # [nu, n_neighbors]

    edges_others=torch.concatenate([torch.arange(nu).unsqueeze(1).repeat(1,n_neighbors).reshape(-1)[:,None],i_s_knn_others.reshape(-1)[:,None]],dim=-1)
    # edges_others.shape # nu*5
    return edges_others, sq_dists_others

# try to speed up
def get_knn_edges2(n_neighbors,Transls,Contribs,Transls2,Contribs2,contrib_threshold, flag_remove_self=False):
    nt=Transls.shape[0]
    nk=Transls.shape[1]
    nu=Transls2.shape[1]
    if torch.is_tensor(Transls):
        Transls=Transls.cpu()
    if torch.is_tensor(Transls2):
        Transls2=Transls2.cpu()
    if torch.is_tensor(Contribs):
        Contribs=Contribs.cpu().numpy()
    if torch.is_tensor(Contribs2):
        Contribs2=Contribs2.cpu().numpy()
    i_s_knn_others=[]
    sq_dists_others=[]
    # n_neighbors=5
    # contribs_max=Contribs.max(dim=0).values # along time dim # [nk] # if Contribs is torch tensor
    contribs_max=Contribs.max(axis=0)
    ts_most_confident=[]
    for i in range(nu):
        t_most_confident=np.argmax(Contribs2[:,i])
        ts_most_confident.append(t_most_confident)
    ts_most_confident=np.array(ts_most_confident)
    i_s_knn_others={}
    sq_dists_others={}
    # pbar = tqdm(range(0, nt))
    # for t in pbar:
    for t in range(0,nt):
        eff_map=Contribs[t]>contrib_threshold*contribs_max
        pts=Transls[t][eff_map]
        i_s=torch.arange(nk)[eff_map]
        if (ts_most_confident==t).sum()==0:
            continue
        target_i_s=torch.arange(nu)[ts_most_confident==t]
        pts2=Transls2[t][ts_most_confident==t]
        sq_dists, indices = o3d_knn_p(pts2, pts, n_neighbors, flag_remove_self)
        for i, (sq_dists, indices) in enumerate(zip(sq_dists, indices)):
            i_s_knn=i_s[indices]
            i_s_knn_others[target_i_s[i].item()]=i_s_knn
            sq_dists_others[target_i_s[i].item()]=torch.tensor(sq_dists)
        # pbar.set_description(f"get_knn_edges2 t: {t+1}/{nt}")
    i_s_knn_others=torch.stack([i_s_knn_others[i] for i in range(nu)]) # [nu, n_neighbors]
    sq_dists_others=torch.stack([sq_dists_others[i] for i in range(nu)]) # [nu, n_neighbors]


    edges_others=torch.concatenate([torch.arange(nu).unsqueeze(1).repeat(1,n_neighbors).reshape(-1)[:,None],i_s_knn_others.reshape(-1)[:,None]],dim=-1)
    # edges_others.shape # nu*5
    return edges_others, sq_dists_others

def get_knn_neighbors_by_contribs(transls,contribs,n_neighbors,contrib_threshold=0.5):
    # edges_local1,sq_dists_local1=get_knn_edges(n_neighbors,transls,contribs,transls,contribs,contrib_threshold,flag_remove_self=True)
    edges_local,sq_dists_local=get_knn_edges2(n_neighbors,transls,contribs,transls,contribs,contrib_threshold,flag_remove_self=True)
    return edges_local,sq_dists_local

def read_precomputed_pose_som(work_dir): # my code

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
    ## get traj for target nodes
    dicts=[]

    # dict_dir=work_dir+"/out_dicts2" # has w2c and camera's K
    dict_dir=work_dir+"/out_dicts3" # has w2c and camera's K

    def count_out_dict_files(folder_path):
        # Use glob to find files matching the pattern
        file_pattern = os.path.join(folder_path, 'out_dict_*.pkl')
        files = glob.glob(file_pattern)
        return len(files)

    nt=count_out_dict_files(dict_dir)
    for t in range(nt):
        with open(f"{dict_dir}/out_dict_{t}.pkl", 'rb') as file:
            loaded_dict = pickle.load(file)
            dicts.append(loaded_dict)


    ids=np.arange(0,dicts[t]["means"].shape[0]).tolist()

    i_target=0
    i_show=80
    Transls=np.array([dicts[t]["means"][ids].cpu().numpy() for t in range(nt)]) #(t,|ids|,3)
    Quats_wxyz=np.array([dicts[t]["quats"][ids].cpu().numpy() for t in range(nt)]) # wxyz (t,|ids|,4)
    Quats_xyzw=Quats_wxyz[...,[1,2,3,0]] # xyzw
    Contribs=np.array([dicts[t]['contribs'].reshape(-1,3).cpu().numpy()[:,0][ids] for t in range(nt)]) #
    Contribs_sort=np.array([get_order(Contribs[:,i]) for i in range(Contribs.shape[1])]).transpose(1,0)
    Vars=np.array([1/dicts[t]['contribs'].reshape(-1,3).cpu().numpy()[:,0][ids] for t in range(nt)])
    W2Cs=np.array([dicts[t]["w2c"].cpu().numpy() for t in range(nt)])
    SE3s=np.concatenate([Transls,Quats_xyzw],axis=-1) # xyz xyzw (required by pypose)
    SE3s=torch.tensor(SE3s,dtype=torch.float32) #,device='cuda')
    Oris_xyzw=pp.LieTensor(SE3s[:,:,3:],ltype=pp.SO3_type)
    Oris=pp.LieTensor(SE3s[:,:,3:],ltype=pp.SO3_type).Log()
    lie_tensor_SE3s=pp.LieTensor(SE3s, ltype=pp.SE3_type)
    vars=Vars[:,i_target]
    nk=len(ids) # num of key points
    np.histogram(vars[vars<1e0])

    return Transls,Oris_xyzw,Contribs#,Contribs_sort,Vars,W2Cs,ids,nk,lie_tensor_SE3s


def resort_ids(ids,ids2):
    # if ids is torch tensor
    if torch.is_tensor(ids):
        ids=ids.cpu().numpy()
    if torch.is_tensor(ids2):
        ids2=ids2.cpu().numpy()

    # if ids is numpy array
    if isinstance(ids, np.ndarray):
        ids=ids.tolist()
    if isinstance(ids2, np.ndarray):
        ids2=ids2.tolist()
    
    
    ids_key,ids_nonkey=[],[]
    i,j,k=0,0,0
    while True:
        n1=len(ids)
        n2=len(ids2)
        assert i+j==k
        if i<n1 and j<n2:
            if ids[i]<ids2[j]:
                ids_key.append(k)
                i+=1
            else:
                ids_nonkey.append(k)
                j+=1
        elif i<n1:
            ids_key.append(k)
            i+=1
        elif j<n2:
            ids_nonkey.append(k)
            j+=1
        else:
            break
        k+=1
    ids_key=np.array(ids_key)
    ids_nonkey=np.array(ids_nonkey)
    return ids_key,ids_nonkey