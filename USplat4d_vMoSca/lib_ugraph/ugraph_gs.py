import sys, os, os.path as osp
import roma
import torch
import torch.nn as nn
import torch.nn.functional as F
# from gsplat.rendering import rasterization
from torch import Tensor
import numpy as np
import functools
import gc

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ugraph_helper import Precompute_helper
from gs_utils.gs_optim_helper import prune_optimizer

class UGraphGaussian(nn.Module):
    def __init__(
        self,
        precompute: Precompute_helper | None =None,
        usage: str="train", # render_tracking
    ):
        super().__init__()

        self.precompute = precompute
        self.usage = usage

    
    # def compute_poses_all2(
    #     self, ts: torch.Tensor | None
    # ) -> tuple[torch.Tensor, torch.Tensor]:
    #     means, quats = self.get_means_quats_by_ts(ts)
    #     if self.has_bg:
    #         bg_means, bg_quats = self.compute_poses_bg()
    #         means = torch.cat(
    #             [means, bg_means[:, None].expand(-1, means.shape[1], -1)], dim=0
    #         ).contiguous()
    #         quats = torch.cat(
    #             [quats, bg_quats[:, None].expand(-1, means.shape[1], -1)], dim=0
    #         ).contiguous()
    #     return means, quats
    
    
    def get_means_quats_by_ts(self,ts: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        usage:
            means, quats = self.get_means_quats_by_ts(ts)
        """
        means = self.precompute.params['transls'][ts]
        quats = self.precompute.params['oris_wxyz'][ts]
        
        quats = F.normalize(quats, p=2, dim=-1)
        means=means.transpose(1,0)
        quats=quats.transpose(1,0)
        return means, quats

    # def get_interpolated_means_quats_by_ts(self,ts: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    #     device=self.precompute.params['transls'].device
    #     means = self.precompute.graph_para['transls_interpolation'][ts]
    #     quats = self.precompute.graph_para['oris_interpolation'][ts]
    #     means=means.to(device)
    #     quats=quats.to(device)
        
    #     quats = F.normalize(quats, p=2, dim=-1)
    #     means=means.transpose(1,0)
    #     quats=quats.transpose(1,0)
    #     print("use interpolated means and quats")
    #     return means, quats

    @staticmethod
    def init_from_state_dict(state_dict, graph_dir, device, prefix="",usage="train"): # prefix="ugraph_model." is not true for ugraph pth
    
        if usage=="train":
            # precompute is None if parameters are not incomplete.
            print("Tring to init_from_state_dict...")
            precompute=Precompute_helper.init_from_state_dict(
                state_dict, prefix=f"{prefix}precompute."
            )
            if precompute==None: # TODO
                print("precompute is None. Start create precompute from graph_dir...")
                if graph_dir is None or graph_dir == '':
                    pass  # breakpoint() disabled for background runs
                    raise ValueError("graph_dir is None or empty")
                # precompute=Precompute_helper() # last.ckpt 
                precompute=Precompute_helper(graph_dir=graph_dir)
                # precompute=Precompute_helper.init_from_ini_ckpt() # ini.ckpt

            model=UGraphGaussian(precompute)
            model = model.to(device)
            if model.precompute is not None:
                model.precompute.set_para_device(device)
            return model
        
        else:
            pass  # breakpoint() disabled for background runs
            return None
        
    def get_optimizable_list(
        self,
        #precompute
        **precompute_lr_kwargs
    ):
        # test
        # precompute_lr_kwargs["lr_transls"]
        # for name, lr in precompute_lr_kwargs.items():
            # print(f"{name} has learning rate {lr}")

        l = []
        # if lr_p is not None:
        #     l.append({"params": [self._xyz], "lr": lr_p, "name": "xyz"})
        
        l = l + self.precompute.get_optimizable_list(**precompute_lr_kwargs)
        return l
    

    def _prune_points(self, optimizers_ugraph, should_fg_cull):
        valid_points_mask = ~should_fg_cull
        assert should_fg_cull.shape[0] == self.precompute.params['transls'].shape[1]
        # if valid_points_mask.all():
        #     return
        
        precompute_param_map, _, should_fg_cull_nonkey = self.precompute.cull_params(should_cull=should_fg_cull)
        for param_name, new_params in precompute_param_map.items():
            full_param_name = f"precompute.params.{param_name}" # CHECK
            optimizer = optimizers_ugraph[full_param_name]
            self.remove_from_optim_graph(optimizer, [new_params], should_fg_cull, should_fg_cull_nonkey)

        # optimizable_tensors = prune_optimizer(
        #     optimizer,
        #     valid_points_mask,
        #     # exclude_names=self.op_update_exclude,
        # )

        return
    
    def _densify_points(self, optimizers_ugraph, should_fg_densify):
        # densify positions and rotations # my code
        num_fg_splits=0 # test
        num_fg_dups = int(should_fg_densify.sum().item())
        precompute_param_map, _ = self.precompute.densify_params(should_split=None, should_dup=should_fg_densify)
        for param_name, new_params in precompute_param_map.items():
            full_param_name = f"precompute.params.{param_name}" # CHECK
            optimizer = optimizers_ugraph[full_param_name]
            self.dup_in_optim_graph(optimizer, [new_params], should_dup=should_fg_densify, num_dups=num_fg_splits * 2 +num_fg_dups)

    def dup_in_optim_graph(self, optimizer, new_params: list, should_dup: torch.Tensor, num_dups: int):
        assert len(optimizer.param_groups) == len(new_params)
        full_param_name = optimizer.param_groups[0]["name"]
        for i, p_new in enumerate(new_params):
            old_params = optimizer.param_groups[i]["params"][0]
            param_state = optimizer.state[old_params]
            if len(param_state) == 0:
                return
            for key in param_state:
                if key == "step":
                    continue
                p = param_state[key]
                # param_state[key] = torch.cat(
                #     [p[~should_dup], p.new_zeros(num_dups, *p.shape[1:])],
                #     dim=0,
                # )
                if full_param_name in ['precompute.params.transls','precompute.params.oris_wxyz']: # (nt,ng,3) or (nt,ng,4)
                    param_state[key] = torch.cat(
                        [p, p.new_zeros(p.shape[0],num_dups, *p.shape[2:])],
                        dim=1,
                    )
                elif "_nonkey" in full_param_name: # (nu, ...) or (nu)
                    param_state[key] = torch.cat(
                        [p, p.new_zeros(num_dups, *p.shape[1:])],
                        dim=0,
                    )
                elif "_key" in full_param_name: # (nk, ...) or (nk)
                    continue
                else:
                    pass  # breakpoint() disabled for background runs
            optimizer.state[old_params] = {}  # Clear before deletion
            del optimizer.state[old_params]
            optimizer.state[p_new] = param_state
            optimizer.param_groups[i]["params"] = [p_new]
            del old_params
            gc.collect()
            torch.cuda.empty_cache()
    
    def remove_from_optim_graph(self,optimizer, new_params: list, _should_cull: torch.Tensor, _should_cull_nonkey: torch.Tensor):
        assert len(optimizer.param_groups) == len(new_params)
        full_param_name = optimizer.param_groups[0]["name"]
        for i, p_new in enumerate(new_params):
            old_params = optimizer.param_groups[i]["params"][0]
            param_state = optimizer.state[old_params]
            if len(param_state) == 0:
                return
            for key in param_state:
                if key == "step":
                    continue
                if full_param_name in ['precompute.params.transls','precompute.params.oris_wxyz']:
                    param_state[key] = param_state[key][:,~_should_cull]
                elif "_nonkey" in full_param_name:
                    param_state[key] = param_state[key][~_should_cull_nonkey]
                elif "_key" in full_param_name:
                    continue
                else:
                    pass  # breakpoint() disabled for background runs
            optimizer.state[old_params] = {}  # Clear before deletion
            del optimizer.state[old_params]
            optimizer.state[p_new] = param_state
            optimizer.param_groups[i]["params"] = [p_new]
            del old_params
            gc.collect()
            torch.cuda.empty_cache()
    
    def configure_optimizers(self,lr_dict):
        def _exponential_decay(step, *, lr_init, lr_final):
            t = np.clip(step / self.optim_cfg.max_steps, 0.0, 1.0)
            lr = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
            return lr / lr_init

        # lr_dict = asdict(self.lr_cfg)
        optimizers = {}
        schedulers = {}
        # named parameters will be [part].params.[field]
        # e.g. fg.params.means
        # lr config is a nested dict for each fg/bg part
        for name, params in self.named_parameters():
            part, _, field = name.split(".")
            # lr = lr_dict[part][field]
            lr = lr_dict['lr_'+field]
            
            optim = torch.optim.Adam([{"params": params, "lr": lr, "name": name}])

            if "scales" in name: # will not come here
                fnc = functools.partial(_exponential_decay, lr_final=0.1 * lr)
            else:
                fnc = lambda _, **__: 1.0 # TODO

            optimizers[name] = optim
            schedulers[name] = torch.optim.lr_scheduler.LambdaLR(
                optim, functools.partial(fnc, lr_init=lr)
            )
        return optimizers, schedulers