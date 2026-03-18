from typing import Dict, List, TypedDict

import pdb
import time
import math
import wandb
import torch
import imageio
import numpy as np
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

from einops import rearrange
from ..Misc import Const
from ..Misc import Logger as log
from ..Misc.BBox import BoundingBox
from torch.utils.checkpoint import checkpoint

from diffusers.models.attention_processor import Attention as CrossAttention

from TrailBlazer.Pipeline.Loss_factory import l2_loss
from bin.utils.misc import (convert_bbox_to_image_space)

from bin.utils.plot_helpers import (convert_to_numpy, plot2chk_image, plot_opt_spatial_maps, 
                                    plot_single_spatial_out_map,
                                    plot_opt_temporal_maps, 
                                    draw_bbox, save_cv2_image,
                                    plot_bboxes)

from TrailBlazer.CrossAttn.Utils import create_attention_video, create_diff_bbox_heatmap, create_diff_bbox_heatmap_gaussian_2, \
            compute_bbox_LRTB_HW, compute_sigma, get_patch, get_patch_old, reshape_fortran, gaussian_2d, get_edge_coords, \
                process_temporal2, process_temporal1, get_layer_id_info



INJECTION_SCALE = 1.0
KERNEL_DIVISION = 3.

class BundleType(TypedDict):
    selected_inds: List[int]  # the 1-indexed indices of a subject
    trailing_inds: List[int]  # the 1-indexed indices of trailings
    bbox: List[
        float
    ]  # four floats to determine the bounding box [left, right, top, bottom]




class CrossAttnProcessorBase:

    MAX_LEN_CLIP_TOKENS = 77
    DEVICE = "cuda"

    def __init__(self, bundle, is_text2vidzero=False):

        self.prompt = bundle["prompt_base"]
        self.bundle = bundle
        base_prompt = self.prompt.split(";")[0]
        self.len_prompt = len(base_prompt.split(" "))
        self.prompt_len = len(self.prompt.split(" "))
        self.timestep = None 
        self.opt_id = None 
        self.n_opt_iterations = None 
        self.use_dd = False
        self.repeat_opt = False
        self.name = ""
        self.loss_dict= {}
        self.use_trg_unscaled = False
        self.chosen_temp_block = ""
        self.temp_edit_at_low_res = False
        self.edit_before_softmax = False
        self.use_dd_temporal = False
        self.unet_chunk_size = 2

        self.use_grad_chkpt = False
        self.use_reentrant = True
        self._cross_attention_map = None
        self._cross_attention_map_wk = None
        self._cross_attention_map_str = None
        self._loss = None
        self._parameters = None
        self.is_text2vidzero = is_text2vidzero
        bbox = None
        self.wandb_log = False
        self.output_opt_viz_path = ""
   
    # note: @property manages attributes so you can read/modify without changing 
    # existing code making it backward compatible
    @property
    def cross_attention_map(self):
        return self._cross_attention_map # reads attribute
    
    @property
    def cross_attention_map_wk(self):
        return self._cross_attention_map_wk

    @property
    def cross_attention_map_str(self):
        return self._cross_attention_map_str

    @property
    def loss(self):
        return self._loss

    @property
    def parameters(self):
        if type(self._parameters) == type(None):
            log.warn("No parameters being initialized. Be cautious!")
        return self._parameters
    
    def __call__(
        self,
        attn: CrossAttention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        cross_attention_kwargs=None,
    ):  
 

        # TODO:
        # pass `use_grad_chkpt` and `use_reentrant` as an argument
        # use_grad_chkpt = True
        # use_reentrant = True

        # self.vis = True
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        #print("====================")

        if self.use_grad_chkpt:
            query = checkpoint(attn.to_q, hidden_states, use_reentrant=self.use_reentrant)
        else:
            query = attn.to_q(hidden_states)
        
        is_cross_attention = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        # elif attn.cross_attention_norm:
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)

        # note: likely that 240 for prompt or 65_536 for cross-frame come from this linear projection

        if self.use_grad_chkpt:
            key = checkpoint(attn.to_k, encoder_hidden_states, use_reentrant=self.use_reentrant)
            value = checkpoint(attn.to_v, encoder_hidden_states, use_reentrant=self.use_reentrant)
        else:
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
        

        # if key.shape[-2]!=77:

        def rearrange_3(tensor, f):
            F, D, C = tensor.size()
            return torch.reshape(tensor, (F // f, f, D, C))

        def rearrange_4(tensor):
            B, F, D, C = tensor.size()
            return torch.reshape(tensor, (B * F, D, C))
        
        
        # Cross Frame Attention
        if not is_cross_attention and self.is_text2vidzero:
            pdb.set_trace()

            video_length = key.size()[0] // 2
            first_frame_index = [0] * video_length

            # rearrange keys to have batch and frames in the 1st and 2nd dims respectively
            key = rearrange_3(key, video_length)
            key = key[:, first_frame_index]
            # rearrange values to have batch and frames in the 1st and 2nd dims respectively
            value = rearrange_3(value, video_length)
            value = value[:, first_frame_index]

            # rearrange back to original shape
            key = rearrange_4(key)
            value = rearrange_4(value)

        # if key.shape[-2]!=77:
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # Cross attention map
        attention_probs = attn.get_attention_scores(query, key) # note: temporal 65536, 24, 24 | spatial 240, 4096, 77
        device = attention_probs.device


        # ---- NEW 
        if self.bundle.get("bkwd_guidance"):
            if self.use_dd and self.use_dd_temporal:
                n_input_latents = 1
            else:
                n_input_latents = 2
        else:
            n_input_latents = 2
        
        # temporal cross-frame: 65_536, 24, 24 -> n=32_768 | spatial-cross_on_prompt: 240, 4096, 77 -> n=120
        # NOTE: why do we split into two?
        n = attention_probs.shape[0] // n_input_latents 

        token_inds = self.bundle.get("token_inds")

        new_attention_probs_wk = None
        all_tokens_inds = None

        # all assume H==W, except one (576x320)
        temporal_resolution_meta = {
                            576: 46_080, # 576x320
                            512: 65_536, # transformer_in module
                            504: 63_504,
                            480: 57_600,
                            448: 50_176,
                            416: 43_264,
                            384: 36_864,
                            320: 25_600,
                            304: 23_104,
                            256: 16_384,
                            160: 6_400,
                            128: 4_096
                                }

        # TODO: remove backward guidance code?
        if self.bundle.get("bkwd_guidance"):
            if self.use_dd and self.use_dd_temporal:
                for img_size in temporal_resolution_meta:
                    temporal_resolution_meta[img_size] = temporal_resolution_meta[img_size] // 2
        
        temporal_res_meta_size = temporal_resolution_meta.get(self.bundle['width'])
        if temporal_res_meta_size == None:
            raise ValueError(f"No temporal meta size provided for resolution {self.bundle['width']}")

        # TODO: relocate
        def get_temp_block_name(chosen_temp_block):
            if chosen_temp_block=="mid_block_temp":
                chosen_temporal_block = "mid_block.temp_attentions.0.transformer_blocks.0.attn2"
            elif chosen_temp_block=="transformer_in":
                chosen_temporal_block = "transformer_in.transformer_blocks.0.attn2"
            elif chosen_temp_block=='None':
                chosen_temporal_block = chosen_temp_block
            else:
                raise NotImplementedError
            return chosen_temporal_block

        if attention_probs.shape[-1] == self.num_frames:
            
            if temporal_res_meta_size==65_536:
                assert self.bundle['width'] == 512 and self.bundle['height'] == 512, 'temporal_res_meta_size 65_536 is associated with 512x512 resolution'
                # 512x512 resolution
                chosen_temporal_block = get_temp_block_name(self.chosen_temp_block)
            else:
                # non_512x512 resolution
                assert self.bundle['width'] != 512 and self.bundle['height'] != 512, "`no temporal editing` is only expected at non-512x512 resolution"

                if self.temp_edit_at_low_res:
                    chosen_temporal_block = get_temp_block_name(self.chosen_temp_block)
                    # if self.opt_id==0 and self.timestep==0:
                    #     log.info(f"Running TEMPORAL EDITING at lower resolution {self.bundle['width']}")
                else:
                    # no temporal editing at low-res, as per trailblazer original
                    # ref: https://github.com/hohonu-vicml/TrailBlazer/blob/de2696ef50537a473ab5afc57de89a5edc97c00b/TrailBlazer/CrossAttn/BaseProc.py#L163
                    chosen_temporal_block = None
                    
            

        if attention_probs.shape[-1] == CrossAttnProcessorBase.MAX_LEN_CLIP_TOKENS:  # prompt attention, 77 tokens
            gcd = np.gcd(Const.DEFAULT_HEIGHT, Const.DEFAULT_WIDTH)
            height_multiplier = Const.DEFAULT_HEIGHT / gcd
            width_multiplier = Const.DEFAULT_WIDTH / gcd
            factor = attention_probs.shape[1] // (height_multiplier * width_multiplier)
            factor = int(np.sqrt(factor))
            dim_y = int(factor * height_multiplier)
            dim_x = int(factor * width_multiplier)

            # NEW      
            # -------------------------------------------------------------
            # for spatial editing and/or non-editing steps (if required)
            token_inds = self.bundle.get("token_inds") # note: focus token idx e.g 2
            trailing_length = self.bundle["trailblazer"]["trailing_length"]
            trailing_inds = list(range(self.len_prompt + 1, self.len_prompt + trailing_length + 1))
            all_tokens_inds = list(set(token_inds).union(set(trailing_inds)))
            # -------------------------------------------------------------
        
            #
            attn_type='spatial'

            # for spatial only (because bottom indexing can lead to an empty tensor in bkwd_guidance)
            if self.bundle.get("bkwd_guidance"):
                sub_n = 0 # selects all
            else:
                sub_n = n # selects half, as latent input was doubled (trailblazer)
            
            # Select which cross-attention layer(s) to visualize.
            # User controls via bundle key `vis_layer` (set by CLI flag --vis_layer).
            # Format: comma-separated list of short ids, e.g. "up.2.2.0" or "up.2.2.0,down.0.0.0".
            layer_id_to_name = {
                "down.0.0.0": "down_blocks.0.attentions.0.transformer_blocks.0.attn2",
                "up.2.2.0": "up_blocks.2.attentions.2.transformer_blocks.0.attn2",
            }

            vis_layer = self.bundle.get("vis_layer", "up.2.2.0")
            requested = [s.strip() for s in str(vis_layer).split(",") if s.strip()]
            if not requested:
                raise ValueError("vis_layer is empty; expected e.g. 'up.2.2.0'")

            vis_dict = {}
            for lid in requested:
                if lid not in layer_id_to_name:
                    raise ValueError(f"Unknown vis_layer '{lid}'. Valid: {sorted(layer_id_to_name)}")
                vis_dict[layer_id_to_name[lid]] = lid

            if self.use_dd:

                attention_probs_clone = attention_probs.clone()
                bottom_attention_copy = attention_probs_clone.reshape(attention_probs_clone.shape[0], dim_y, dim_x, attention_probs_clone.shape[-1])[sub_n:] #.clone()
                # note: wherever CrossAttention/attn2 is True (early steps), core DD injection is done (no optimization)
                
                Attn_reshaped = attention_probs_clone.reshape(attention_probs_clone.shape[0], dim_y, dim_x, attention_probs_clone.shape[-1])
                
                'NOTE: a copied version is made internally in dd_core function, and the modified copy is returned'
                # attention_probs_4d contains both prompt attn maps and trailing maps
                'modified_bottom_attention_copy_str is edited by both strength and weaken map'
                
                modified_bottom_attention_copy_wk, modified_bottom_attention_copy_str, diff_bboxmap_str, unscaled_diff_bboxmap_str, unscaled_weaken_map = self.dd_core(bottom_attention_copy, dim_x, dim_y, 
                                                                                                     use_spatial=self.use_dd, timestep=self.timestep)

                
                self._sub_attn = bottom_attention_copy[..., all_tokens_inds]
                # for out-in loss
                self._modified_sub_attn_wk = modified_bottom_attention_copy_wk[..., all_tokens_inds]
                self._modified_sub_attn_str = modified_bottom_attention_copy_str[..., all_tokens_inds]

                # for max cross loss - 
                # NOTE: 
                # 1) edit occurs sequentially after each cross-attention layer, maximize loss occurs at once at the end, for all layers 
                # 2) input attention is edited into next layer, the following output attention is maximized
                # 3) TO BE CLEAR - the `first` raw attention i.e (bottom_attention_copy) is considered an output attention - hence also maximized
                # 4) maximum loss is applied on different degrees of localized attention/granularity
                self._masked_map_wk_values = bottom_attention_copy[..., all_tokens_inds] * unscaled_weaken_map[..., all_tokens_inds]
                self._masked_map_str_values = bottom_attention_copy[..., all_tokens_inds] * unscaled_diff_bboxmap_str[..., all_tokens_inds]


                # set overlay map to default None
                self._overlay_attn_maps = None
                
                # TODO: add normalization. Sum is too large for fp16, 
                trg = unscaled_diff_bboxmap_str[..., all_tokens_inds] if self.use_trg_unscaled else diff_bboxmap_str[..., all_tokens_inds]
                # trg = diff_bboxmap_str[..., all_tokens_inds]
                
                pred = bottom_attention_copy[..., all_tokens_inds]

                if self.loss_dict['diff_loss']:
                    _diff_loss = l2_loss(trg, pred, use_mean=True)
                else:
                    _diff_loss = torch.tensor([0])


                # --------------------------------
                if self.bundle['vis_maps'] or self.bundle['vis_opt_and_denoise_maps']:
                    if self.name in vis_dict:
                        """
                        plot_single_spatial_out_map(bottom_attention_copy.detach().cpu(), 
                                                    n_frames=len(self.bbox_per_frame),
                                                    opt_id=self.opt_id, 
                                                    timestep=self.timestep, 
                                                    all_tokens_inds=all_tokens_inds, 
                                                    n_opt_iterations=self.n_opt_iterations, 
                                                    output_opt_viz_path=self.output_opt_viz_path,
                                                    # save plot image media either online/local 
                                                    wandb_log=self.wandb_log, 
                                                    plot_local=not self.wandb_log, 
                                                    mini_label=vis_dict[f'{self.name}'])
                        """

                        
                        # 
                        # """
                        # ATTENTION EDIT VIS FOR DEMO
                        """
                        plot_single_spatial_out_map(bottom_attention_copy.detach().cpu(), 
                                                    n_frames=len(self.bbox_per_frame),
                                                    opt_id=self.opt_id, 
                                                    timestep=self.timestep, 
                                                    all_tokens_inds=all_tokens_inds, 
                                                    n_opt_iterations=self.n_opt_iterations, 
                                                    output_opt_viz_path=self.output_opt_viz_path,
                                                    # save plot image media either online/local 
                                                    wandb_log=self.wandb_log, 
                                                    plot_local=not self.wandb_log, 
                                                    mini_label=vis_dict[f'{self.name}'])
                        """

                        
                        # 
                        # """
                        # ATTENTION EDIT VIS FOR DEMO
                        """
                        plot_single_spatial_out_map(bottom_attention_copy.detach().cpu(), 
                                                    n_frames=len(self.bbox_per_frame),
                                                    opt_id=self.opt_id, 
                                                    timestep=self.timestep, 
                                                    all_tokens_inds=all_tokens_inds, 
                                                    n_opt_iterations=self.n_opt_iterations, 
                                                    output_opt_viz_path=self.output_opt_viz_path,
                                                    # save plot image media either online/local 
                                                    wandb_log=self.wandb_log, 
                                                    plot_local=not self.wandb_log, 
                                                    mini_label=vis_dict[f'{self.name}'])
                        """

                        
                        # 
                        # """
                        # ATTENTION EDIT VIS FOR DEMO
                        plot_opt_spatial_maps(bottom_attention_copy.detach().cpu(), 
                                              unscaled_diff_bboxmap_str.detach().cpu(), 
                                              modified_bottom_attention_copy_str.detach().cpu(), 
                                              modified_bottom_attention_copy_wk.detach().cpu(),
                                             scale_box_map = diff_bboxmap_str,
                                        opt_id=self.opt_id, timestep=self.timestep, all_tokens_inds=all_tokens_inds, 
                                        n_opt_iterations=self.n_opt_iterations, output_opt_viz_path=self.output_opt_viz_path,
                                        wandb_log=self.wandb_log, plot_local=not self.wandb_log, mini_label=vis_dict[f'{self.name}'])

                # --------------------------------
                # cache attention maps for post-opt frames overlay
                # TODO: put into function

                if self.bundle['overlay_maps']:
                    
                    if self.name in vis_dict:
                        # save overlay at last editing and optimization step
                        if (self.timestep == self.bundle['trailblazer']['num_dd_spatial_steps']-1) and (self.opt_id == self.n_opt_iterations-1):
                        # if (self.timestep in np.arange(self.bundle['trailblazer']['num_dd_spatial_steps'])) and (self.opt_id == self.n_opt_iterations-1):
                    
                            n_channels = bottom_attention_copy.detach().cpu().shape[0]
                            window_size = n_channels // self.num_frames
                            assert n_channels % self.num_frames == 0, 'number of channels must be perfectly divisible by number of frames'

                            overlay_maps = []
                            for ir, range_id in enumerate(range(0, n_channels, window_size)):

                                first_token = 0
                                frame_avg_attn_map = bottom_attention_copy[range_id : range_id + window_size, :, :, all_tokens_inds][..., first_token].detach().cpu().mean(dim=0)
                                overlay_maps.append(frame_avg_attn_map)
                                
                            overlay_attn_maps = torch.stack(overlay_maps)
                            self._overlay_attn_maps = overlay_attn_maps
                        
                # ----------------------

                def process_spatial_dd(modified_bottom_attention_copy, attention_probs, dim_x, dim_y, n, device):
                    # modified_bottom_attention_copy = self.dd_core(bottom_attention_copy, dim_x, dim_y, use_spatial=self.use_dd)
                    modified_bottom_attention_copy = modified_bottom_attention_copy.reshape(modified_bottom_attention_copy.shape[0], dim_y * dim_x, modified_bottom_attention_copy.shape[-1])
                    
                    # differentiable edit operations    
                    mask = torch.ones_like(attention_probs)
                    if n==0:
                        assert self.bundle.get("bkwd_guidance"), 'whole map is modified only in bkwd_guidance and when latent is a single input'
                        new_attention_probs = modified_bottom_attention_copy
                        return new_attention_probs
                    
                    mask[n:] = 0
                    top_part_0s = torch.zeros_like(attention_probs[:n])
                    assert modified_bottom_attention_copy.shape[0] == n, f'modified copy needs correct shape {n}'
                    new_attention_probs = (attention_probs * mask) + torch.cat([top_part_0s, modified_bottom_attention_copy]).to(device)
                    return new_attention_probs
                

                # strengthen map goes fwd to next layer
                new_attention_probs_wk = process_spatial_dd(modified_bottom_attention_copy_wk, attention_probs, dim_x, dim_y, sub_n, device)
                new_attention_probs_str = process_spatial_dd(modified_bottom_attention_copy_str, attention_probs, dim_x, dim_y, sub_n, device)

                if self.bundle['fwd_edit_map']:
                    new_attention_probs = new_attention_probs_str
                else:
                    new_attention_probs = attention_probs # dropped clone
                

            else:
                # NOTE: its either your editing the attention maps or not doing anything to it
                new_attention_probs = attention_probs
                _diff_loss = 0

                self._sub_attn = None
                self._modified_sub_attn_wk = None
                self._modified_sub_attn_str = None
                self._masked_map_wk_values = None
                self._masked_map_str_values = None
                self._overlay_attn_maps = None

                # note: visualize spatial non-editing steps (at same token idx)

                # --------------------------------
                if self.bundle['vis_opt_and_denoise_maps']:
                    if self.name in vis_dict:
                        new_bottom_attention_copy = new_attention_probs.reshape(new_attention_probs.shape[0], dim_y, dim_x, new_attention_probs.shape[-1])[sub_n:] #.clone()
                        plot_single_spatial_out_map(new_bottom_attention_copy.detach().cpu(), 
                                                    n_frames=len(self.bbox_per_frame),
                                                    opt_id=self.opt_id, 
                                                    timestep=self.timestep, 
                                                    all_tokens_inds=all_tokens_inds, 
                                                    n_opt_iterations=self.n_opt_iterations, 
                                                    output_opt_viz_path=self.output_opt_viz_path,
                                                    wandb_log=self.wandb_log, 
                                                    plot_local=not self.wandb_log, 
                                                    mini_label=vis_dict[f'{self.name}'])
                        
                        # plot_opt_spatial_maps(bottom_attention_copy.detach().cpu(), 
                        #                       unscaled_diff_bboxmap_str.detach().cpu(), 
                        #                       modified_bottom_attention_copy_str.detach().cpu(), 
                        #                       modified_bottom_attention_copy_wk.detach().cpu(),
                        #                      scale_box_map = diff_bboxmap_str,
                        #                 opt_id=self.opt_id, timestep=self.timestep, all_tokens_inds=all_tokens_inds, 
                        #                 n_opt_iterations=self.n_opt_iterations, output_opt_viz_path=self.output_opt_viz_path,
                        #                 # save plot image media either online/local 
                        #                 wandb_log=self.wandb_log, plot_local=not self.wandb_log, mini_label=vis_dict[f'{self.name}'])
                # """
                # --------------------------------


            # note: only `[sub_n:] - [2nd part]` selected
            self._all_tokens_inds = all_tokens_inds
            self._loss = _diff_loss

        elif attention_probs.shape[-1] == self.num_frames and self.name==chosen_temporal_block:

            if chosen_temporal_block == "transformer_in.transformer_blocks.0.attn2":
                assert attention_probs.shape[0]==temporal_res_meta_size, 'input temporal_res_meta_size is different.'

            if self.opt_id==0 and self.timestep==0 and self.bundle['width'] < 512:
                log.info(f"Running TEMPORAL EDITING at lower resolution {self.bundle['width']}")

            # dim = int(np.sqrt(attention_probs.shape[0] // (2 * attn.heads)))
            gcd = np.gcd(Const.DEFAULT_HEIGHT, Const.DEFAULT_WIDTH)
            height_multiplier = Const.DEFAULT_HEIGHT / gcd
            width_multiplier = Const.DEFAULT_WIDTH / gcd
            factor = (attention_probs.shape[0] // (n_input_latents * attn.heads)) // (height_multiplier * width_multiplier)
            factor = int(np.sqrt(factor))
            dim_y = int(factor * height_multiplier)
            dim_x = int(factor * width_multiplier)
            
            #
            attn_type='temporal'

            if self.use_dd_temporal:

            # '''
                def temporal_doit(origin_attn):
                    _diff_loss = None

                    temporal_attn = reshape_fortran(
                        origin_attn,
                        (attn.heads, dim_y, dim_x, self.num_frames, self.num_frames),
                    )
                    temporal_attn = torch.transpose(temporal_attn, 1, 2)

                    # note: wherever CrossAttention/attn2 is True (early steps), core DD injection is done (no optimization)
                    # 8*, 64, 64, 24, 24 - size 5 | * means attention heads
                    'NOTE: a copied version is made internally in dd_core function, and the modified copy is returned'

                    modified_copy_temporal_attn_wk, modified_copy_temporal_attn_str, diff_bboxmap_str, unscaled_diff_bboxmap_str, unscaled_weaken_map = self.dd_core(temporal_attn, dim_x, dim_y, 
                                                                                                use_temporal=self.use_dd_temporal, timestep=self.timestep)
                    
                    trg = unscaled_diff_bboxmap_str if self.use_trg_unscaled else diff_bboxmap_str
                    
                    if self.loss_dict['diff_loss']:
                        _diff_loss = l2_loss(trg, temporal_attn, use_mean=True)
                    else:
                        _diff_loss = torch.tensor([0])
                    
                    self._sub_attn = temporal_attn
                    self._modified_sub_attn_wk = modified_copy_temporal_attn_wk
                    self._modified_sub_attn_str = modified_copy_temporal_attn_str

                    # for max cross loss
                    self._masked_map_wk_values = temporal_attn * unscaled_weaken_map
                    self._masked_map_str_values = temporal_attn * unscaled_diff_bboxmap_str
 
                    # set overlay map to default None
                    self._overlay_attn_maps = None

                    # --------------------------------
                    if self.bundle['vis_maps']:
                        temp_vis_dict = {"mid_block.temp_attentions.0.transformer_blocks.0.attn2": "mid_block_temp", 
                                        "transformer_in.transformer_blocks.0.attn2": "transformer_in"}
                        
                        if self.name in temp_vis_dict:
                           
                            plot_opt_temporal_maps(temporal_attn.detach().cpu(), 
                                                   unscaled_diff_bboxmap_str.detach().cpu(), 
                                                   modified_copy_temporal_attn_str.detach().cpu(), 
                                            modified_copy_temporal_attn_wk.detach().cpu(), 
                                            opt_id=self.opt_id, timestep=self.timestep,
                                            n_opt_iterations=self.n_opt_iterations, output_opt_viz_path=self.output_opt_viz_path,
                                            wandb_log=self.wandb_log, plot_local=not self.wandb_log, mini_label=temp_vis_dict[f'{self.name}'])
                    # --------------------------------

                    modified_copy_temporal_attn_wk = process_temporal1(modified_copy_temporal_attn_wk, dim_x=dim_x, dim_y=dim_y, num_frames=self.num_frames, attn_heads=attn.heads)
                    modified_copy_temporal_attn_str = process_temporal1(modified_copy_temporal_attn_str, dim_x=dim_x, dim_y=dim_y, num_frames=self.num_frames, attn_heads=attn.heads)
                        
                    return modified_copy_temporal_attn_wk, modified_copy_temporal_attn_str, _diff_loss

                # NOTE: So null text embedding for classification free guidance
                # doesn't really help?
                # NOTE: da note - attention_probs[:n] contains both prompt attn maps and trailing maps
                # DEBUG:  temporal_attn = reshape_fortran(attention_probs[:n], (attn.heads, dim_y, dim_x, self.num_frames, self.num_frames))

                modified_copy_temp_attn_wk, modified_copy_temp_attn_str, _diff_loss = temporal_doit(attention_probs[:n].clone())

                # differentiable edit operation
                # TODO: rename?
                new_attention_probs_wk = process_temporal2(modified_copy_temp_attn_wk, attention_probs=attention_probs, n=n, device=device)
                new_attention_probs_str = process_temporal2(modified_copy_temp_attn_str, attention_probs=attention_probs, n=n, device=device)
                
                if self.bundle['fwd_edit_map']:
                    new_attention_probs = new_attention_probs_str
                else:
                    new_attention_probs = attention_probs # dropped clone

            else:
                # NOTE: its either your editing the attention maps or not doing anything to it
                new_attention_probs = attention_probs
                _diff_loss = 0

                self._sub_attn = None
                self._modified_sub_attn_wk = None
                self._modified_sub_attn_str = None
                self._masked_map_wk_values = None
                self._masked_map_str_values = None
                self._overlay_attn_maps = None


            self._all_tokens_inds = None
            self._loss = _diff_loss
 

        else:
            new_attention_probs = attention_probs
            
            self._sub_attn = None
            self._modified_sub_attn_wk = None
            self._modified_sub_attn_str = None
            self._masked_map_wk_values = None
            self._masked_map_str_values = None
            self._overlay_attn_maps = None

        new_attention_probs_abs = torch.abs(new_attention_probs)
        # e.g 240, 1600, 77


        # -------------------------------

        hidden_states = torch.bmm(new_attention_probs_abs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        if self.use_grad_chkpt:
            hidden_states = (checkpoint(attn.to_out[0], hidden_states, use_reentrant=self.use_reentrant))
        else:
            hidden_states = attn.to_out[0](hidden_states)
            
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        
        # note: conventional return output, as per original diffuser's code (https://github.com/huggingface/diffusers/blob/91ddd2a25b848df0fa1262d4f1cd98c7ccb87750/src/diffusers/models/attention.py#L284)
        assert hidden_states.is_cuda, 'hidden_states needs to be on GPU'
        return hidden_states

    @abstractmethod
    def dd_core(self):
        # note: abstractmethod: subclasses of CrossAttnProcessorBase must implement dd_core
        """All DD variants implement this function"""
        pass


    @staticmethod
    def localized_weight_map(attention_probs_4d, token_inds, bbox_per_frame, dim_x, dim_y, sigma_strength, 
                             clip_box_values = False, scale_local_foreground=False, local_scale=1,
                             box_with_gauss=False, gauss_only=False, use_bkgd_zero=False, minimize_bkgd=False,
                             use_high_box_only=False, normalize_gauss=True, normalize_mask=True,
                             allow_edge_margin=False, edge_margin=2):
  
        # Using Gaussian 2d distribution to generate weight map (same size as attention argument).
        atten_dtype = attention_probs_4d.dtype
        weight_map = []
        if minimize_bkgd:
            weaken_map = []
        n_c, map_h, map_w, n_tokens = attention_probs_4d.shape

        # explains every 5 channel per frame in e.g 120 total channels
        frame_size = attention_probs_4d.shape[0] // len(bbox_per_frame)
        bbox_per_frame_len = len(bbox_per_frame)

        for i in range(bbox_per_frame_len):
            bbox_ratios = bbox_per_frame[i]
            left, right, top, bottom, bbox_h, bbox_w = compute_bbox_LRTB_HW(dim_x, dim_y, bbox_ratios, use_int=clip_box_values, margin=0.01)

            if gauss_only:
            
                sigma_x = bbox_w / KERNEL_DIVISION
                sigma_y = bbox_h / KERNEL_DIVISION 

                mask_sigma = compute_sigma(bbox_h, bbox_w, fraction=sigma_strength) # 0.03

                if scale_local_foreground:
                    local_scale = (attention_probs_4d.max() * INJECTION_SCALE).detach()
                    
                bbox_map = create_diff_bbox_heatmap_gaussian_2(
                        height=map_h,
                        width=map_w,
                        top=top, 
                        bottom=bottom, 
                        left=left, 
                        right=right, 
                        sigma_x=sigma_x, 
                        sigma_y=sigma_y, 
                        mask_sigma=mask_sigma,
                        use_high_box_only=use_high_box_only,
                        scale_local_foreground = scale_local_foreground,
                        local_scale = local_scale,
                        act=torch.sigmoid,
                        normalize_gauss=normalize_gauss,
                        normalize_mask=normalize_mask,
                        index=i
                        ).unsqueeze(0).unsqueeze(-1).repeat(frame_size, 1, 1, n_tokens).to(attention_probs_4d.device).to(atten_dtype) #.half()
                
                
            else:
                raise NotImplementedError

            
            # scale option removed: influences values outside box and affects weaken map
            if box_with_gauss:
                left_int, right_int, top_int, bottom_int, bbox_h_int, bbox_w_int = compute_bbox_LRTB_HW(dim_x, dim_y, bbox_ratios, use_int=True, margin=0.01)

                x = torch.linspace(0, bbox_h_int.item(), bbox_h_int.item())
                y = torch.linspace(0, bbox_w_int.item(), bbox_w_int.item())
                x, y = torch.meshgrid(x, y, indexing="ij")
                noise_patch = (
                    gaussian_2d(
                        x,
                        y,
                        mx=int(bbox_h_int / 2),
                        my=int(bbox_w_int / 2),
                        sx=float(bbox_h_int / KERNEL_DIVISION),
                        sy=float(bbox_w_int / KERNEL_DIVISION),
                    )
                    .unsqueeze(0)
                    .unsqueeze(-1)
                    .repeat(frame_size, 1, 1, len(token_inds))
                    .to(attention_probs_4d.device)
                ).to(atten_dtype) # .half()
                # noise_patch = noise_patch * scale.detach()
                # ----------------------------------------------------------------------

        
            # zero-out non-interest tokens
            token_masks_zero = torch.zeros_like(bbox_map).to(attention_probs_4d.device).to(atten_dtype) # .half()
            token_masks_zero[:, :, :, token_inds] = 1

            core_bbox_map = bbox_map * token_masks_zero

            if minimize_bkgd:
                # create bkgd map (used for weaken map)
                max_values, _ = torch.max(core_bbox_map.view(frame_size, -1), dim=1)
                bkgd_map = max_values[..., None, None, None] - core_bbox_map
                
                core_bkgd_map = bkgd_map * token_masks_zero

            if box_with_gauss:
                box_inject = torch.zeros_like(bbox_map).to(attention_probs_4d.device).to(atten_dtype) #.half()

            if box_with_gauss:
                box_inject[:, top_int:bottom_int,left_int:right_int, :][..., token_inds] = noise_patch

            if box_with_gauss:
                core_bbox_map = core_bbox_map + box_inject

            weight_map.append(core_bbox_map)
            if minimize_bkgd:
                weaken_map.append(core_bkgd_map)
            
        new_weight_map = torch.cat(weight_map)
        if minimize_bkgd:
            new_weaken_map = torch.cat(weaken_map)
        
        return new_weight_map, new_weaken_map
    

    @staticmethod
    def localized_temporal_weight_map(attention_probs_5d, bbox_per_frame, dim_x, dim_y, sigma_strength, 
                                      clip_box_values = False, scale_local_foreground=False, local_scale=1,
                                      box_with_gauss=False,gauss_only=False, use_bkgd_zero=False, minimize_bkgd=False, 
                                      use_high_box_only=False, normalize_gauss=True, normalize_mask=True,
                                      allow_edge_margin=False, edge_margin=2):
        # Using Gaussian 2d distribution; weight map same size as attention argument.
        atten_dtype = attention_probs_5d.dtype
        # resolution viz can be increased using `transformer_in` temporal layer, instead of `mid_block_temp` temporal layer
        if minimize_bkgd: assert use_bkgd_zero, 'bbox mask is required to minimize background'
        weight_map = []
        if minimize_bkgd:
            weaken_map = []
        
        n_c, map_h, map_w, n_frames, n_frames = attention_probs_5d.shape
        if scale_local_foreground:
            local_scale = (attention_probs_5d.max() * INJECTION_SCALE).detach()

        # e.g (8, 16, 16, 24, 24])
        for j in range(len(bbox_per_frame)):
            temp_collect_bbox = []
            if minimize_bkgd:
                temp_collect_bbkgd = []

            for i in range(len(bbox_per_frame)):
                # Combine heatmaps using element-wise maximum (to avoid in-place injection or masking)
                map_i, dist_i, weight_i = get_patch(bbox_per_frame[i], i, j, bbox_per_frame, attention_probs_5d, dim_x, dim_y, 
                                  sigma_strength, gauss_only=gauss_only, allow_edge_margin=allow_edge_margin,
                                  use_high_box_only=use_high_box_only, normalize_gauss=normalize_gauss, 
                                  normalize_mask=normalize_mask, minimize_bkgd=minimize_bkgd, 
                                  clip_box_values = clip_box_values,
                                  scale_local_foreground = scale_local_foreground,
                                  local_scale = local_scale,
                                  )
                map_j, dist_j, weight_j = get_patch(bbox_per_frame[j], i, j, bbox_per_frame, attention_probs_5d, dim_x, dim_y, 
                                  sigma_strength, gauss_only=gauss_only, allow_edge_margin=allow_edge_margin,
                                  use_high_box_only=use_high_box_only, normalize_gauss=normalize_gauss,
                                  normalize_mask=normalize_mask, minimize_bkgd=minimize_bkgd, 
                                  clip_box_values = clip_box_values,
                                  scale_local_foreground = scale_local_foreground,
                                  local_scale = local_scale
                                  )


                combined_map = map_i + map_j

                # TODO: please drop `box_with_gauss`
                if box_with_gauss:
                    combined_inject_i = torch.zeros_like(combined_map).to(attention_probs_5d.device).to(atten_dtype) #.half()
                    combined_inject_j = torch.zeros_like(combined_map).to(attention_probs_5d.device).to(atten_dtype) #.half()
      
                if box_with_gauss:
                    patch_i, bbox_i = get_patch_old(bbox_per_frame[i], i, j, bbox_per_frame, attention_probs_5d, dim_x, dim_y, INJECTION_SCALE)
                    patch_j, bbox_j = get_patch_old(bbox_per_frame[j], i, j, bbox_per_frame, attention_probs_5d, dim_x, dim_y, INJECTION_SCALE)

                core_combined_bbox_map = combined_map
                if minimize_bkgd:
                    max_values, _ = torch.max(combined_map.view(n_c, -1), dim=1)
                    core_combined_bkgd_map = max_values[..., None, None] - combined_map
                 
                if box_with_gauss:
                    combined_inject_i[:, top_ia:bottom_ia, left_ia:right_ia] = patch_i
                    combined_inject_j[:, top_ja:bottom_ja, left_ja:right_ja] = patch_j

                    combined_map = (1/3) * (combined_map + combined_inject_i + combined_inject_j)

                temp_collect_bbox.append(core_combined_bbox_map)
                if minimize_bkgd:
                    temp_collect_bbkgd.append(core_combined_bkgd_map)

            weight_map.append(torch.stack(temp_collect_bbox, dim=-1))
            if minimize_bkgd:
                weaken_map.append(torch.stack(temp_collect_bbkgd, dim=-1))
        
        new_weight_map = torch.stack(weight_map, dim=-2)
        if minimize_bkgd:
            new_weaken_map = torch.stack(weaken_map, dim=-2)

        return new_weight_map, new_weaken_map

