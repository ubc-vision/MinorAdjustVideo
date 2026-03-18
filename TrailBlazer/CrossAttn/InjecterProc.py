from typing import Dict, List, TypedDict
import numpy as np
import timeit
import torch
import time
import pdb

from ..Misc import Logger as log

from .BaseProc import CrossAttnProcessorBase
from .BaseProc import BundleType
from ..Misc.BBox import BoundingBox

import imageio
from bin.utils.plot_helpers import convert_to_numpy, plot2chk_image
from TrailBlazer.CrossAttn.Utils import create_attention_video, time_taken


# --- NEW
class InjecterProcessor(CrossAttnProcessorBase):
    def __init__(
        self,
        bundle: BundleType,
        bbox_per_frame: List[BoundingBox],
        name: str,
        chosen_temp_block: str,
        temp_edit_at_low_res: bool = False,
        # edit_before_softmax: bool = False,
        use_trg_unscaled: bool = False,
        strengthen_scale: float = 0.0,
        weaken_scale: float = 1.0,
        sigma_strength: float = 0.03,
        clip_box_values: bool = False,
        scale_local_foreground: bool = False,
        box_with_gauss: bool = False,
        gauss_only: bool = False,
        use_high_box_only: bool = False,
        normalize_gauss: bool = False,
        normalize_mask: bool = False,
        use_bkgd_zero: bool = False,
        minimize_bkgd: bool = False,
        allow_edge_margin: bool = False,
        is_text2vidzero: bool = False,
        vis=False,
        aggregate_str = None, 
        debug=False,
        no_opt: bool = True,
        time_bf_motion: bool = False, 
    ):
        super().__init__(bundle, is_text2vidzero=is_text2vidzero)
        self.strengthen_scale = strengthen_scale
        self.weaken_scale = weaken_scale
        self.sigma_strength = sigma_strength
        self.clip_box_values = clip_box_values
        self.scale_local_foreground = scale_local_foreground
        self.box_with_gauss = box_with_gauss
        self.gauss_only = gauss_only
        self.use_high_box_only= use_high_box_only
        self.normalize_gauss = normalize_gauss
        self.normalize_mask = normalize_mask
        self.use_bkgd_zero = use_bkgd_zero
        self.minimize_bkgd = minimize_bkgd
        self.allow_edge_margin = allow_edge_margin
        self.bundle = bundle
        self.num_frames = len(bbox_per_frame)
        self.bbox_per_frame = bbox_per_frame
        self.use_weaken = True
        self.name = name
        self.use_trg_unscaled = use_trg_unscaled
        self.chosen_temp_block = chosen_temp_block
        self.temp_edit_at_low_res = temp_edit_at_low_res
        # self.edit_before_softmax = edit_before_softmax
        self.vis = vis
        self.no_opt = no_opt
        self.aggregate_str = aggregate_str
        self.time_bf_motion = time_bf_motion
        
    
    def dd_core(self, attention_probs: torch.Tensor, dim_x, dim_y, 
                use_spatial=False, use_temporal=False, timestep=None):
        # Makes a copy, modifies it and returns the modified copy.
        frame_size = attention_probs.shape[0] // self.num_frames
        num_affected_frames = self.num_frames
        
        # note: copy attention map A, to form D
        attention_probs_copied_weak = attention_probs.clone() # 120, 64, 64, 77
        attention_probs_copied_strength = attention_probs.clone()

        token_inds = self.bundle.get("token_inds") # note: focus token idx e.g 2
        trailing_length = self.bundle["trailblazer"]["trailing_length"]
        # note: len of base prompt e.g A cat is walking on the grass -> 7, trailing_length is 15
        # drops other words in base prompt [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
        trailing_inds = list(
            range(self.len_prompt + 1, self.len_prompt + trailing_length + 1)
        )

        if use_spatial:
            attn_type='spatial'
        elif use_temporal:
            attn_type='temporal'

        # NOTE: Spatial cross attention editing
        if len(attention_probs.size()) == 4:
            # note: union of trailing and token indices --> selected attention maps D = A^(i)
            # [2**, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
            all_tokens_inds = list(set(token_inds).union(set(trailing_inds)))
            
            # note: Sm = s(D)
            # start_time = time.time()
            strengthen_map, weaken_map = self.localized_weight_map(
                attention_probs_copied_strength,
                token_inds=all_tokens_inds,
                bbox_per_frame=self.bbox_per_frame,
                dim_x = dim_x,
                dim_y = dim_y,
                sigma_strength = self.sigma_strength,
                clip_box_values = self.clip_box_values,
                scale_local_foreground = self.scale_local_foreground,
                box_with_gauss = self.box_with_gauss,
                gauss_only = self.gauss_only,
                use_high_box_only= self.use_high_box_only,
                normalize_gauss= self.normalize_gauss,
                normalize_mask= self.normalize_mask,
                use_bkgd_zero=self.use_bkgd_zero,
                minimize_bkgd=self.minimize_bkgd,
                allow_edge_margin = self.allow_edge_margin
            )
 


            # Free up unused memory
            torch.cuda.empty_cache()
            attention_probs_copied_weak[..., all_tokens_inds] = attention_probs_copied_weak[..., all_tokens_inds] * (self.weaken_scale * weaken_map[..., all_tokens_inds])

            # note: Ds = D + Sm; strengthen
            if self.aggregate_str=='add':

                # NEW
                modified_weaken_map = (self.weaken_scale * weaken_map[..., all_tokens_inds]) + strengthen_map[..., all_tokens_inds]
                attention_probs_copied_strength[..., all_tokens_inds] = attention_probs_copied_strength[..., all_tokens_inds] * modified_weaken_map + (self.strengthen_scale * strengthen_map[..., all_tokens_inds]) # 120, 64, 64, 77

            elif self.aggregate_str=='mul':
                attention_probs_copied_strength[..., all_tokens_inds] = attention_probs_copied_strength[..., all_tokens_inds] * (self.strengthen_scale * strengthen_map[..., all_tokens_inds]) # 120, 64, 64, 77
            
            elif self.aggregate_str=='mul+add':
                attention_probs_copied_strength[..., all_tokens_inds] = attention_probs_copied_strength[..., all_tokens_inds] * (self.strengthen_scale * strengthen_map[..., all_tokens_inds]) # 120, 64, 64, 77
                attention_probs_copied_strength[..., all_tokens_inds] = attention_probs_copied_strength[..., all_tokens_inds] + (self.strengthen_scale * strengthen_map[..., all_tokens_inds]) # 120, 64, 64, 77
                attention_probs_copied_strength[..., all_tokens_inds] = attention_probs_copied_strength[..., all_tokens_inds] + (self.weaken_scale * weaken_map[..., all_tokens_inds]) # 120, 64, 64, 77
                
            else:
                raise NotImplementedError
            

        # NOTE: Temporal cross attention editing
        elif len(attention_probs.size()) == 5:
            # pass

            strengthen_map, weaken_map = self.localized_temporal_weight_map(
                attention_probs_copied_strength,
                bbox_per_frame=self.bbox_per_frame,
                dim_x = dim_x,
                dim_y = dim_y,
            sigma_strength = self.sigma_strength,
            clip_box_values = self.clip_box_values,
            scale_local_foreground = self.scale_local_foreground,
            box_with_gauss = self.box_with_gauss,
            gauss_only = self.gauss_only,
            use_high_box_only= self.use_high_box_only,
            normalize_gauss= self.normalize_gauss,
            normalize_mask= self.normalize_mask,
            use_bkgd_zero=self.use_bkgd_zero,
            minimize_bkgd=self.minimize_bkgd,
            allow_edge_margin = self.allow_edge_margin
            )
            

            # weakening
            attention_probs_copied_weak = attention_probs_copied_weak * weaken_map
            
            # strengthen
            if self.aggregate_str=='add':
                # weak and strength scaling
                # NEW
                modified_weaken_map = (self.weaken_scale * weaken_map) + strengthen_map
                attention_probs_copied_strength = attention_probs_copied_strength * modified_weaken_map + (self.strengthen_scale * strengthen_map) 

            elif self.aggregate_str=='mul':
                attention_probs_copied_strength = attention_probs_copied_strength * (self.strengthen_scale * strengthen_map)
            
            elif self.aggregate_str=='mul+add':
                attention_probs_copied_strength = attention_probs_copied_strength * (self.strengthen_scale * strengthen_map)
                attention_probs_copied_strength = attention_probs_copied_strength + (self.strengthen_scale * strengthen_map)
                attention_probs_copied_strength = attention_probs_copied_strength + (self.weaken_scale * weaken_map)
                
            else:
                raise NotImplementedError
            
            # -----------
            if self.vis:
                # TODO: 
                pass

            # --------------
            # '''

        # NOTE `attention_probs_copied_strength` contains both weakening and strengthening
        return attention_probs_copied_weak, attention_probs_copied_strength, self.strengthen_scale * strengthen_map, strengthen_map, weaken_map

