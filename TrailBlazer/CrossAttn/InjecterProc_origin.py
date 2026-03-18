
import pdb
import math
import torch
import numpy as np
from typing import Dict, List, TypedDict

from ..Misc import Logger as log

from .BaseProc_origin import CrossAttnProcessorBase
from .BaseProc_origin import BundleType
from ..Misc.BBox import BoundingBox

from bin.utils.plot_helpers import plot2chk_image

class InjecterProcessor(CrossAttnProcessorBase):
    def __init__(
        self,
        bundle: BundleType,
        bbox_per_frame: List[BoundingBox],
        name: str,
        strengthen_scale: float = 0.0,
        weaken_scale: float = 1.0,
        is_text2vidzero: bool = False,
    ):
        super().__init__(bundle, is_text2vidzero=is_text2vidzero)
        self.strengthen_scale = strengthen_scale
        self.weaken_scale = weaken_scale
        self.bundle = bundle
        self.num_frames = len(bbox_per_frame)
        self.bbox_per_frame = bbox_per_frame
        self.use_weaken = True
        self.name = name

    def dd_core(self, attention_probs: torch.Tensor, dim_x, dim_y):
        """ """

        frame_size = attention_probs.shape[0] // self.num_frames
        num_affected_frames = self.num_frames
        attention_probs_copied = attention_probs.detach().clone()
        attention_probs_copied_weak = attention_probs.detach().clone()

        token_inds = self.bundle.get("token_inds")
        trailing_length = self.bundle["trailblazer"]["trailing_length"]
        trailing_inds = list(
            range(self.len_prompt + 1, self.len_prompt + trailing_length + 1)
        )
        # NOTE: Spatial cross attention editing
        if len(attention_probs.size()) == 4:
            all_tokens_inds = list(set(token_inds).union(set(trailing_inds)))
            strengthen_map = self.localized_weight_map(
                attention_probs_copied,
                token_inds=all_tokens_inds,
                bbox_per_frame=self.bbox_per_frame,
                dim_x = dim_x,
                dim_y = dim_y
            )

            weaken_map = torch.ones_like(strengthen_map)
            zero_indices = torch.where(strengthen_map == 0)
            weaken_map[zero_indices] = self.weaken_scale

            # for vis or loss compute
            attention_probs_copied_weak[..., all_tokens_inds] = attention_probs_copied_weak[..., all_tokens_inds] * weaken_map[..., all_tokens_inds]

            # weakening
            attention_probs_copied[..., all_tokens_inds] *= weaken_map[
                ..., all_tokens_inds
            ]

            # print(f'noise_patch max {strengthen_map[..., all_tokens_inds].max()}| strengthen scale {self.strengthen_scale}| strengthened noise_patch max {self.strengthen_scale * strengthen_map[..., all_tokens_inds].max()}\n')
            # strengthen
            attention_probs_copied[..., all_tokens_inds] += (
                self.strengthen_scale * strengthen_map[..., all_tokens_inds]
            )
            
            # plot2chk_image(self.strengthen_scale * strengthen_map[..., all_tokens_inds][0,:,:,0].detach().cpu(), filename=f"origin_strengthen_map_t_idx_{self.timestep:04d}")
            # plot2chk_image(weaken_map[..., all_tokens_inds][0,:,:,0].detach().cpu(), filename=f"origin_weaken_map_t_idx_{self.timestep:04d}")

        # NOTE: Temporal cross attention editing
        elif len(attention_probs.size()) == 5:
            strengthen_map = self.localized_temporal_weight_map(
                attention_probs_copied,
                bbox_per_frame=self.bbox_per_frame,
                dim_x = dim_x,
                dim_y = dim_y
            )
            weaken_map = torch.ones_like(strengthen_map)
            zero_indices = torch.where(strengthen_map == 0)
            weaken_map[zero_indices] = self.weaken_scale

            # for vis
            attention_probs_copied_weak = attention_probs_copied_weak * weaken_map

            # weakening
            attention_probs_copied *= weaken_map
            # strengthen
            attention_probs_copied += self.strengthen_scale * strengthen_map

        # return attention_probs_copied
        return attention_probs_copied_weak, attention_probs_copied, self.strengthen_scale * strengthen_map, strengthen_map, weaken_map

