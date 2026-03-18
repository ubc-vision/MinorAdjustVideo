from typing import Dict, List, TypedDict
import numpy as np
import pdb
import math
import torch
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from diffusers.models.attention_processor import Attention as CrossAttention
from einops import rearrange
from ..Misc import Logger as log
from ..Misc.BBox import BoundingBox
from ..Misc import Const

from bin.utils.plot_helpers import (plot2chk_image, plot_opt_spatial_maps, 
                                    plot_single_spatial_out_map)

KERNEL_DIVISION = 3.
INJECTION_SCALE = 1.0


def reshape_fortran(x, shape):
    """ Reshape a tensor in the fortran index. See
    https://stackoverflow.com/a/63964246
    """
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))


def gaussian_2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):
    """ 2d Gaussian weight function
    """
    gaussian_map = (
        1
        / (2 * math.pi * sx * sy)
        * torch.exp(-((x - mx) ** 2 / (2 * sx**2) + (y - my) ** 2 / (2 * sy**2)))
    )

    gaussian_map.div_(gaussian_map.max())

    # plot2chk_image(gaussian_map, filename="gaussian_map_origin")
    return gaussian_map


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
        base_prompt = self.prompt.split(";")[0]
        self.len_prompt = len(base_prompt.split(" "))
        self.prompt_len = len(self.prompt.split(" "))
        self.timestep = None 
        self.use_dd = False
        self.use_dd_temporal = False
        self.unet_chunk_size = 2
        self._cross_attention_map = None
        self._loss = None
        self._parameters = None
        self.is_text2vidzero = is_text2vidzero
        bbox = None
        self.wandb_log = False
        self.output_opt_viz_path = ""

    @property
    def cross_attention_map(self):
        return self._cross_attention_map

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
    ):

        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )
        #print("====================")
        query = attn.to_q(hidden_states)

        is_cross_attention = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        # elif attn.cross_attention_norm:
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        def rearrange_3(tensor, f):
            F, D, C = tensor.size()
            return torch.reshape(tensor, (F // f, f, D, C))

        def rearrange_4(tensor):
            B, F, D, C = tensor.size()
            return torch.reshape(tensor, (B * F, D, C))

        # Cross Frame Attention
        if not is_cross_attention and self.is_text2vidzero:
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

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        # Cross attention map
        #print(query.shape, key.shape, value.shape)
        attention_probs = attn.get_attention_scores(query, key)
        # print(attention_probs.shape)
        # torch.Size([960, 77, 64]) torch.Size([960, 256, 64]) torch.Size([960, 77, 64]) torch.Size([960, 256, 77])
        # torch.Size([10240, 24, 64]) torch.Size([10240, 24, 64]) torch.Size([10240, 24, 64]) torch.Size([10240, 24, 24])

        # set overlay map to default None
        self._overlay_attn_maps = None

        n = attention_probs.shape[0] // 2
        if attention_probs.shape[-1] == CrossAttnProcessorBase.MAX_LEN_CLIP_TOKENS:
            # NEW      
            # -------------------------------------------------------------
            # for spatial editing and/or non-editing steps (if required)
            token_inds = self.bundle.get("token_inds") # note: focus token idx e.g 2
            trailing_length = self.bundle["trailblazer"]["trailing_length"]
            trailing_inds = list(range(self.len_prompt + 1, self.len_prompt + trailing_length + 1))
            all_tokens_inds = list(set(token_inds).union(set(trailing_inds)))
            # -------------------------------------------------------------

            gcd = np.gcd(Const.DEFAULT_HEIGHT, Const.DEFAULT_WIDTH)
            height_multiplier = Const.DEFAULT_HEIGHT / gcd
            width_multiplier = Const.DEFAULT_WIDTH / gcd
            factor = attention_probs.shape[1] // (height_multiplier * width_multiplier)
            factor = int(np.sqrt(factor))
            dim_y = int(factor * height_multiplier)
            dim_x = int(factor * width_multiplier)

            vis_dict = {
                        # "down_blocks.0.attentions.0.transformer_blocks.0.attn2": "down.0.0.0"
                        "up_blocks.2.attentions.2.transformer_blocks.0.attn2": "up.2.2.0"
                        }

            # print(f"self.timestep {self.timestep}")
            if self.use_dd:
                # print('----------------- edit: YES -----------------')
                
                # self.use_dd = False
                # please NOTE the dim_y, dim_x order
                attention_probs_4d = attention_probs.view(
                    attention_probs.shape[0], dim_y, dim_x, attention_probs.shape[-1]
                )[n:]

                bottom_attention_copy = attention_probs_4d.clone()

                modified_bottom_attention_copy_wk, modified_bottom_attention_copy_str, diff_bboxmap_str, unscaled_diff_bboxmap_str, unscaled_weaken_map = self.dd_core(attention_probs_4d, dim_x, dim_y)
                
                # print(f'------ self.timestep {self.timestep}')
                # if self.timestep==0:
                
                attention_probs_4d = modified_bottom_attention_copy_str
                # attention_probs_4d = self.dd_core(attention_probs_4d, dim_x, dim_y)
                
                # update attention map with forward guidance
                attention_probs[n:] = attention_probs_4d.reshape(
                    attention_probs_4d.shape[0], dim_y * dim_x, attention_probs_4d.shape[-1]
                )

                # ------------
                
                # # set overlay map to default None
                # self._overlay_attn_maps = None

                if self.name in vis_dict: 
                    # print(f"self.timestep {self.timestep}")
                    plot_single_spatial_out_map(bottom_attention_copy.detach().cpu(), 
                                                    n_frames=len(self.bbox_per_frame),
                                                    opt_id=0, 
                                                    timestep=self.timestep, 
                                                    all_tokens_inds=all_tokens_inds, 
                                                    n_opt_iterations=0, 
                                                    output_opt_viz_path=self.output_opt_viz_path,
                                                    # save plot image media either online/local 
                                                    wandb_log=self.wandb_log, 
                                                    plot_local=not self.wandb_log, 
                                                    mini_label=vis_dict[f'{self.name}'])    
                    """
                    plot_opt_spatial_maps(bottom_attention_copy.detach().cpu(), 
                                            unscaled_diff_bboxmap_str.detach().cpu(), 
                                            modified_bottom_attention_copy_str.detach().cpu(), 
                                            modified_bottom_attention_copy_wk.detach().cpu(),
                                            scale_box_map = diff_bboxmap_str,
                                    opt_id=0, timestep=self.timestep, all_tokens_inds=all_tokens_inds, 
                                    n_opt_iterations=0, output_opt_viz_path=self.output_opt_viz_path,
                                    # save plot image media either online/local 
                                    wandb_log=self.wandb_log, plot_local=not self.wandb_log, mini_label=vis_dict[f'{self.name}'])
                    """
               
                if self.bundle['overlay_maps']:

                    if self.name in vis_dict:
                        # save overlay at last editing step
                        # print(f"self.timestep OVERLAY {self.timestep}")
                        if (self.timestep == self.bundle['trailblazer']['num_dd_spatial_steps']-1):
                            # print(f"self.timestep OVERLAY {self.timestep}")
                        # if (self.timestep in np.arange(self.bundle['trailblazer']['num_dd_spatial_steps'])):
                            
                            n_channels = bottom_attention_copy.detach().cpu().shape[0]
                            window_size = n_channels // self.num_frames
                            assert n_channels % self.num_frames == 0, 'number of channels must be perfectly divisible by number of frames'

                            overlay_maps = []
                            for ir, range_id in enumerate(range(0, n_channels, window_size)):

                                first_token = 0
                                # print(f'range {range_id, range_id + window_size}')
                                frame_avg_attn_map = bottom_attention_copy[range_id : range_id + window_size, :, :, all_tokens_inds][..., first_token].detach().cpu().mean(dim=0)
                                overlay_maps.append(frame_avg_attn_map)

                                '''
                                fontsize = 10
                                fig, axs = plt.subplots(1, 1, figsize=(10, 10)) 
                                # for first_id in range(range_):
                                #     img0 = axs[first_id].imshow(bottom_attention_copy[first_id,:,:,all_tokens_inds][..., first_token].detach().cpu()) 
                                #     axs[first_id].set_title(f"{first_id}", fontsize=fontsize)
                                #     axs[first_id].axis("off")
                                #     colorbar = plt.colorbar(img0, ax=axs[first_id], fraction=0.046, pad=0.04)

                                # if ir==0:
                                img1 = axs.imshow(frame_avg_attn_map) 
                                axs.set_title(f"{ir}", fontsize=fontsize)
                                axs.axis("off")
                                colorbar = plt.colorbar(img1, ax=axs, fraction=0.046, pad=0.04)
                            
                                fig.tight_layout()
                                save_dir = 'checkers/imgs'
                                fig.savefig(f"{str(save_dir)}/maps_origin_map-attn_map_spatial_{ir:04d}_t_idx_{self.timestep}.png", bbox_inches='tight')
                                '''
                                
                            overlay_attn_maps = torch.stack(overlay_maps)
                            self._overlay_attn_maps = overlay_attn_maps
                
                
                # ------------

            else:
                # set overlay map to default None
                # self._overlay_attn_maps = None
                
                # --------------------------------
                # TEMPORARY CODE
                # """
                if self.bundle['vis_opt_and_denoise_maps']:
                    if self.name in vis_dict:

                        # please NOTE the dim_y, dim_x order
                        attention_probs_4d_non_edit = attention_probs.view(
                            attention_probs.shape[0], dim_y, dim_x, attention_probs.shape[-1]
                        )[n:]

                        bottom_attention_copy_non_edit = attention_probs_4d_non_edit.clone()

                        # print(f"self.timestep {self.timestep}")
                        plot_single_spatial_out_map(bottom_attention_copy_non_edit.detach().cpu(), 
                                                    n_frames=len(self.bbox_per_frame),
                                                    opt_id=0, 
                                                    timestep=self.timestep, 
                                                    all_tokens_inds=all_tokens_inds, 
                                                    n_opt_iterations=0, 
                                                    output_opt_viz_path=self.output_opt_viz_path,
                                                    # save plot image media either online/local 
                                                    wandb_log=self.wandb_log, 
                                                    plot_local=not self.wandb_log, 
                                                    mini_label=vis_dict[f'{self.name}'])
                # """
                # --------------------------------

            self._cross_attention_map = attention_probs.view(
                attention_probs.shape[0], dim_y, dim_x, attention_probs.shape[-1]
            )[n:]

        elif (
            attention_probs.shape[-1] == self.num_frames
            and (attention_probs.shape[0] == 65536)
        ):  
            #dim = int(np.sqrt(attention_probs.shape[0] // (2 * attn.heads)))
            gcd = np.gcd(Const.DEFAULT_HEIGHT, Const.DEFAULT_WIDTH)
            height_multiplier = Const.DEFAULT_HEIGHT / gcd
            width_multiplier = Const.DEFAULT_WIDTH / gcd
            factor = (attention_probs.shape[0] // (2 * attn.heads)) // (height_multiplier * width_multiplier)
            factor = int(np.sqrt(factor))
            dim_y = int(factor * height_multiplier)
            dim_x = int(factor * width_multiplier)
            if self.use_dd_temporal:
                # self.use_dd_temporal = False
                def temporal_doit(origin_attn):
                    temporal_attn = reshape_fortran(
                        origin_attn,
                        (attn.heads, dim_y, dim_x, self.num_frames, self.num_frames),
                    )
                   

                    temporal_attn = torch.transpose(temporal_attn, 1, 2)
                    modified_bottom_attention_copy_wk, modified_bottom_attention_copy_str, diff_bboxmap_str, unscaled_diff_bboxmap_str, unscaled_weaken_map = self.dd_core(temporal_attn, dim_x, dim_y)
                    temporal_attn = modified_bottom_attention_copy_str
                    # temporal_attn = self.dd_core(temporal_attn, dim_x, dim_y)
                    # torch.Size([8, 64, 64, 24, 24])

                    temporal_attn = torch.transpose(temporal_attn, 1, 2)
                    temporal_attn = reshape_fortran(
                        temporal_attn,
                        (attn.heads * dim_y * dim_x, self.num_frames, self.num_frames),
                    )
                    return temporal_attn


                # NOTE: So null text embedding for classification free guidance
                # doesn't really help?
                #attention_probs[n:] = temporal_doit(attention_probs[n:])
                
                # update attention map with forward guidance
                attention_probs[:n] = temporal_doit(attention_probs[:n])
                
            #     # set overlay map to default None
            #     self._overlay_attn_maps = None
            # else:
            #     # set overlay map to default None
            #     self._overlay_attn_maps = None
        
            self._cross_attention_map = reshape_fortran(
                attention_probs[:n],
                (attn.heads, dim_y, dim_x, self.num_frames, self.num_frames),
            )
            self._cross_attention_map = self._cross_attention_map.mean(dim=0)
            self._cross_attention_map = torch.transpose(self._cross_attention_map, 0, 1)

        # else:
        #     # set overlay map to default None
        #     self._overlay_attn_maps = None

        attention_probs = torch.abs(attention_probs)

        # -------------------------------

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

    @abstractmethod
    def dd_core(self):
        """All DD variants implement this function"""
        pass

    @staticmethod
    def localized_weight_map(attention_probs_4d, token_inds, bbox_per_frame, dim_x, dim_y, scale=1):
        """Using guassian 2d distribution to generate weight map and return the
        array with the same size of the attention argument.
        """
        dim = int(attention_probs_4d.size()[1])
        max_val = attention_probs_4d.max()
        weight_map = torch.zeros_like(attention_probs_4d).half()
        frame_size = attention_probs_4d.shape[0] // len(bbox_per_frame)
        
        for i in range(len(bbox_per_frame)):
            bbox_ratios = bbox_per_frame[i]
            bbox = BoundingBox(dim_x, dim_y, bbox_ratios)
            # Generating the gaussian distribution map patch
            x = torch.linspace(0, bbox.height, bbox.height)
            y = torch.linspace(0, bbox.width, bbox.width)
            x, y = torch.meshgrid(x, y, indexing="ij")
            noise_patch = (
                gaussian_2d(
                    x,
                    y,
                    mx=int(bbox.height / 2),
                    my=int(bbox.width / 2),
                    sx=float(bbox.height / KERNEL_DIVISION),
                    sy=float(bbox.width / KERNEL_DIVISION),
                )
                .unsqueeze(0)
                .unsqueeze(-1)
                .repeat(frame_size, 1, 1, len(token_inds))
                .to(attention_probs_4d.device)
            ).half()

            scale = attention_probs_4d.max() * INJECTION_SCALE
            # original_noise_patch  = noise_patch.max()
            noise_patch.mul_(scale)
            # if i==23:
            #     print(f'original noise_patch max {original_noise_patch.max()}|attn max {attention_probs_4d.max()}|curr noise_patch max {noise_patch.max()}')

            b_idx = frame_size * i
            e_idx = frame_size * (i + 1)
            try:
                # NOTE: 0-collapse handled in BBox.py

                bbox.sliced_tensor_in_bbox(weight_map)[
                    b_idx:e_idx, ..., token_inds
                ] = noise_patch
                """
                if attention_probs_4d.shape == (480, 5, 5, 77):
                    print(f"left {bbox.left}, right {bbox.right}, top {bbox.top}, bottom {bbox.bottom}")
                if bbox.left == 1 and bbox.right == 1 and bbox.top == 2 and bbox.bottom == 3 and i==6 and attention_probs_4d.shape == (480, 5, 5, 77):
                    pdb.set_trace()
                """
            except:
                pdb.set_trace()
            
            # boxmap_origin = weight_map[b_idx:e_idx]
            # boxmap_origin.mean(), boxmap_origin.std()
            # plot2chk_image(boxmap_origin[0,:,:,token_inds[0]], filename="boxmap_origin")
            # weight_map[:, bbox.top : bbox.bottom, bbox.left : bbox.right, :][b_idx:e_idx, ..., token_inds].shape

            # bbox.left, bbox.right, bbox.top, bbox.bottom, bbox.height, bbox.width 
        
       
        return weight_map

    @staticmethod
    def localized_temporal_weight_map(attention_probs_5d, bbox_per_frame, dim_x, dim_y, scale=1):
        """Using guassian 2d distribution to generate weight map and return the
        array with the same size of the attention argument.
        """
        dim = int(attention_probs_5d.size()[1])
        f = attention_probs_5d.shape[-1]
        max_val = attention_probs_5d.max()
        weight_map = torch.zeros_like(attention_probs_5d).half()

        def get_patch(bbox_at_frame, i, j, bbox_per_frame):
            bbox = BoundingBox(dim_x, dim_y, bbox_at_frame)
            # Generating the gaussian distribution map patch
            x = torch.linspace(0, bbox.height, bbox.height)
            y = torch.linspace(0, bbox.width, bbox.width)
            x, y = torch.meshgrid(x, y, indexing="ij")
            noise_patch = (
                gaussian_2d(
                    x,
                    y,
                    mx=int(bbox.height / 2),
                    my=int(bbox.width / 2),
                    sx=float(bbox.height / KERNEL_DIVISION),
                    sy=float(bbox.width / KERNEL_DIVISION),
                )
                .unsqueeze(0)
                .repeat(attention_probs_5d.shape[0], 1, 1)
                .to(attention_probs_5d.device)
            ).half()
            scale = attention_probs_5d.max() * INJECTION_SCALE
            noise_patch.mul_(scale)
            inv_noise_patch = noise_patch - noise_patch.max()
            dist = (float(abs(j - i))) / len(bbox_per_frame)
            final_patch = inv_noise_patch * dist + noise_patch * (1. - dist)
            #final_patch = noise_patch * (1. - dist)
            #final_patch = inv_noise_patch * dist
            return final_patch, bbox


        for j in range(len(bbox_per_frame)):
            for i in range(len(bbox_per_frame)):
                patch_i, bbox_i = get_patch(bbox_per_frame[i], i, j, bbox_per_frame)
                patch_j, bbox_j = get_patch(bbox_per_frame[j], i, j, bbox_per_frame)
                bbox_i.sliced_tensor_in_bbox(weight_map)[..., i, j] = patch_i
                bbox_j.sliced_tensor_in_bbox(weight_map)[..., i, j] = patch_j

        return weight_map
