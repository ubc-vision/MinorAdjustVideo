import inspect
from typing import Any, Callable, Dict, List, Optional, Union


import os
import re
import pdb
import torch
import numpy as np
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer
from dataclasses import dataclass

from diffusers.loaders import LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, UNet3DConditionModel
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    deprecate,
    logging,
    replace_example_docstring,
    BaseOutput,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_synth import (
    tensor2vid,
)
from ..CrossAttn.InjecterProc import InjecterProcessor
from ..Misc import Logger as log
from ..Misc import Const

from pathlib import Path
from .Loss_factory import compute_loss
from TrailBlazer.CrossAttn.Utils import create_motion_timestep_video, add_timestep_to_video_array, create_timestep_motion_video


# NOTE: attn2 is for cross-attention while attn1 is self-attention



def is_close(norm_latents_grad, norm_latents_grad_no_checkpoint, dtype):
    """Check for agreement up to a certain degree
    Ref: CGPT"""
    # fp16 - precise only up to 3-4 decimal points | fp32 6–7 decimal points | fp64 15–16 decimal points.

    # Check agreement with tolerance 
    if dtype==torch.float16:
        response = np.isclose(norm_latents_grad, norm_latents_grad_no_checkpoint, rtol=1e-3, atol=1e-4)
    
    elif dtype==torch.float32:
        response = np.isclose(norm_latents_grad, norm_latents_grad_no_checkpoint, rtol=1e-6, atol=1e-7)
    else:
        raise NotImplementedError
    
    return response

def extract_overlap_maps(self, overlay_attn_maps):
    overlay_attn_maps = None

    for name, module in self.named_modules():
        module_name = type(module).__name__

        if module_name == "Attention" and "attn2" in name:
            if module.processor._overlay_attn_maps!=None:
                # print('FOUND OVERLAY MAP')
                # update
                overlay_attn_maps = module.processor._overlay_attn_maps
    
    return overlay_attn_maps

def verify_gradient_chkpt(self, bundle, init_latent_model_input, t, t_idx, opt_idx, prompt_embeds, 
                      cross_attention_kwargs, use_mean, bboxes_ratios, init_bboxes_ratios,
                      loss_dict, strengthen_scale, weaken_scale, wandb_log, latents_clone,
                      init_latents_grad, optimizer, do_classifier_free_guidance
                      ):

    model_dtype = self.unet.dtype
    # set gradients to None, to avoid gradient accumulation
    optimizer.zero_grad()
    # reset 
    latents_clone = latents_clone.requires_grad_(True)
    # note: cast interactions back to fp16 if needed
    latent_model_input = (torch.cat([latents_clone] * 2) if do_classifier_free_guidance else latents_clone).to(model_dtype)
    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)  # output dim is same

    _, cross_attn_loss = self.unet(
    sample=latent_model_input, # 2, 4, 24, 16, 16
    timestep=t,
    timestep_iter=t_idx,
    opt_idx=opt_idx,
    encoder_hidden_states=prompt_embeds, # e.g 50, 77, 1024 (n_framex2, 77-dim since its clip embedding)
    cross_attention_kwargs=cross_attention_kwargs,
    return_dict=False,

    skip_bkgd_layers = bundle.get('skip_bkgd_layers'),
    specific_layers = bundle.get('specific_layers'),
    use_mean=use_mean,
    single_layer = bundle.get('single_layer'),
    get_loss = True,
    use_grad_chkpt = False,
    # use_reentrant = bundle.get('use_reentrant'),
    out_in_loss = bundle.get('out_in_loss'),
    max_cross_loss = bundle.get('max_cross_loss'),
    use_diff_loss = bundle.get('use_diff_loss')
    )

    loss = compute_loss(cross_attn_loss, bundle, bboxes_ratios, 
            init_bboxes_ratios=init_bboxes_ratios, loss_dict=loss_dict,
            t_idx=t_idx, t=t, opt_idx=opt_idx, n_opt_iterations=bundle.get('n_opt_iterations'),
            strengthen_scale=strengthen_scale, weaken_scale=weaken_scale, 
            outside_bbox_loss_scale = bundle['outside_bbox_loss_scale'],
            inside_bbox_attn_loss_scale = bundle['inside_bbox_attn_loss_scale'],
            box_temp_smooth_scale = bundle['box_temp_smooth_scale'],
            box_flip_thresh = bundle['box_flip_thresh'],
            box_flip_thresh_scale = bundle['box_flip_thresh_scale'],
            use_mean=use_mean, wandb_log=wandb_log)

    loss.backward()
    latents_grad_no_checkpting = latents_clone.grad

    norm_init_latents_grad = torch.norm(init_latents_grad)
    norm_latents_grad_no_checkpting = torch.norm(latents_grad_no_checkpting)
    print(f'norm_init_latents_grad {norm_init_latents_grad}, norm_latents_grad_no_checkpting {norm_latents_grad_no_checkpting}')
    
    close_check = is_close(norm_init_latents_grad.detach().cpu().numpy(), 
                           norm_latents_grad_no_checkpting.detach().cpu().numpy(), 
                           model_dtype)

    if close_check:
        print('\ncheckpointing verification: PASSED\n')
    else:
        diff = (norm_init_latents_grad - norm_latents_grad_no_checkpting).item()
        print(f'INCORRECT checkpointing gradients, diff {diff}: latent gradients are NOT the same as without checkpointing\n')

    exit()
    # fp16: latents_grad 0.23194[2817568779], latents_grad_no_checkpoint 0.23194[37712430954]
    # fp32: latents_grad 0.15975892543792725, latents_grad_no_checkpoint 0.15975892543792725
    # ------------

def get_prev_module(self, curr_name_, check_key=False):

    out_in_layers_near = {
    'down_blocks[0].attentions[1].transformer_blocks[0].attn2' : self.down_blocks[0].attentions[0].transformer_blocks[0].attn2,
    'down_blocks[1].attentions[0].transformer_blocks[0].attn2' : self.down_blocks[0].attentions[1].transformer_blocks[0].attn2,
    'down_blocks[1].attentions[1].transformer_blocks[0].attn2' : self.down_blocks[1].attentions[0].transformer_blocks[0].attn2,
    'down_blocks[2].attentions[0].transformer_blocks[0].attn2' : self.down_blocks[1].attentions[1].transformer_blocks[0].attn2,
    'down_blocks[2].attentions[1].transformer_blocks[0].attn2' : self.down_blocks[2].attentions[0].transformer_blocks[0].attn2,
    'mid_block.attentions[0].transformer_blocks[0].attn2' : self.down_blocks[2].attentions[1].transformer_blocks[0].attn2,
    'up_blocks[1].attentions[0].transformer_blocks[0]attn2' : self.mid_block.attentions[0].transformer_blocks[0].attn2,
    'up_blocks[1].attentions[1].transformer_blocks[0].attn2' : self.up_blocks[1].attentions[0].transformer_blocks[0].attn2,
    'up_blocks[1].attentions[2].transformer_blocks[0].attn2' : self.up_blocks[1].attentions[1].transformer_blocks[0].attn2, 
    'up_blocks[2].attentions[0].transformer_blocks[0].attn2' : self.up_blocks[1].attentions[2].transformer_blocks[0].attn2,
    'up_blocks[2].attentions[1].transformer_blocks[0].attn2' : self.up_blocks[2].attentions[0].transformer_blocks[0].attn2, 
    'up_blocks[2].attentions[2].transformer_blocks[0].attn2' : self.up_blocks[2].attentions[1].transformer_blocks[0].attn2, 
    'up_blocks[3].attentions[0].transformer_blocks[0].attn2' : self.up_blocks[2].attentions[2].transformer_blocks[0].attn2, 
    'up_blocks[3].attentions[1].transformer_blocks[0].attn2' : self.up_blocks[3].attentions[0].transformer_blocks[0].attn2,
    'up_blocks[3].attentions[2].transformer_blocks[0].attn2' : self.up_blocks[3].attentions[1].transformer_blocks[0].attn2,
        }
    
    if check_key:
        return curr_name_ in out_in_layers_near
    return out_in_layers_near[curr_name_]


# ref https://github.com/silent-chen/layout-guidance/blob/50fcf7525fa59681510d90759136e7e470903136/utils.py#L8
def get_layer_cross_loss(extracted_attention_map, token_ids, num_frames, use_mean=True):

    n_sub = extracted_attention_map.shape[0] // 2
    if extracted_attention_map.shape[-1]==77:
        assert token_ids!=None, 'token_ids is None'
        sub_cross_attn = extracted_attention_map[n_sub:, :, :, token_ids] # .clone()
        
    elif extracted_attention_map.shape[-1]==num_frames:
        sub_cross_attn = extracted_attention_map[:n_sub] # .clone()
    else:
        raise NotImplementedError

    # using the mean makes the loss invariant to bbox sizes and normalizes it for stability
    if use_mean:
        layer_loss = sub_cross_attn.mean()
    else:
        layer_loss = sub_cross_attn.sum()

    return layer_loss

def get_layer_direct_sub_cross_loss(edit_sub_attn_map_, sub_attn_map_, token_ids, num_frames, eps=1e-32):
    edit_sub_attn_map = edit_sub_attn_map_.to(torch.float32)
    sub_attn_map = sub_attn_map_.to(torch.float32)
    
    if edit_sub_attn_map.shape[-1]==num_frames:
        b, h, w, nf, nf = edit_sub_attn_map.shape
        sub_cross_attn_value = edit_sub_attn_map.reshape(b,-1).sum(dim=-1)/(sub_attn_map.reshape(b, -1).sum(dim=-1) + eps)
        layer_loss = torch.mean((1 - sub_cross_attn_value) ** 2)

    elif edit_sub_attn_map.shape[-1]==len(token_ids):
        # pdb.set_trace()
        assert token_ids!=None, 'token_ids is None'
        b, h, w, n_select_tokens = edit_sub_attn_map.shape # e.g 120, 40, 40, 16
        sub_cross_attn_value = edit_sub_attn_map.reshape(b,-1).sum(dim=-1)/(sub_attn_map.reshape(b, -1).sum(dim=-1) + eps) # e.g 120
        layer_loss = torch.mean((1 - sub_cross_attn_value) ** 2)
    else:
        raise NotImplementedError

    assert not (layer_loss.isinf().any() or layer_loss.isnan().any() or layer_loss > 1000), f'layer loss value {layer_loss} is running into numerical issues'
    layer_loss = layer_loss.to(torch.float16)
    return layer_loss

def get_layer_direct_sub_cross_attn_mean_values(edit_sub_attn_map_, sub_attn_map_, token_ids, num_frames, eps=1e-32):
    edit_sub_attn_map = edit_sub_attn_map_.to(torch.float32)
    sub_attn_map = sub_attn_map_.to(torch.float32)
    
    if edit_sub_attn_map.shape[-1]==num_frames:
        b, h, w, nf, nf = edit_sub_attn_map.shape
        sub_cross_attn_value = edit_sub_attn_map.reshape(b,-1).sum(dim=-1)/(sub_attn_map.reshape(b, -1).sum(dim=-1) + eps)
        layer_mean_value = torch.mean(sub_cross_attn_value)

    elif edit_sub_attn_map.shape[-1]==len(token_ids):
        assert token_ids!=None, 'token_ids is None'
        b, h, w, n_select_tokens = edit_sub_attn_map.shape
        sub_cross_attn_value = edit_sub_attn_map.reshape(b,-1).sum(dim=-1)/(sub_attn_map.reshape(b, -1).sum(dim=-1) + eps)
        layer_mean_value = torch.mean(sub_cross_attn_value)
    else:
        raise NotImplementedError

    assert not (layer_mean_value.isinf().any() or layer_mean_value.isnan().any() or layer_mean_value > 1000), f'layer loss value {layer_mean_value} is running into numerical issues'
    layer_mean_value = layer_mean_value.to(torch.float16)
    return layer_mean_value

def get_layer_out_in_loss(input, output, use_mean=True):
    if use_mean:
        layer_loss = ((input-output)**2).mean()
    else:
        layer_loss = ((input-output)**2).sum()

    return layer_loss

def get_cross_attn_loss(self, num_frames, single_layer=False, skip_bkgd_layers=True, 
                        specific_layers = [], timestep_iter=None, opt_idx=None,
                        out_in_loss=False, max_cross_loss=False, use_diff_loss=False, 
                        use_mean=True):

    if not single_layer:
        if timestep_iter==0 and opt_idx==0:
            print('extracting multi-layer cross attention...')

        n_layers = 0
        total_loss_wk, total_loss_str = 0, 0
        total_mean_attn_value_wk, total_mean_attn_value_str = 0, 0
        total_loss_out_in = 0
        total_diff_loss = 0
        
        for name, module in self.named_modules():
            module_name = type(module).__name__

            if module_name == "Attention" and "attn2" in name:
                extracted_attention_map_wk = module.processor._cross_attention_map_wk
                extracted_attention_map_str = module.processor._cross_attention_map_str

                _sub_attn = module.processor._sub_attn
                _modified_sub_attn_wk = module.processor._modified_sub_attn_wk 
                _modified_sub_attn_str = module.processor._modified_sub_attn_str 

                _masked_map_wk_values = module.processor._masked_map_wk_values
                _masked_map_str_values = module.processor._masked_map_str_values
                
                if "temp_attentions" in name:
                    if extracted_attention_map_str!=None:
                        pass
                else:
                    if extracted_attention_map_str!=None:
                        pass

                if out_in_loss:
                    if _sub_attn!=None:
                        name_ = replace_method(name)

                        if get_prev_module(self, curr_name_=name_, check_key=True):
                            prev_module = get_prev_module(self, curr_name_=name_)
                            # print(name_, _sub_attn.shape)
                            prev_modified_sub_attn = prev_module.processor._modified_sub_attn_str

                            if prev_modified_sub_attn.shape != _sub_attn.shape:
                                nc, h, w, tokens = prev_modified_sub_attn.shape
                   
                                # bicubic interpolation to interpolate the resolution dimensions
                                up_sub_attn_bicubic = F.interpolate(_sub_attn.permute(0,3,1,2), size=(h, w), mode='bicubic', align_corners=False)
                                # trilinear interpolation to interpolate the batch dimension
                                up_sub_attn_bicubic = up_sub_attn_bicubic.permute(0, 2, 3, 1).permute(3, 1, 2, 0).unsqueeze(0)
                                up_sub_attn_full = F.interpolate(up_sub_attn_bicubic, size=(h, w, nc), mode='trilinear', align_corners=False)
                                
                                up_sub_attn_full = up_sub_attn_full.squeeze(0).permute(3, 1, 2, 0)
                                layer_loss_out_in = get_layer_out_in_loss(prev_modified_sub_attn, up_sub_attn_full)
                            
                            else:
                                layer_loss_out_in = get_layer_out_in_loss(prev_modified_sub_attn, _sub_attn)

                            total_loss_out_in = total_loss_out_in + layer_loss_out_in
                            n_layers = n_layers + 1 
                
                # if max_cross_loss or diff_loss:
                # if extracted_attention_map_str!=None:
                if _sub_attn!=None:
                    xx = name.split('.')
                    layer_info = list(filter(lambda x:x.isdigit(), xx))
                    layer_id_info = ".".join(layer_info)

                    if skip_bkgd_layers:
                        if layer_id_info=='1.0.0':
                            if timestep_iter==0 and opt_idx==0:
                                print(f"skipping layer:{name}")
                            continue

                    token_ids = module.processor._all_tokens_inds

                    if max_cross_loss:
                        if _masked_map_wk_values!=None:
                            layer_loss_wk = get_layer_direct_sub_cross_loss(_masked_map_wk_values, _sub_attn, token_ids=token_ids, num_frames=num_frames)
                            layer_mean_attn_value_wk = get_layer_direct_sub_cross_attn_mean_values(_masked_map_wk_values, _sub_attn, token_ids=token_ids, num_frames=num_frames)
                            
                        layer_loss_str = get_layer_direct_sub_cross_loss(_masked_map_str_values, _sub_attn, token_ids=token_ids, num_frames=num_frames)
                        layer_mean_attn_value_str = get_layer_direct_sub_cross_attn_mean_values(_masked_map_str_values, _sub_attn, token_ids=token_ids, num_frames=num_frames)


                        if _masked_map_wk_values!=None:
                            total_loss_wk = total_loss_wk + layer_loss_wk
                            total_mean_attn_value_wk = total_mean_attn_value_wk + layer_mean_attn_value_wk

                        total_loss_str = total_loss_str + layer_loss_str
                        total_mean_attn_value_str = total_mean_attn_value_str + layer_mean_attn_value_str

                    if use_diff_loss:
                        layer_diff_loss = module.processor._loss
                        total_diff_loss = total_diff_loss + layer_diff_loss

                    n_layers = n_layers + 1 

        # normalize total (mean or sum) loss by the no of layers
        cross_atten_loss_wk = total_loss_wk/n_layers if n_layers!=0 else 0
        cross_atten_loss_str = total_loss_str/n_layers if n_layers!=0 else 0

        cross_atten_mean_attn_value_wk = total_mean_attn_value_wk/n_layers if n_layers!=0 else 0
        cross_atten_mean_attn_value_str = total_mean_attn_value_str/n_layers if n_layers!=0 else 0

        # sum over mean aggregation
        cross_atten_diff_loss = total_diff_loss/n_layers if n_layers!=0 else 0
        cross_atten_out_in_loss = total_loss_out_in # /n_layers if n_layers!=0 else 0

    else:
        if timestep_iter==0:
            print('extracting single-layer cross attention...')
        # layer = self.up_blocks[2].attentions[2].transformer_blocks[0].attn2.processor
        
        cross_atten_out_in_loss = 0
        
        # following boxdiff (16x16 layer)
        spatial_layer = self.up_blocks[2].attentions[2].transformer_blocks[0].attn2.processor
        temporal_layer = self.transformer_in.transformer_blocks[0].attn2.processor

        token_ids = spatial_layer._all_tokens_inds
        spatial_layer_loss_wk = get_layer_cross_loss(spatial_layer._cross_attention_map_wk, token_ids=token_ids, num_frames=num_frames, use_mean=use_mean)
        temporal_layer_loss_wk = get_layer_cross_loss(temporal_layer._cross_attention_map_wk, token_ids=token_ids, num_frames=num_frames, use_mean=use_mean)

        spatial_layer_loss_str = get_layer_cross_loss(spatial_layer._cross_attention_map_str, token_ids=token_ids, num_frames=num_frames, use_mean=use_mean)
        temporal_layer_loss_str = get_layer_cross_loss(temporal_layer._cross_attention_map_str, token_ids=token_ids, num_frames=num_frames, use_mean=use_mean)
        
        spatial_layer_diff_loss = spatial_layer._loss
        temporal_layer_diff_loss = temporal_layer._loss

        if use_mean:
            print('using mean losses')
            cross_atten_loss_wk = (spatial_layer_loss_wk + temporal_layer_loss_wk) * 0.5
            cross_atten_loss_str = (spatial_layer_loss_str + temporal_layer_loss_str) * 0.5
        else:
            print('using sum losses')
            cross_atten_loss_wk = (spatial_layer_loss_wk + temporal_layer_loss_wk) 
            cross_atten_loss_str = (spatial_layer_loss_str + temporal_layer_loss_str) 

        # uses mean
        cross_atten_diff_loss = (spatial_layer_diff_loss + temporal_layer_diff_loss) * 0.5


    if isinstance(cross_atten_loss_str, torch.Tensor):
        if cross_atten_loss_str.isnan().any() or cross_atten_loss_str.isinf().any():
            raise ValueError("cross_atten_loss_str contains NaN or Inf")

    return cross_atten_loss_wk, cross_atten_loss_str, cross_atten_mean_attn_value_wk, cross_atten_mean_attn_value_str, cross_atten_diff_loss, cross_atten_out_in_loss

def zero_gradient(tensor):
    if tensor.grad is not None:
        tensor.grad.zero_()

def compute_manual_grad(bundle, t_idx, loss, bboxes_ratios=None,strengthen_scale=None,
                        weaken_scale=None,latents=None, 
                        lr=None, sigma_series=None):
    # manually compute gradients with custom stepsize
    params = []
    params_id = {}

    if bundle.get('opt_box'):
        params_id['opt_box'] = len(params)
        params = params + [bboxes_ratios] 
        zero_gradient(bboxes_ratios)

    if bundle.get('opt_str_scale'):
        params_id['opt_str_scale'] = len(params)
        params = params + [strengthen_scale]
        zero_gradient(strengthen_scale)

    if bundle.get('opt_wk_scale'):
        params_id['opt_wk_scale'] = len(params)
        params = params + [weaken_scale]
        zero_gradient(weaken_scale)
        
    # NOTE: from small to large lr, shouldnt it be the opposite?
    # self.scheduler.sigmas[0:1000:num_inf_steps] ** 2 [0.0009, 0.0017, 0.0026, 0.0034, 0.0043, 0.0052, 0.0061]
    
    if bundle.get('opt_latents'):
        params_id['latents'] = len(params)
        params = params + [latents]
        zero_gradient(latents)

    # compute gradients together
    grads = torch.autograd.grad(loss.requires_grad_(True), params)
    assert len(params)==len(grads), 'the no of optimized variables should be equal to the no of items in gradient output'

    # with torch.no_grad():
    # prevent memory leaks; avoid tracking update rule in the computational graph
    if bundle.get('opt_box'):
        grads_opt_box = grads[params_id['opt_box']]
        bboxes_ratios = bboxes_ratios - grads_opt_box * float(lr)

    if bundle.get('opt_str_scale'):
        grads_opt_str_scale = grads[params_id['opt_str_scale']]
        strengthen_scale = strengthen_scale - grads_opt_str_scale * float(lr)

    if bundle.get('opt_wk_scale'):
        grads_opt_wk_scale = grads[params_id['opt_wk_scale']]
        weaken_scale = weaken_scale - grads_opt_wk_scale * float(lr)
    
    if bundle.get('opt_latents'):
        grads_latents = grads[params_id['latents']]
        latents = latents - grads_latents * sigma_series[t_idx] ** 2

    # grad_cond = torch.autograd.grad(loss.requires_grad_(True), [params])[0]
    # latents = latents - grad_cond * sigma_series[t_idx] ** 2
    # latents = latents - grad_cond * self.scheduler.sigmas[t_idx] ** 2

def monitor_statistics(latents):
    bsz, channel, frames, width, height = latents.shape
    mean = latents.mean().item()
    variance = latents.var().item()
    norm = torch.norm(latents.reshape(bsz, -1), dim=1).mean().item()
    return mean, variance, norm

def create_videos(i, t, opt_idx, n_opt_iterations, n_edit_steps,
                  attn_video_path, no_opt, time_bf_motion, injection,
                  spatial_caption, temporal_caption, temporal_cross_frame_caption
                  ):
    
    # if no_opt== False:
    #     if opt_idx < n_opt_iterations and i < (num_dd_spatial_steps):
    #         attn_video_str = attn_video_str.replace('attn_viz', 'opt_viz')
    #     else:
    #         # skip opt video for non-editing steps for now
    #         continue
    #     
    if os.path.exists(attn_video_path):

        if time_bf_motion:
            add_timestep_to_video_array(attn_video_path, spatial_caption, t, opt_idx, no_opt=no_opt, time_bf_motion=time_bf_motion)
            add_timestep_to_video_array(attn_video_path, temporal_caption, t, opt_idx, no_opt=no_opt, time_bf_motion=time_bf_motion)
            add_timestep_to_video_array(attn_video_path, temporal_cross_frame_caption, t, opt_idx, no_opt=no_opt, time_bf_motion=time_bf_motion)

        else:

            # try:
            create_motion_timestep_video(attn_video_path, spatial_caption, t, opt_idx, n_edit_steps, attn_type="spatial", inject=injection)


# Add forward hook to check device placement of inputs/outputs
def add_hooks(module, device):
    def hook_fn(module, input, output):
        for i, inp in enumerate(input):
            if isinstance(inp, torch.Tensor):
                if inp.device!=device:
                    print(f"{module.__class__.__name__} input {i}: {inp.device}")
        if isinstance(output, torch.Tensor):
            if output.device!=device:
                print(f"{module.__class__.__name__} output: {output.device}")
        elif isinstance(output, tuple):
            for i, out in enumerate(output):
                if isinstance(out, torch.Tensor):
                    if out.device!=device:
                        print(f"{module.__class__.__name__} output {i}: {out.device}")
    module.register_forward_hook(hook_fn)

# Apply hooks to specific layers
def apply_hooks(model, device):
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm) or isinstance(module, nn.GroupNorm) or isinstance(module, nn.Linear):
            add_hooks(module, device)

def is_model_on_gpu(model, device):
        for name, param in model.named_parameters():
            if param.device != device:
                print(f"Parameter {name} is on device {param.device}, expected device {device}")

def check_device(module, device):
    for name, param in module.named_parameters(recurse=True):
        if param.device != device:
            print(f"Parameter {name} is on device {param.device}, expected device {device}")

def turn_off_gradients_storage(model):
    # NOTE - turn off gradient still calculates gradients but but does not store it
    # storing gradients whose weights you wont update will simply fill up your GPU memory -> turn it off!
    for param in model.vae.parameters():
        param.requires_grad = False
    for param in model.text_encoder.parameters():
        param.requires_grad = False
    for param in model.unet.parameters():
        param.requires_grad = False
    # for name, param in model.unet.named_parameters(): print(f"Parameter name: {name}, requires_grad: {param.requires_grad}")


def print_layer_norm_params(unet):
    # Ensure all layer normalization layers have their weights set
    for name, module in unet.named_modules():
        if isinstance(module, torch.nn.LayerNorm):
            print(f"{name}: LayerNorm weights {module.weight.requires_grad} and biases {module.bias.requires_grad}")
            if module.weight is None or module.bias is None:
                print(f"Warning: {name} does not have weights or biases set.")

def print_attention_maps(unet, all_modules=False):
    for name, module in unet.named_modules():
        module_name = type(module).__name__

        if not all_modules:
            if module_name == "Attention" and "attn2" in name:
                if "temp_attentions" in name:
                    extracted_attention_map = module.processor.cross_attention_map
                    if extracted_attention_map!=None:
                        print(replace_method(name), extracted_attention_map.shape)
                else:
                    extracted_attention_map = module.processor.cross_attention_map
                    if extracted_attention_map!=None:
                        print(replace_method(name), extracted_attention_map.shape)
        else:
            print(replace_method(name), module_name)
            

# Using multiple replace calls
def replace_method(input_string):
    """faster fixed no of replacements; limited to future n replacements"""
    return (input_string
            .replace(".0", "[0]")
            .replace(".1", "[1]")
            .replace(".2", "[2]")
            .replace(".3", "[3]")
            .replace(".4", "[4]")
            .replace(".5", "[5]"))

# Using regex
pattern = re.compile(r'(\w+\.\w+(\.\d+)+(\.\w+)*)')

def regex_method(input_string):
    """scales flexibly to future n replacements; slower in fixed small no of replacements"""
    def replace_dot_with_bracket(match):
        parts = match.group(0).split('.')
        result = parts[0]
        for part in parts[1:]:
            if part.isdigit():
                result += f'[{part}]'
            else:
                result += f'.{part}'
        return result

    return pattern.sub(replace_dot_with_bracket, input_string)

def use_dd_temporal(unet, opt_idx=None, timestep=None, loss_dict={}, wandb_log=None,
                    n_opt_iterations=None, output_opt_viz_path="", edit_before_softmax=False,
                    use_grad_chkpt=None, use_reentrant=True, 
                    use=True):
    """ To determine using the temporal attention editing at a step
    """
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "Attention" and "attn2" in name:
            module.processor.use_dd_temporal = use
            module.processor.output_opt_viz_path=output_opt_viz_path
            module.processor.edit_before_softmax = edit_before_softmax

            if use_grad_chkpt!=None: module.processor.use_grad_chkpt = use_grad_chkpt
            if timestep!=None: module.processor.timestep = timestep
            if opt_idx!=None: module.processor.opt_id = opt_idx
            if n_opt_iterations!=None: module.processor.n_opt_iterations = n_opt_iterations
            if wandb_log!=None: module.processor.wandb_log = wandb_log
            if loss_dict!={}: module.processor.loss_dict = loss_dict
           


def use_dd(unet, opt_idx=None, timestep=None, loss_dict={}, wandb_log=None, 
           n_opt_iterations=None, output_opt_viz_path="", edit_before_softmax=False,
           use_grad_chkpt=None, use_reentrant=True, 
           use=True):
    """ To determine using the spatial attention editing at a step
    """
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        # if module_name == "CrossAttention" and "attn2" in name:
        if module_name == "Attention" and "attn2" in name:
            module.processor.use_dd = use
            module.processor.output_opt_viz_path=output_opt_viz_path
            module.processor.edit_before_softmax = edit_before_softmax

            if use_grad_chkpt!=None: module.processor.use_grad_chkpt = use_grad_chkpt
            if timestep!=None: module.processor.timestep = timestep
            if opt_idx!=None: module.processor.opt_id = opt_idx
            if n_opt_iterations!=None: module.processor.n_opt_iterations = n_opt_iterations
            if wandb_log!=None: module.processor.wandb_log = wandb_log
            if loss_dict!={}: module.processor.loss_dict = loss_dict
            


def initialization(unet, bundle, bbox_per_frame, vis, no_opt=True, time_bf_motion=False, use_bkgd_zero=False,
                    aggregate_str=None, use_trg_unscaled=False, chosen_temp_block="", 
                    temp_edit_at_low_res=False, 
                    # edit_before_softmax=False,
                    minimize_bkgd=False, 
                    debug=False, strengthen_scale=None, weaken_scale=None
                    ):
    log.info("Intialization")
    # note: inject a processor class (efficient object-oriented programming) into the diffuser's arch modules

    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "Attention" and "attn2" in name:
            
            module.processor.name = name
            module.processor.chosen_temp_block = chosen_temp_block

            bundle.get('sigma_strength')
            if "temp_attentions" in name:
                processor = InjecterProcessor(
                    bundle=bundle,
                    bbox_per_frame=bbox_per_frame,
                    strengthen_scale=strengthen_scale, # bundle["trailblazer"]["temp_strengthen_scale"],
                    weaken_scale= weaken_scale, # bundle["trailblazer"]["temp_weaken_scale"],
                    sigma_strength=bundle.get('sigma_strength'), 
                    clip_box_values=bundle.get('clip_box_values'), 
                    scale_local_foreground=bundle.get('scale_local_foreground'), 
                    box_with_gauss=bundle.get('box_with_gauss'),
                    gauss_only=bundle.get('gauss_only'), 
                    use_high_box_only= bundle.get('use_high_box_only'),
                    normalize_gauss= bundle.get('normalize_gauss'),
                    normalize_mask= bundle.get('normalize_mask'),
                    allow_edge_margin=bundle.get('allow_edge_margin'),
                    no_opt=no_opt,
                    aggregate_str=aggregate_str,
                    use_trg_unscaled = use_trg_unscaled,
                    chosen_temp_block=chosen_temp_block, 
                    temp_edit_at_low_res=temp_edit_at_low_res,
                    # edit_before_softmax=edit_before_softmax,
                    time_bf_motion=time_bf_motion,
                    use_bkgd_zero=use_bkgd_zero,
                    minimize_bkgd=minimize_bkgd,
                    is_text2vidzero=False,
                    name=name,
                    vis=vis,
                    debug=debug
                )
            else:
                processor = InjecterProcessor(
                    bundle=bundle,
                    bbox_per_frame=bbox_per_frame,
                    strengthen_scale= strengthen_scale, # bundle["trailblazer"]["spatial_strengthen_scale"],
                    weaken_scale= weaken_scale, # bundle["trailblazer"]["spatial_weaken_scale"],
                    sigma_strength=bundle.get('sigma_strength'),
                    clip_box_values=bundle.get('clip_box_values'), 
                    scale_local_foreground=bundle.get('scale_local_foreground'), 
                    box_with_gauss=bundle.get('box_with_gauss'),
                    gauss_only=bundle.get('gauss_only'),
                    use_high_box_only= bundle.get('use_high_box_only'),
                    normalize_gauss= bundle.get('normalize_gauss'),
                    normalize_mask= bundle.get('normalize_mask'),
                    allow_edge_margin=bundle.get('allow_edge_margin'),
                    no_opt=no_opt,
                    aggregate_str=aggregate_str,
                    use_trg_unscaled = use_trg_unscaled,
                    chosen_temp_block=chosen_temp_block, 
                    temp_edit_at_low_res=temp_edit_at_low_res,
                    # edit_before_softmax=edit_before_softmax,
                    time_bf_motion=time_bf_motion,
                    use_bkgd_zero=use_bkgd_zero,
                    minimize_bkgd=minimize_bkgd,
                    is_text2vidzero=False,
                    name=name,
                    vis=vis,
                    debug=debug
                )

            module.processor = processor
    log.info("Initialized")



"""
NOTE:
This function is intentionally delegated to the updated implementation in `TrailBlazer/Pipeline/Utils_origin.py`
to avoid code duplication between origin vs non-origin pipelines.
"""
# Lazy import to avoid unnecessary import-time coupling.
from .Utils_origin import keyframed_bbox as _keyframed_bbox
from .Utils_origin import keyframed_prompt_embeds as _keyframed_prompt_embeds


def keyframed_prompt_embeds(bundle, encode_prompt_func, device):
    return _keyframed_prompt_embeds(bundle, encode_prompt_func, device)

def keyframed_bbox(bundle):
    return _keyframed_bbox(bundle)
