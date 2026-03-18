
import os
import pdb
import time
import inspect
from tqdm import tqdm, trange
from typing import Any, Callable, Dict, List, Optional, Union

import gc
import torch
import torch.nn as nn
from PIL import Image
# import bitsandbytes as bnb
# identify exact operation breaking gradient flow computation issue
torch.autograd.set_detect_anomaly(True)

import wandb
import imageio
# import deepspeed
import numpy as np
from pathlib import Path
import torch.optim as optim
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.loaders import LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, UNet3DConditionModel
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    # deprecate,
    # logging,
    # replace_example_docstring,
    BaseOutput,
)
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_synth import (
    tensor2vid,
)
from TrailBlazer.CrossAttn.Utils import create_motion_timestep_video, add_timestep_to_video_array, create_timestep_motion_video, get_loss, get_params
from TrailBlazer.Misc.Painter import CrossAttnPainter


from ..Misc import Logger as log
from ..Misc import Const
from .Loss_factory import get_shrink_threshold, compute_loss
from bin.utils.misc import count_params
from .Utils import (initialization, keyframed_bbox, keyframed_prompt_embeds, use_dd, use_dd_temporal, 
                    monitor_statistics, replace_method, print_attention_maps, check_device, turn_off_gradients_storage, 
                    print_layer_norm_params, is_model_on_gpu, apply_hooks, create_videos, 
                    compute_manual_grad, zero_gradient, verify_gradient_chkpt, extract_overlap_maps)


import traceback
from glob import glob
# import torch.utils.checkpoint as checkpoint
# from torch.cuda.amp import autocast, GradScaler
from TrailBlazer.CrossAttn.Utils import time_taken
from bin.utils.plot_helpers import plot_bboxes, plot_bbox_gradients, plot2chk_image, save_image


# https://github.com/shubham-goel/4D-Humans/blob/6ec79656a23c33237c724742ca2a0ec00b398b53/hmr2/datasets/vitdet_dataset.py#L13
DEFAULT_MEAN = torch.tensor([0.485, 0.456, 0.406]) # * 255.
DEFAULT_STD = torch.tensor([0.229, 0.224, 0.225]) # * 255.

@dataclass
class TextToVideoSDPipelineOutput(BaseOutput):
    """
    Output class for text-to-video pipelines.

    Args:
        frames (`List[np.ndarray]` or `torch.FloatTensor`)
            List of denoised frames (essentially images) as NumPy arrays of shape `(height, width, num_channels)` or as
            a `torch` tensor. The length of the list denotes the video length (the number of frames).
    """

    frames: Union[List[np.ndarray], torch.FloatTensor]
    latents: Union[List[np.ndarray], torch.FloatTensor]
    latents_x0_given_xt: Union[List[np.ndarray], torch.FloatTensor]
    init_bbox_per_frame: torch.tensor
    opt_bbox_per_frame: torch.tensor
    opt_bboxes_motion: torch.tensor
    overlay_attn_maps: torch.tensor


def is_inject(bundle):
    return not (bundle['trailblazer']['spatial_strengthen_scale']==0 and bundle['trailblazer']['spatial_weaken_scale'] == 1 \
            and bundle['trailblazer']['temp_strengthen_scale']==0 and bundle['trailblazer']['temp_weaken_scale'] == 1)


def _update_latent(latents: torch.Tensor, loss: torch.Tensor, step_size: float) -> torch.Tensor:
        """ Update the latent according to the computed loss. 
        # https://github.com/showlab/BoxDiff/blob/c89f083d37e50be5fff427cde4026d027306b81d/pipeline/sd_pipeline_boxdiff.py#L297
        """
        grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents], retain_graph=True)[0]
        latents = latents - step_size * grad_cond
        return latents

# @torch.no_grad()
def text_to_video_sd_pipeline_call(
    self,
    bundle=None,
    # prompt: Union[str, List[str]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    # num_frames: int = 16,
    lr=1e-4,
    latent_lr=1e-4,
    no_opt=False,
    aggregate_str=None,
    use_trg_unscaled=False,
    chosen_temp_block="",
    temp_edit_at_low_res=False,
    edit_before_softmax=False,
    time_bf_motion=False,
    use_bkgd_zero=False,
    minimize_bkgd=False,
    n_opt_iterations = 5,
    focus_1channel_only=False,
    vis_maps=False,
    vis_opt_bboxes=False,
    debug=False,
    wandb_log=False,
    automatic_grad=True,
    num_inference_steps: int = 50,
    # num_dd_steps: int = 0,
    guidance_scale: float = 9.0,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "np",
    return_dict: bool = True,
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    callback_steps: int = 1,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    progress = None,
    output_path: Optional[str] = None, 
    output_opt_viz_path: Optional[str] = None, 
):
    r"""
    The call function to the pipeline for generation.

    Args:
        prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
        height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
            The height in pixels of the generated video.
        width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
            The width in pixels of the generated video.
        num_frames (`int`, *optional*, defaults to 16):
            The number of video frames that are generated. Defaults to 16 frames which at 8 frames per seconds
            amounts to 2 seconds of video.
        num_inference_steps (`int`, *optional*, defaults to 50):
            The number of denoising steps. More denoising steps usually lead to a higher quality videos at the
            expense of slower inference.
        guidance_scale (`float`, *optional*, defaults to 7.5):
            A higher guidance scale value encourages the model to generate images closely linked to the text
            `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts to guide what to not include in image generation. If not defined, you need to
            pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
        num_images_per_prompt (`int`, *optional*, defaults to 1):
            The number of images to generate per prompt.
        eta (`float`, *optional*, defaults to 0.0):
            Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
            to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
        generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
            A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
            generation deterministic.
        latents (`torch.FloatTensor`, *optional*):
            Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for video
            generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
            tensor is generated by sampling using the supplied random `generator`. Latents should be of shape
            `(batch_size, num_channel, num_frames, height, width)`.
        prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
            provided, text embeddings are generated from the `prompt` input argument.
        negative_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
            not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
        output_type (`str`, *optional*, defaults to `"np"`):
            The output format of the generated video. Choose between `torch.FloatTensor` or `np.array`.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~pipelines.text_to_video_synthesis.TextToVideoSDPipelineOutput`] instead
            of a plain tuple.
        callback (`Callable`, *optional*):
            A function that calls every `callback_steps` steps during inference. The function is called with the
            following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
        callback_steps (`int`, *optional*, defaults to 1):
            The frequency at which the `callback` function is called. If not specified, the callback is called at
            every step.
        cross_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
            [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

    Examples:

    Returns:
        [`~pipelines.text_to_video_synthesis.TextToVideoSDPipelineOutput`] or `tuple`:
            If `return_dict` is `True`, [`~pipelines.text_to_video_synthesis.TextToVideoSDPipelineOutput`] is
            returned, otherwise a `tuple` is returned where the first element is a list with the generated frames.
    """
    # ----------------------
    
    assert (
        len(bundle["keyframe"]) >= 2
    ), "Must be greater than 2 keyframes. Input {} keys".format(len(bundle["keyframe"]))

    assert (
        bundle["keyframe"][0]["frame"] == 0
    ), "First keyframe must indicate frame at 0, but given {}".format(
        bundle["keyframe"][0]["frame"]
    )

    if bundle["keyframe"][-1]["frame"] != 23:
        log.info(
            "It's recommended to set the last key to 23 to match"
            " the sequence length 24 used in training ZeroScope"
        )

    for i in range(len(bundle["keyframe"]) - 1):
        log.info
        assert (
            bundle["keyframe"][i + 1]["frame"] > bundle["keyframe"][i]["frame"]
        ), "The keyframe indices must be ordered in the config file, Sorry!"

    # device = self._execution_device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    device = self.unet.device
    model_dtype = self.unet.dtype
    # ----------------------

    
    # Check the device for the model components
    check_device(self.unet, device)
    check_device(self.vae, device)
    check_device(self.text_encoder, device)

    # # freeze gradients storage for all main components (until required buy opt)
    turn_off_gradients_storage(self)
    # ----------------------
    
    bundle["prompt_base"] = bundle["keyframe"][0]["prompt"]
    prompt = bundle["prompt_base"]
    #prompt += Const.POSITIVE_PROMPT
    num_frames = bundle["keyframe"][-1]["frame"] + 1
    num_dd_spatial_steps = bundle["trailblazer"]["num_dd_spatial_steps"]
    num_dd_temporal_steps = bundle["trailblazer"]["num_dd_temporal_steps"]

    # note: interpolate bboxes if required
    bbox_per_frame = keyframed_bbox(bundle)

    # create differentiable bboxes for initialization
    """KEY NOTE: the variable you are optimizing`v` should remain on fp32, 
    all large tensors e.g `a` and follow-up interactions e.g `(v.a).half()` should be on fp16"""
    
    bboxes_ratios = torch.Tensor(bbox_per_frame).to(device)
    init_bboxes_ratios = bboxes_ratios.clone()

    strengthen_scale = torch.tensor(bundle["trailblazer"]["spatial_strengthen_scale"]).to(device)
    weaken_scale = torch.tensor(bundle["trailblazer"]["spatial_weaken_scale"]).to(device)

    if no_opt==False: # means opt
        # computing gradients due to bbox takes 7 secs :) whether you use it or not
        if bundle.get('opt_box'):
            bboxes_ratios.requires_grad = True 
            print('\n optimizing bboxes... \n')
            box_ids = [0,12]
            print(f"Before opt: \n bbox {box_ids[0]}: {bboxes_ratios[box_ids[0]]} \n box {box_ids[1]}: {bboxes_ratios[box_ids[1]]}")

        if bundle.get('opt_str_scale'):
            assert bundle["trailblazer"]["spatial_strengthen_scale"]==bundle["trailblazer"]["temp_strengthen_scale"], 'you need to define separate spatial and temporal variables.'
            strengthen_scale.requires_grad = True 
            print('\n optimizing strengthen_scale... \n')
            print(f"Before opt: \n strengthen_scale {strengthen_scale}")

        if bundle.get('opt_wk_scale'):
            assert bundle["trailblazer"]["spatial_weaken_scale"]==bundle["trailblazer"]["temp_weaken_scale"], 'you need to define separate spatial and temporal variables.'
            weaken_scale.requires_grad = True 
            print(f"Before opt: weaken_scale {weaken_scale}")
        
        time.sleep(0.5)
        
    # else:
    #     strengthen_scale = bundle["trailblazer"]["temp_strengthen_scale"]
        # weaken_scale = bundle["trailblazer"]["temp_weaken_scale"]

    initialization(unet=self.unet, bundle=bundle, bbox_per_frame=bboxes_ratios, vis=vis_maps, 
                    no_opt=no_opt, time_bf_motion=time_bf_motion, use_bkgd_zero=use_bkgd_zero,
                    use_trg_unscaled=use_trg_unscaled,
                    aggregate_str=aggregate_str, chosen_temp_block=chosen_temp_block,
                    temp_edit_at_low_res=temp_edit_at_low_res, 
                    minimize_bkgd=minimize_bkgd, debug=debug, # focus_1channel_only=focus_1channel_only
                    strengthen_scale=strengthen_scale,
                    weaken_scale=weaken_scale
                    )
    
    from pprint import pprint
    log.info("Experiment parameters:")
    print("==========================================")
    pprint(bundle)
    print("==========================================")
    # 0. Default height and width to unet
    height = height or self.unet.config.sample_size * self.vae_scale_factor
    width = width or self.unet.config.sample_size * self.vae_scale_factor

    num_images_per_prompt = 1
    negative_prompt = Const.NEGATIVE_PROMPT
    # 1. Check inputs. Raise error if not correct
    # self.check_inputs(
    #     prompt,
    #     height,
    #     width,
    #     callback_steps,
    #     negative_prompt,
    #     prompt_embeds,
    #     negative_prompt_embeds,
    # )

    # # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    
    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier-free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0

    # 3. Encode input prompt
    text_encoder_lora_scale = (
        cross_attention_kwargs.get("scale", None)
        if cross_attention_kwargs is not None
        else None
    )


    # 4gb -> 5gb
    prompt_embeds, negative_prompt_embeds = keyframed_prompt_embeds(
        bundle, self.encode_prompt, device
    )
    

    # For classifier free guidance, we need to do two forward passes.
    # Here we concatenate the unconditional and text embeddings into a single batch
    # to avoid doing two forward passes
    if do_classifier_free_guidance:
        #negative prompt embedding (used for the unconditional part A of the model) contains what to not include in the image generation,
        # instead of passing pseudo zeros"""
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds]) # n_frame, 77, 1024

    
    # 4. Prepare timesteps
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps

    # 5. Prepare latent variables
    # note: initial latent (technically a noise) is generated here
    num_channels_latents = self.unet.config.in_channels # 4
    latents = self.prepare_latents(batch_size * num_images_per_prompt, num_channels_latents, num_frames, 
                                height, width, prompt_embeds.dtype, device, generator, latents) # 1, 4, 25, 64, 64

    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
    if bundle.get('store_latent_xo_xt'):
        extra_step_kwargs_xo_xt = self.prepare_extra_step_kwargs(generator, eta)

    # 7. Denoising loop
    # num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    # --------------------
  
    # --------------- optimization starts here --------------------
    use_mean = bundle.get('use_mean')
    use_diff_loss = bundle.get('use_diff_loss')
    max_cross_loss = bundle.get('max_cross_loss')
    shrink_gap_ratio=0.90

    if use_diff_loss:
        use_mean=True

    optimizer = None
    if no_opt==False:
        if bundle.get('opt_latents') or bundle.get('verify_grad_chkpt'):
            # optimized variable in fp32
            latents = latents.to(torch.float32)

        if automatic_grad and not bundle.get('opt_latents'):

            # Define single optimizer [without optimizing latents]
            params = []
            if bundle.get('opt_box'):
                params = params + [bboxes_ratios]  # input_scale
            if bundle.get('opt_str_scale'):
                params = params + [strengthen_scale]
            if bundle.get('opt_wk_scale'):
                params = params + [weaken_scale]

            # if verifying gradients with latents variable, make it differentiable
            if bundle.get('verify_grad_chkpt'):
                latents = latents.requires_grad_(True)
                params = params + [latents]
            
            optimizer = optim.Adam(params, lr=float(lr))
            print('SINGLE OPTIMIZER')
        else:
            print(f'computing manual gradients...')#; time(0.5)

    else:
        assert n_opt_iterations==1, 'no of iterations must be 1 for differentiable base model'

    if no_opt==False:
        init_bboxes_ratios = init_bboxes_ratios.detach()
        bboxes_shrink_thresh = get_shrink_threshold(init_bboxes_ratios, gap_ratio=shrink_gap_ratio)
        # bboxes_shrink_thresh = bboxes_shrink_thresh.detach()

    latents_at_steps = []
    latents_x0_given_xt_at_steps = []
    opt_bboxes_motion_clips = []

    overlay_attn_maps=None
    opt_bboxes_motion_video=None


    # -------------------------------
    num_inf_steps = num_inference_steps
    timesteps_ = timesteps

    
    # 7. Denoising loop
    num_warmup_steps = len(timesteps_) - num_inf_steps * self.scheduler.order
    # -------------------------------

    
    if bundle.get('use_grad_chkpt') and no_opt==False:
        print('using gradient checkpointing...'); time.sleep(0.5)

    
    # start_time = time.time()

    
    with self.progress_bar(total=num_inf_steps) as progress_bar:
        if progress is not None and hasattr(progress, "tqdm"):
            timesteps_ = progress.tqdm(timesteps_, desc="Processing")
        else:
            timesteps_ = tqdm(timesteps_, desc="Processing")

        i = 0
        for t_idx, t in enumerate(timesteps_):
            with torch.enable_grad():
                
                # track iterations within while loop
                opt_idx = 0
    
                if i == 0:
                    count_params(self.unet, name="unet")
                    count_params(self.vae, name="vae")
                    count_params(self.text_encoder, name="text_encoder")

                # --------------
                if vis_opt_bboxes:
                    if t_idx==0:
                        print('sticking with wxh format...'); time.sleep(1)
                    white_bkgd = np.ones((bundle['width'], bundle['height'], 3), dtype=np.uint8) * 255
                    bbox_on_white_img = plot_bboxes(bboxes_ratios.detach().cpu(), white_bkgd, 
                                                    opt_iter=opt_idx, abs_denoise_step=t, rel_denoise_step=t_idx, total_rel_denoise_step=num_inference_steps-1)
                    opt_bboxes_motion_clips.append(bbox_on_white_img)
                # --------------


                # pass attribute in across layers
                loss_dict = {
                    'diff_loss': bundle.get('use_diff_loss'),
                    'max_cross_loss': bundle.get('max_cross_loss'),
                    'out_in_loss': bundle.get('out_in_loss'),
                    'box_temp_smooth_loss': bundle.get('box_temp_smooth_loss')
                }

             
               
                         
                # 1st phase (edit steps)
                was_edited = False
                while (t_idx < (num_dd_spatial_steps) or t_idx < (num_dd_temporal_steps)) and opt_idx < n_opt_iterations: 
                    was_edited = True

                    # -----------------------------------
                    if bundle.get('opt_latents'):
                        latents = latents.requires_grad_(True)

                    # expand the latents if we are doing classifier free guidance; cast back to fp16 if used
                    latent_model_input = (torch.cat([latents] * 2) if do_classifier_free_guidance else latents).to(model_dtype)
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)  # output dim is same

                    if t_idx < (num_dd_spatial_steps):
                        use_dd(self.unet, opt_idx=opt_idx, timestep=t_idx, loss_dict=loss_dict, 
                            wandb_log=wandb_log, n_opt_iterations=n_opt_iterations,
                            use_grad_chkpt=bundle.get('use_grad_chkpt'), 
                            use_reentrant=bundle.get('use_reentrant'), 
                            edit_before_softmax=edit_before_softmax,
                            output_opt_viz_path=output_opt_viz_path, use=True)
                        
                    if t_idx < (num_dd_temporal_steps):
                        use_dd_temporal(self.unet, opt_idx=opt_idx, timestep=t_idx, loss_dict=loss_dict, 
                                        wandb_log=wandb_log, n_opt_iterations=n_opt_iterations,
                                        use_grad_chkpt=bundle.get('use_grad_chkpt'),
                                        use_reentrant=bundle.get('use_reentrant'), 
                                        edit_before_softmax=edit_before_softmax,
                                        output_opt_viz_path=output_opt_viz_path, use=True)


                    if bundle.get('opt_latents'):
                        assert latent_model_input.requires_grad, "latent_model_input does not require gradients"

                    # assert not latent_model_input.requires_grad
                    noise_pred, cross_attn_loss = self.unet(
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
                        use_grad_chkpt = bundle.get('use_grad_chkpt'),
                        use_reentrant = bundle.get('use_reentrant'),
                        out_in_loss = bundle.get('out_in_loss'),
                        max_cross_loss = bundle.get('max_cross_loss'),
                        use_diff_loss = bundle.get('use_diff_loss')
                        )
  
                    
                    # TODO: please MOVE to a new function
                    # ---------------------
                    if no_opt==False:
                        assert num_dd_temporal_steps==num_dd_spatial_steps, 'num_dd_temporal_steps is not equal to num_dd_spatial_steps'
                        
                        # do house cleaning early
                        if automatic_grad:
                            optimizer.zero_grad()
                            
                        loss = compute_loss(cross_attn_loss, bundle, bboxes_ratios, 
                                            init_bboxes_ratios=init_bboxes_ratios, loss_dict=loss_dict,
                                            t_idx=t_idx, t=t, opt_idx=opt_idx, n_opt_iterations=n_opt_iterations,
                                            strengthen_scale=strengthen_scale, weaken_scale=weaken_scale, 
                                            outside_bbox_loss_scale = bundle['outside_bbox_loss_scale'],
                                            inside_bbox_attn_loss_scale = bundle['inside_bbox_attn_loss_scale'],
                                            box_temp_smooth_scale = bundle['box_temp_smooth_scale'],
                                            box_flip_thresh = bundle['box_flip_thresh'],
                                            box_flip_thresh_scale = bundle['box_flip_thresh_scale'],
                                            use_mean=use_mean, wandb_log=wandb_log)
                        
                        if hasattr(timesteps_, "set_postfix"):
                            timesteps_.set_postfix(EDIT="True", t_idx=t_idx, opt_idx=opt_idx, LOSS=f"{loss:.4f}")
                        else:
                            print(f'EDIT: denoise t_idx {t_idx} opt_idx {opt_idx} LOSS {loss} ')

                        if automatic_grad:
                            try:
                                # torch.cuda.empty_cache()
                                loss.backward()

                                if bundle.get('verify_grad_chkpt'):

                                    assert bundle.get('use_grad_chkpt'), 'gradient checkpointing needs to be turned ON for verification'
                                    init_latents_grad = latents.grad.clone()
                                    print(f'latents norm {torch.norm(latents)} latents grad norm {torch.norm(init_latents_grad)}')

                                    zero_gradient(latents)
                                    print('init_latents_grad before zero_gradient', init_latents_grad[0,0,0,0,0].item(), 'after zero_gradient', latents.grad[0,0,0,0,0].item())
                        
                            except Exception as error:
                                print(f'error: {error}')
                                traceback.print_exc()
                                pdb.set_trace()

                            if bundle.get('verify_grad_chkpt') and bundle.get('use_grad_chkpt'):
                                if t_idx==0 and opt_idx==0:
                                    verify_gradient_chkpt(self, bundle, init_latent_model_input=latent_model_input, 
                                                          t=t, t_idx=t_idx, opt_idx=opt_idx, 
                                                          prompt_embeds=prompt_embeds, 
                                                        cross_attention_kwargs=cross_attention_kwargs, 
                                                        use_mean=use_mean, 
                                                        bboxes_ratios=bboxes_ratios, init_bboxes_ratios=init_bboxes_ratios,
                                                        loss_dict=loss_dict, 

                                                        strengthen_scale=strengthen_scale, weaken_scale=weaken_scale, 
                                                        wandb_log=wandb_log, latents_clone=latents,
                                                        init_latents_grad=init_latents_grad, 
                                                        optimizer=optimizer,
                                                        do_classifier_free_guidance=do_classifier_free_guidance
                                                        )


                        
                        if bundle.get('opt_box'):   
                            pass    

                        if bundle.get('opt_str_scale'):
                            print(f"strengthen_scale: {strengthen_scale}, strengthen_scale grad {strengthen_scale.grad}")
                        if bundle.get('opt_wk_scale'):
                            print(f"weaken_scale: {weaken_scale}, weaken_scale grad {weaken_scale.grad}")
                        
                        if automatic_grad:
                            optimizer.step() 

                        torch.cuda.empty_cache()

                    # projected gradient descent - clip box values to 0-1 bounds
                    # .data modifies values in-place without interfering with pytorch gradient tracking
                    if bundle.get('proj_grad_descent'):
                        # always-on projection
                        bboxes_ratios.data = torch.clamp(bboxes_ratios.data, min=0.0, max=1.0)
                        assert ((bboxes_ratios >= 0).all() and (bboxes_ratios <= 1).all()), 'out-of-bound box values are not clipped to 0-1'
                            
                    # track iterations within while loop
                    opt_idx += 1 
                    
                    # ---------------------
                    
            if was_edited==True:
                current_memory_usage = round(torch.cuda.max_memory_allocated() / 1e9)
                # print(f"Step {t_idx}: Current max GPU memory usage: {current_memory_usage} GB")
            
            # NON-EDIT STEPS
            # ------------------------------
            if was_edited==False:
                
                # -----------------
                # after total forward editing (with/without box optimization) is done
                if bundle['overlay_maps']: # and no_opt==False:
                    overlap_maps = extract_overlap_maps(self.unet, overlay_attn_maps)
                    if overlap_maps!=None: 
                        overlay_attn_maps = overlap_maps 
                # -----------------

                # reset 
                opt_idx = 0

                # turn off gradients for 2nd phase (non-edit steps)
                with torch.no_grad(): # or set all optimized variables to False

                    # 'TENTATIVE remove?'
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = (torch.cat([latents] * 2) if do_classifier_free_guidance else latents).half()
                    # note: added noise?
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)  # output dim is same

                    
                    # TURN OFF editing
                    use_dd(self.unet, opt_idx=opt_idx, timestep=t_idx, wandb_log=wandb_log, 
                           n_opt_iterations=n_opt_iterations, use_grad_chkpt=False, 
                            # edit_before_softmax=edit_before_softmax,
                            output_opt_viz_path=output_opt_viz_path, use=False)
                    
                    use_dd_temporal(self.unet, opt_idx=opt_idx, timestep=t_idx, wandb_log=wandb_log,
                                    n_opt_iterations=n_opt_iterations, use_grad_chkpt=False, 
                                    #  edit_before_softmax=edit_before_softmax,
                                     output_opt_viz_path=output_opt_viz_path, use=False)

                    
                    noise_pred, cross_attn_loss = self.unet( 
                        latent_model_input, # 2, 4, 24, 16, 16
                        t,
                        timestep_iter = t_idx,
                        opt_idx = opt_idx,
                        encoder_hidden_states=prompt_embeds, # e.g 50, 77, 1024 (n_framex2, 77-dim since its clip embedding)
                        cross_attention_kwargs=cross_attention_kwargs,
                        return_dict=False,

                        skip_bkgd_layers = bundle.get('skip_bkgd_layers'),
                        specific_layers = bundle.get('specific_layers'),
                        use_mean=use_mean,
                        single_layer = bundle.get('single_layer'),
                        get_loss = False,
                        use_grad_chkpt = False,
                        # use_reentrant = False,
                        out_in_loss = bundle.get('out_in_loss'),
                        max_cross_loss = bundle.get('max_cross_loss'),
                        use_diff_loss = bundle.get('use_diff_loss')
                        ) 
                    


            # --------------------------------------------------------------
            # PERFORM GUIDANCE AND DENOISE
            
            if do_classifier_free_guidance:
                # note: both output from uncond and cond part come out
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # reshape latents
            bsz, channel, frames, width, height = latents.shape

            latents = latents.permute(0, 2, 1, 3, 4).reshape(
                bsz * frames, channel, width, height
            )
            
            try:
                noise_pred = noise_pred.permute(0, 2, 1, 3, 4).reshape(
                    bsz * frames, channel, width, height
                )
            except:
                pdb.set_trace()

            # if latents.requires_grad:

            # DENOISE by computing the previous noisy sample x_t -> x_t-1
            """key NOTE: denoising i.e `scheduler.step` is not applied within each optimization `while` loop above, but only after the optimization loop"""
            # print(f'-------- t {t}')
            curr_latents = latents.clone()
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            # DENOISE by computing the previous noisy sample x_t -> x_0
            if bundle.get('store_latent_xo_xt'):
                fixed_x0_step = timesteps_[-1]
                # print(f'-------- fixed_x0_step {fixed_x0_step}')
                latents_x0_given_xt = self.scheduler.step(noise_pred, fixed_x0_step, curr_latents.detach(), **extra_step_kwargs_xo_xt).prev_sample

            # under enable grad, but please detach if not optimizing latent
            if not bundle.get('opt_latents'):
                latents = latents.detach()
                if bundle.get('store_latent_xo_xt'):
                    latents_x0_given_xt = latents_x0_given_xt.detach()

            torch.cuda.empty_cache()
            

            # reshape latents back
            latents = (
                latents[None, :]
                .reshape(bsz, frames, channel, width, height)
                .permute(0, 2, 1, 3, 4)
            )
            if bundle.get('store_latent_xo_xt'):
                    latents_x0_given_xt = (
                    latents_x0_given_xt[None, :]
                    .reshape(bsz, frames, channel, width, height)
                    .permute(0, 2, 1, 3, 4)
                )

            if wandb_log:
                # Monitor statistics
                mean, variance, norm = monitor_statistics(latents.detach())
                if wandb_log:
                    track_dict = {
                    "track/latents_mean": mean,
                    "track/latents_variance": variance,
                    "track/latents_norm": norm
                    }
                    if was_edited==True:
                        opt_idx_ = opt_idx-1
                        log_step = (t_idx * n_opt_iterations) + opt_idx_
                        wandb.log(track_dict, step=log_step)
                    

            # store latents
            latents_at_steps.append(latents.half())
            if bundle.get('store_latent_xo_xt'):
                latents_x0_given_xt_at_steps.append(latents_x0_given_xt.half())
            
            # torch.cuda.empty_cache()

            # call the callback, if provided
            if i == len(timesteps_) - 1 or (
                (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
            ):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)
            i += 1

    print(f"\n********** Overall Peak GPU memory usage: {round(torch.cuda.max_memory_allocated() / 1e9)} GB\n")

    # return back to fp16
    if bundle.get('opt_latents'):
        latents = latents.half()
    
    if output_type == "latent":
        return TextToVideoSDPipelineOutput(frames=latents)

    if latents.requires_grad:
        latents = latents.detach()
        
    video_tensor = self.decode_latents(latents)
    
    if output_type == "pt":
        video = video_tensor
    else:
        video = tensor2vid(video_tensor)

    if vis_opt_bboxes:
        path2save = f'checkers/imgs'
        filename = f'{path2save}/optimized_bboxes_n_opt_iter_{n_opt_iterations}_edit_steps_{num_dd_spatial_steps}.mp4' 
        opt_bboxes_motion_video = np.stack(opt_bboxes_motion_clips, axis=0)

    # Offload all models
    self.maybe_free_model_hooks()

    if not return_dict:
        return (video,)

    # collate
    latents_at_steps = torch.cat(latents_at_steps)
    if bundle.get('store_latent_xo_xt'):
        latents_x0_given_xt_at_steps = torch.cat(latents_x0_given_xt_at_steps)
    else:
        latents_x0_given_xt_at_steps = None

    init_bbox_per_frame = init_bboxes_ratios.detach().cpu()
    if no_opt==False:
        bbox_per_frame=bboxes_ratios.detach().cpu()

    return TextToVideoSDPipelineOutput(frames=video, 
                                       latents=latents_at_steps, 
                                       latents_x0_given_xt = latents_x0_given_xt_at_steps,
                                       init_bbox_per_frame = init_bbox_per_frame,
                                       opt_bbox_per_frame = bbox_per_frame,
                                       opt_bboxes_motion=opt_bboxes_motion_video,
                                       overlay_attn_maps=overlay_attn_maps
                                       )
