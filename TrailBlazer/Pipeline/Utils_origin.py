import inspect
from typing import Any, Callable, Dict, List, Optional, Union
import pdb

import numpy as np
import torch
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
from ..CrossAttn.InjecterProc_origin import InjecterProcessor
from ..Misc import Logger as log
from ..Misc import Const




def use_dd_temporal(unet, timestep=None, wandb_log=None, output_opt_viz_path="", use=True):
    """ To determine using the temporal attention editing at a step
    """
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "Attention" and "attn2" in name:
            if use!=None: module.processor.use_dd_temporal = use
            if output_opt_viz_path!="": module.processor.output_opt_viz_path=output_opt_viz_path

            if timestep!=None: module.processor.timestep = timestep
            if wandb_log!=None: module.processor.wandb_log = wandb_log


def use_dd(unet, timestep=None, wandb_log=None, output_opt_viz_path="", use=True):
    """ To determine using the spatial attention editing at a step
    """
    # print(f'use_dd timestep {timestep}')
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        # if module_name == "CrossAttention" and "attn2" in name:
        if module_name == "Attention" and "attn2" in name:
            if use!=None: module.processor.use_dd = use
            if output_opt_viz_path!="": module.processor.output_opt_viz_path=output_opt_viz_path

            if timestep!=None: module.processor.timestep = timestep
            if wandb_log!=None: module.processor.wandb_log = wandb_log


def initiailization(unet, bundle, bbox_per_frame):
    log.info("Intialization")

    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "Attention" and "attn2" in name:
            if "temp_attentions" in name:
                processor = InjecterProcessor(
                    bundle=bundle,
                    bbox_per_frame=bbox_per_frame,
                    strengthen_scale=bundle["trailblazer"]["temp_strengthen_scale"],
                    weaken_scale=bundle["trailblazer"]["temp_weaken_scale"],
                    is_text2vidzero=False,
                    name=name,
                )
            else:
                processor = InjecterProcessor(
                    bundle=bundle,
                    bbox_per_frame=bbox_per_frame,
                    strengthen_scale=bundle["trailblazer"]["spatial_strengthen_scale"],
                    weaken_scale=bundle["trailblazer"]["spatial_weaken_scale"],
                    is_text2vidzero=False,
                    name=name,
                )
            module.processor = processor
            # print(name)
    log.info("Initialized")


def keyframed_prompt_embeds(bundle, encode_prompt_func, device):
    # note: linearly interpolate prompt embeddings
    
    num_frames = bundle["keyframe"][-1]["frame"] + 1
    keyframe = bundle["keyframe"]
    f = lambda start, end, index: (1 - index) * start + index * end
    n = len(keyframe)
    keyed_prompt_embeds = []
    all_prompts = [kf["prompt"] for kf in keyframe]

    if len(keyframe) == num_frames and len(set(all_prompts)) == 1:
        print('\nfull frame bboxes provided + frame prompts are the same\n')
        for i in range(n):
            prompt = keyframe[i]["prompt"] + Const.POSITIVE_PROMPT

            prompt_embeds, negative_prompt_embeds = encode_prompt_func(
                prompt,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=Const.NEGATIVE_PROMPT,
            )
            keyed_prompt_embeds.append(prompt_embeds)
    
    # NEW CODE: handle sparse keyframes + refeed full optim + different prompts
    elif len(keyframe) == num_frames and len(set(all_prompts)) != 1:
        print('\nfull frame bboxes provided, but frame prompts are different\n')
        anchor_indices = [i for i, prompt in enumerate(all_prompts) if prompt is not None]

        for i in range(len(anchor_indices) - 1):
            if i == 0:
                start_fr = anchor_indices[i]  
            else: 
                start_fr = anchor_indices[i] + 1
            end_fr = anchor_indices[i + 1]
            clip_length = end_fr - start_fr + 1

            start_prompt = keyframe[anchor_indices[i]]["prompt"] + Const.POSITIVE_PROMPT
            end_prompt = keyframe[anchor_indices[i + 1]]["prompt"] + Const.POSITIVE_PROMPT
            
            print(f'start_fr {start_fr} end_fr {end_fr}')
            print(f'start_prompt {start_prompt} end_prompt {end_prompt}')
            print(f'clip_length {clip_length}')

            start_prompt_embeds, _ = encode_prompt_func(
                start_prompt,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=Const.NEGATIVE_PROMPT,
            )

            end_prompt_embeds, negative_prompt_embeds = encode_prompt_func(
                end_prompt,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=Const.NEGATIVE_PROMPT,
            )

            for fr in range(clip_length):
                index = float(fr) / (clip_length - 1)
                keyed_prompt_embeds.append(f(start_prompt_embeds, end_prompt_embeds, index))

    else:
        print('\nsparse bboxes frames provided + sparse prompts\n')

        for i in range(n - 1):
            if i == 0:
                start_fr = keyframe[i]["frame"]
            else:
                start_fr = keyframe[i]["frame"] + 1
            end_fr = keyframe[i + 1]["frame"]

            start_prompt = keyframe[i]["prompt"] + Const.POSITIVE_PROMPT
            end_prompt = keyframe[i + 1]["prompt"] + Const.POSITIVE_PROMPT
            clip_length = end_fr - start_fr + 1

            start_prompt_embeds, _ = encode_prompt_func(
                start_prompt,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=Const.NEGATIVE_PROMPT,
            )

            end_prompt_embeds, negative_prompt_embeds = encode_prompt_func(
                end_prompt,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=Const.NEGATIVE_PROMPT,
            )

            for fr in range(clip_length):
                index = float(fr) / (clip_length - 1)
                keyed_prompt_embeds.append(f(start_prompt_embeds, end_prompt_embeds, index))
    
    assert len(keyed_prompt_embeds) == num_frames
    return torch.cat(keyed_prompt_embeds), negative_prompt_embeds.repeat_interleave(num_frames, dim=0)


def keyframed_bbox(bundle):
    
    keyframe = bundle["keyframe"]
    bbox_per_frame = []
   
    last_frame_id = keyframe[-1]['frame']

    if last_frame_id == len(keyframe) - 1:
        print('full bboxes frames provided')
        for box_dict in keyframe:
            bbox_per_frame.append(box_dict['bbox_ratios'])

    else:
        print(f'linearly interpolate bboxes')
        f = lambda start, end, index: (1 - index) * start + index * end
        n = len(keyframe)
        for i in range(n - 1):
            if i == 0:
                start_fr = keyframe[i]["frame"]
            else:
                start_fr = keyframe[i]["frame"] + 1
            end_fr = keyframe[i + 1]["frame"]
            start_bbox = keyframe[i]["bbox_ratios"]
            end_bbox = keyframe[i + 1]["bbox_ratios"]
            print(f'start_fr {start_fr} end_fr {end_fr}')
            clip_length = end_fr - start_fr + 1
            for fr in range(clip_length):
                index = float(fr) / (clip_length - 1)
                bbox = []
                for j in range(4):
                    bbox.append(f(start_bbox[j], end_bbox[j], index))
                bbox_per_frame.append(bbox)
    
    return bbox_per_frame

# def keyframed_bbox(bundle):

#     keyframe = bundle["keyframe"]
#     bbox_per_frame = []
#     f = lambda start, end, index: (1 - index) * start + index * end
#     n = len(keyframe)
#     for i in range(n - 1):
#         if i == 0:
#             start_fr = keyframe[i]["frame"]
#         else:
#             start_fr = keyframe[i]["frame"] + 1
#         end_fr = keyframe[i + 1]["frame"]
#         start_bbox = keyframe[i]["bbox_ratios"]
#         end_bbox = keyframe[i + 1]["bbox_ratios"]
#         clip_length = end_fr - start_fr + 1
#         for fr in range(clip_length):
#             index = float(fr) / (clip_length - 1)
#             bbox = []
#             for j in range(4):
#                 bbox.append(f(start_bbox[j], end_bbox[j], index))
#             bbox_per_frame.append(bbox)

#     return bbox_per_frame

