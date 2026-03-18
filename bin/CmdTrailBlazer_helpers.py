

import os
import cv2
import pdb
import time
import glob
import torch
import wandb
import imageio
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import torch.nn.functional as F

from TrailBlazer.Misc import Const
from TrailBlazer.Misc import ConfigIO
from TrailBlazer.Misc import Logger as log

from bin.utils.plot_helpers import (draw_arrow, draw_bbox, save_cv2_image, write_video, delete_images,
                                plot2chk_image)
from bin.utils.misc import (
    get_bbox_midpoint,
    get_2d_orientation,
    convert_pt_to_image_space,
    convert_bbox_to_image_space,
    convert_to_int,
    args_to_str,
    time_taken,
    video_to_gif,
    convert_video,
)

from TrailBlazer.Pipeline.Utils import keyframed_bbox, keyframed_prompt_embeds


def get_unique_seed_from_config_path(config_path: str) -> int:
    """
    Determine a single deterministic seed from a config path.

    - If config_path is a YAML file: load and return its 'seed'.
    - If config_path is a directory: scan **/*.yaml, require every yaml has 'seed',
      require exactly one unique seed across them, and return it.

    No guessing/silent fallback: raises ValueError if missing/mixed/non-int seeds.
    """
    if not config_path:
        raise ValueError("config_path is empty")

    if os.path.isdir(config_path):
        yamls = sorted(glob.glob(os.path.join(config_path, "**", "*.yaml"), recursive=True))
        if len(yamls) == 0:
            raise ValueError(f"No YAML files found under config dir: {config_path}")

        seeds = set()
        missing = []
        non_int = []

        for y in yamls:
            cfg = ConfigIO.config_loader(y)
            if not isinstance(cfg, dict) or "seed" not in cfg:
                missing.append(y)
                continue
            seed = cfg["seed"]
            if not isinstance(seed, int):
                non_int.append((y, seed))
                continue
            seeds.add(seed)

        if missing:
            raise ValueError(
                "All YAMLs in config dir must define integer 'seed'. "
                f"Missing in {len(missing)} file(s), e.g. {missing[0]}"
            )
        if non_int:
            raise ValueError(
                "All YAMLs in config dir must define integer 'seed'. "
                f"Non-int seed in {len(non_int)} file(s), e.g. {non_int[0][0]} has seed={non_int[0][1]!r}"
            )
        if len(seeds) != 1:
            raise ValueError(
                "Expected exactly one unique seed across YAMLs in config dir. "
                f"Found seeds: {sorted(seeds)}"
            )

        return next(iter(seeds))

    cfg = ConfigIO.config_loader(config_path)
    if not isinstance(cfg, dict) or "seed" not in cfg:
        raise ValueError("--set_global_deterministic requires a 'seed' in the main config YAML.")
    seed = cfg["seed"]
    if not isinstance(seed, int):
        raise ValueError(f"--set_global_deterministic requires integer 'seed' in the main config YAML (got {seed!r}).")
    return seed


def resolve_set_global_deterministic(
    cli_value: str | None,
    shared_config_path: str | None,
    *,
    shared_key: str = "set_global_deterministic",
) -> bool:
    """
    Resolve tri-state determinism flag.

    Precedence:
    - CLI: cli_value in {"true","false"} overrides everything
    - Else: shared_config_path YAML key `shared_key` (bool) if present
    - Else: False

    No silent fallback:
    - If shared_config_path is provided but missing, raise FileNotFoundError.
    - If shared_config does not load to dict, raise ValueError.
    - If key exists but is not bool, raise ValueError.
    """
    if cli_value is not None:
        return cli_value.lower() == "true"

    if shared_config_path is None or shared_config_path == "":
        return False

    if not os.path.exists(shared_config_path):
        raise FileNotFoundError(
            f"shared_config not found at '{shared_config_path}'. "
            "Provide a valid --shared_config or pass --set_global_deterministic explicitly."
        )

    shared_cfg = ConfigIO.config_loader(shared_config_path)
    if not isinstance(shared_cfg, dict):
        raise ValueError(f"shared_config must load to a dict, got {type(shared_cfg)}")

    if shared_key not in shared_cfg:
        return False

    det = shared_cfg[shared_key]
    if not isinstance(det, bool):
        raise ValueError(f"shared_config key '{shared_key}' must be a bool")
    return det


def initialize_wandb(bundle):
        wandb_proj_name = bundle.get('wandb_project_name') 

        if bundle.get('debug'):
            wandb_proj_name = bundle.get('wandb_project_name') + "_debug" 

        if bundle.get('wandb_log'):
            # initialize wandb
            wandb_name = args_to_str(bundle, bundle.get('wandb_name'))

            if bundle.get('source_clip'):
                wandb_name += '_clip_{}'.format(bundle.get('source_clip'))
            if bundle.get('motion_id'):
                wandb_name += '_motion_id_{}'.format(bundle.get('motion_id'))

            wandb_name += '_no_opt_{}'.format(bundle.get('no_opt'))
            wandb_name += '_lr_{}'.format(bundle.get('lr'))
            wandb_name += '_bb_lambda_{}'.format(bundle.get('bb_deviate_lambda'))
            wandb_name += '_n_opt_iters_{}'.format(bundle.get('n_opt_iterations'))
            wandb_name += f"_{bundle.get('machine_name')}"
            
            # auto-generated run id
            run_id = wandb.util.generate_id()
            print(f"creating new run id: {run_id}")
            resume_type = "allow"

        if bundle.get('wandb_log'):
            wandb.init(
                        id=run_id,
                        project= wandb_proj_name, 
                        name=f"{wandb_name}",
                        resume=resume_type, 
                        config={
                        "timestamp": bundle.get('timestamp'),
                        "n_opt_iters": bundle.get('n_opt_iterations'),
                        "lr": bundle.get('lr'),
                        "no_opt": bundle.get('no_opt'),
                        "minimize_bkgd": bundle.get('minimize_bkgd'),
                        "use_bkgd_zero": bundle.get('use_bkgd_zero')
                    }
                    )

def overwrite_bundle_box(obj_bundle, optim_box_pt):
    # load meta data
    data = torch.load(optim_box_pt)
    optim_box = data["opt_bbox"]
    num_optim = optim_box.shape[0]

    # Backward compatible path: if counts already match, keep the old behavior (1:1 overwrite).
    if len(obj_bundle["keyframe"]) == num_optim:
        optim_box_list = optim_box.tolist()
        for i in range(num_optim):
            obj_bundle["keyframe"][i]["bbox_ratios"] = optim_box_list[i]

        # add a modification note
        obj_bundle["original_box_overwritten"] = True
        log.info("\n NEWS: original box is updated with an optimized box (1:1 keyframes)")
        return

    
    else:
        "da: sparse keyframes to be fed with full optimized boxes"

        # New path: if the config only has sparse keyframes, densify keyframes first by:
        # - interpolating bboxes to per-frame bboxes
        # - keep the original prompts at anchor frames and set prompts in non-anchor frames to None
        print(
            f"{len(obj_bundle['keyframe'])} input boxes is not equal to "
            f"{num_optim} optimized boxes --> ACTION TAKEN: interpolate and "
            f"fill up the missing boxes and prompts"
        )

        original_kfs = sorted(obj_bundle["keyframe"], key=lambda kf: kf["frame"])
        assert len(original_kfs) > 0, "obj_bundle['keyframe'] must contain at least one keyframe"
        original_anchor_frame_ids_and_prompts = {kf["frame"]: kf["prompt"] for kf in original_kfs}

        bbox_per_frame = keyframed_bbox(obj_bundle)  # list of [l, t, r, b]
        num_frames = len(bbox_per_frame)

        # No silent fallback: the interpolated length must match the optimized box length.
        assert (
            num_frames == num_optim
        ), f"Interpolated boxes ({num_frames}) must match optimized boxes ({num_optim})"

        dense_keyframes = []
        n_kf = len(original_kfs)
        kf_idx = 0

        for frame_idx in range(num_frames):
            while kf_idx + 1 < n_kf and frame_idx >= original_kfs[kf_idx + 1]["frame"]:
                kf_idx += 1

            injected_prompt = original_anchor_frame_ids_and_prompts.get(frame_idx, None)
            dense_keyframes.append(
                {
                    "bbox_ratios": bbox_per_frame[frame_idx],  # placeholder; overwritten below
                    "frame": frame_idx,
                    "prompt": injected_prompt, # default to None for non-anchor frames
                }
            )

        obj_bundle["keyframe"] = dense_keyframes

        # Now overwrite with optimized box and kept unrounded (same as original behavior above).
        optim_box_list = optim_box.tolist()
        for i in range(num_optim):
            obj_bundle["keyframe"][i]["bbox_ratios"] = optim_box_list[i]
        
        # add a modification note
        obj_bundle["original_box_overwritten"] = True
        log.info("\n NEWS: original sparse keyframes replaced with dense optimized boxes")

# command-line override final bundle
def cmd_override_final_bundle(cmd_args_bundle, base_bundle):

    # base zeroscope model
    if cmd_args_bundle['zeroscope_xl']:
        # turn off editing
        cmd_args_bundle['num_dd_spatial_steps'] = 0
        cmd_args_bundle['num_dd_temporal_steps'] = 0
        
    # For non-boolean flags, None would mean no change has been requested
    if 'n_opt_iterations' in base_bundle:
        if cmd_args_bundle['n_opt_iterations']!=None:
            base_bundle['n_opt_iterations'] = cmd_args_bundle['n_opt_iterations']

    if 'bb_deviate_lambda' in base_bundle:
        if cmd_args_bundle['bb_deviate_lambda']!=None:
            base_bundle['bb_deviate_lambda'] = cmd_args_bundle['bb_deviate_lambda']

    if 'lr' in base_bundle:
        if cmd_args_bundle['lr']!=None:
            base_bundle['lr'] = cmd_args_bundle['lr']

    if 'outside_bbox_loss_scale' in base_bundle:
        if cmd_args_bundle['outside_bbox_loss_scale']!=None:
            base_bundle['outside_bbox_loss_scale'] = cmd_args_bundle['outside_bbox_loss_scale']

    if 'inside_bbox_attn_loss_scale' in base_bundle:
        if cmd_args_bundle['inside_bbox_attn_loss_scale']!=None:
            base_bundle['inside_bbox_attn_loss_scale'] = cmd_args_bundle['inside_bbox_attn_loss_scale']

    if 'box_temp_smooth_scale' in base_bundle:
        if cmd_args_bundle['box_temp_smooth_scale']!=None:
            base_bundle['box_temp_smooth_scale'] = cmd_args_bundle['box_temp_smooth_scale']

    if 'height' in base_bundle:
        if cmd_args_bundle['height']!=None:
            base_bundle['height'] = cmd_args_bundle['height']
    
    if 'width' in base_bundle:
        if cmd_args_bundle['width']!=None:
            base_bundle['width'] = cmd_args_bundle['width']

    if 'num_inference_steps' in base_bundle:
        if cmd_args_bundle['num_inference_steps']!=None:
            base_bundle['num_inference_steps'] = cmd_args_bundle['num_inference_steps']
    
    if 'sigma_strength' in base_bundle:
        if cmd_args_bundle['sigma_strength']!=None:
            base_bundle['sigma_strength'] = cmd_args_bundle['sigma_strength']

    if 'wandb_name' in base_bundle:
        if cmd_args_bundle['wandb_name']!=None:
            base_bundle['wandb_name'] = cmd_args_bundle['wandb_name']

    if 'trailblazer' in base_bundle:
        if cmd_args_bundle['num_dd_spatial_steps']!=None:
            base_bundle['trailblazer']['num_dd_spatial_steps'] = cmd_args_bundle['num_dd_spatial_steps']

        if cmd_args_bundle['num_dd_temporal_steps']!=None:
            base_bundle['trailblazer']['num_dd_temporal_steps'] = cmd_args_bundle['num_dd_temporal_steps']

        if cmd_args_bundle['spatial_weaken_scale']!=None:
            base_bundle['trailblazer']['spatial_weaken_scale'] = cmd_args_bundle['spatial_weaken_scale']
        
        if cmd_args_bundle['spatial_strengthen_scale']!=None:
            base_bundle['trailblazer']['spatial_strengthen_scale'] = cmd_args_bundle['spatial_strengthen_scale']

        if cmd_args_bundle['temp_weaken_scale']!=None:
            base_bundle['trailblazer']['temp_weaken_scale'] = cmd_args_bundle['temp_weaken_scale']
        
        if cmd_args_bundle['temp_strengthen_scale']!=None:
            base_bundle['trailblazer']['temp_strengthen_scale'] = cmd_args_bundle['temp_strengthen_scale']

        # allow CLI override of trailing_length if provided
        if cmd_args_bundle.get('trailing_length') is not None:
            base_bundle['trailblazer']['trailing_length'] = cmd_args_bundle['trailing_length']

        
    # boolean flags 
    if 'wandb_log' in base_bundle:
        if base_bundle['wandb_log']!=cmd_args_bundle['wandb_log']:
            base_bundle['wandb_log'] = cmd_args_bundle['wandb_log']
    
    if 'box_temp_smooth_loss' in base_bundle:
        if base_bundle['box_temp_smooth_loss']!=cmd_args_bundle['box_temp_smooth_loss']:
            base_bundle['box_temp_smooth_loss'] = cmd_args_bundle['box_temp_smooth_loss']

    if 'init_bbox_area_loss' in base_bundle:
        if base_bundle['init_bbox_area_loss']!=cmd_args_bundle['init_bbox_area_loss']:
            base_bundle['init_bbox_area_loss'] = cmd_args_bundle['init_bbox_area_loss']

    if 'temp_edit_at_low_res'in base_bundle:
        if base_bundle['temp_edit_at_low_res']!=cmd_args_bundle['temp_edit_at_low_res']:
            base_bundle['temp_edit_at_low_res'] = cmd_args_bundle['temp_edit_at_low_res']


    if cmd_args_bundle['off_normalize_gauss']:
        base_bundle['normalize_gauss'] = False

    if cmd_args_bundle['on_clip_box_values']:
        base_bundle['clip_box_values'] = True

    if cmd_args_bundle['use_scale_local_foreground']:
        base_bundle['scale_local_foreground'] = True

    if cmd_args_bundle['off_normalize_mask']:
        base_bundle['normalize_mask'] = False

    if cmd_args_bundle.get("set_global_deterministic"):
        base_bundle["set_global_deterministic"] = True

    if cmd_args_bundle.get("vis_layer") is not None:
        base_bundle["vis_layer"] = cmd_args_bundle["vis_layer"]

    if cmd_args_bundle.get("fps") is not None:
        fps = cmd_args_bundle["fps"]
        if not isinstance(fps, int) or fps <= 0:
            raise ValueError(f"--fps must be a positive int (got {fps!r})")
        base_bundle["fps"] = fps

    # tri-state CLI overrides: None means "no override"
    if "vis_maps" in base_bundle and cmd_args_bundle.get("vis_maps") is not None:
        base_bundle["vis_maps"] = (cmd_args_bundle["vis_maps"] == "true")

    if "overlay_maps" in base_bundle and cmd_args_bundle.get("overlay_maps") is not None:
        base_bundle["overlay_maps"] = (cmd_args_bundle["overlay_maps"] == "true")
    

# def run_bundle(bundle, config, base_bundle, pipe, args, output_folder, start_time):
def run_bundle(bundle, config, pipe, args, output_folder, per_bundle_start_time):

    timestamp = args.timestamp
    # timestamp = bundle.get('timestamp')

    # Note: We use Const module in attention processor as well and that's why here
    Const.DEFAULT_HEIGHT = bundle["height"]
    Const.DEFAULT_WIDTH = bundle["width"]
    print(Const.DEFAULT_HEIGHT, Const.DEFAULT_WIDTH)
    num_inference_steps = (40
                           if not bundle.get("num_inference_steps") 
                           else bundle.get("num_inference_steps")
                           )
    generator = torch.Generator().manual_seed(bundle["seed"])
    
    
    # -----------------
    # where path is update to unqiue meta data
    if args.generate_data or args.validate:
        task_meta_name = os.path.splitext(config)[0].split("config/")[1]
    else:
        task_meta_name = os.path.splitext(os.path.basename(config))[0]

    if args.validate:
        if args.val_model_name not in ['trailblazer_origin', 'trailblazer_diff', 'optim', 'peakaboo', 'text2vidzero']:
            raise ValueError(f'{args.val_model_name} is not included.')
        task_meta_name = os.path.join(args.val_model_name, task_meta_name)
    
    output_path = Path(os.path.join(output_folder, task_meta_name))
    output_opt_viz_path = (output_path / f'{timestamp}/opt_viz')

    # -----------------
    # note: forward pass
    result = pipe(bundle=bundle, height=Const.DEFAULT_HEIGHT, width=Const.DEFAULT_WIDTH, generator=generator, 
                vis_maps=bundle.get('vis_maps'), vis_opt_bboxes=bundle.get('vis_opt_bboxes'), lr=bundle.get('lr'), 
                latent_lr=bundle.get('latent_lr'), no_opt=bundle.get('no_opt'), time_bf_motion=bundle.get('time_bf_motion'), 
                aggregate_str = bundle.get('aggregate_str'), use_trg_unscaled=bundle.get('use_trg_unscaled'),
                chosen_temp_block=bundle.get('chosen_temp_block'), temp_edit_at_low_res=bundle.get('temp_edit_at_low_res'),
                automatic_grad=bundle.get('automatic_grad'), edit_before_softmax=args.edit_before_softmax,
                debug=bundle.get('debug'), wandb_log=bundle.get('wandb_log'), focus_1channel_only=bundle.get('focus_1channel_only'),
                use_bkgd_zero=bundle.get('use_bkgd_zero'), minimize_bkgd=bundle.get('minimize_bkgd'), n_opt_iterations=bundle.get('n_opt_iterations'), 
                num_inference_steps=num_inference_steps, output_path=args.output_path,
                output_opt_viz_path=output_opt_viz_path)
    
    
    video_frames = result.frames
    # video_latent = result.latents
    video_opt_bboxes = result.opt_bboxes_motion
    overlap_maps = result.overlay_attn_maps
    # -----------------
    output_path.mkdir(parents=True, exist_ok=True)

    output_video_path = (output_path / f'{timestamp}/video')
    output_video_path.mkdir(parents=True, exist_ok=True)
    
    # avoid overwriting existing video?
    video_ext = "mp4" # "mp4"
    if os.path.exists(output_video_path):
        
        task_base_name = os.path.splitext(os.path.basename(output_video_path))[0]
        repeated = os.path.join(output_folder, task_meta_name,  timestamp, "video", task_base_name + f"*{video_ext}")
        num_reapts = len(glob.glob(repeated))
        output_video_path = os.path.join(output_folder, task_meta_name, timestamp, "video", task_base_name + f".{num_reapts:04d}.{video_ext}".format(num_reapts))
        # output_video_path = os.path.join(output_folder, task_meta_name, "video", task_meta_name + ".{:04d}.mp4".format(num_reapts))

    # save video frames
    frame_dir = output_path / f'{timestamp}/frames'
    frame_dir.mkdir(parents=True, exist_ok=True)
    if bundle.get('viz_orient2d'):
        frames_with_dir = output_path / f'{timestamp}/frames_with_dir'
        frames_with_dir.mkdir(parents=True, exist_ok=True)

    n_frames = len(video_frames)
    magenta_color, red_color = (255, 0, 255), (255, 0, 0)
    red_color_clipped = (0, 0, 255)

    # video writer definition
    fps = bundle.get("fps", 24)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Choose the codec (codec may vary based on your system and preferences)

    # display_w_size, display_h_size = 512, 512
    display_w_size, display_h_size = 320, 320

    if bundle['width']!=display_w_size:
        # assert bundle['width']==bundle['height'], 'you need to handle unequal resolution...'
        if bundle['width']!=bundle['height']:
            # then we cant resize to same W same H
            display_w_size, display_h_size = bundle['width'], bundle['height']

    # NOTE: to avoid confusion,
    # IMPORTANT: VideoWriter uses WxH; frames come in HW, stored and output in HW

    cv_video_raw = cv2.VideoWriter(output_video_path, fourcc, fps, (bundle['width'], bundle['height']))
    cv_video_disp = cv2.VideoWriter(output_video_path.replace('video.', 'video_disp.'), fourcc, fps, (display_w_size, display_h_size))
    cv_video_with_boxes = cv2.VideoWriter(output_video_path.replace('video.', 'video_with_bboxes.'), fourcc, fps, (display_w_size, display_h_size))
    cv_video_without_opt_box = cv2.VideoWriter(output_video_path.replace('video.', 'video_without_opt_box.'), fourcc, fps, (display_w_size, display_h_size))
    cv_video_with_boxes_with_attn = cv2.VideoWriter(output_video_path.replace('video.', 'video_with_bboxes_with_attn.'), fourcc, fps, (display_w_size, display_h_size))

    # save frame-results
    orient_vectors_2d = []
    img_paths = []

    if isinstance(result.opt_bbox_per_frame, list):
        bbox_per_frame = torch.tensor(result.opt_bbox_per_frame)
    else:
        bbox_per_frame = result.opt_bbox_per_frame

    init_bbox_per_frame = result.init_bbox_per_frame
    
    # rescale attention maps from low to peak values
    if overlap_maps!=None:
        max_vals, max_indices = overlap_maps.reshape(len(video_frames), -1).max(dim=1)
        overlap_maps = overlap_maps / max_vals[..., None, None]

    for idx, frame in enumerate(video_frames):
        
        # image
        im = Image.fromarray(frame) # im W,H for saving <- frame H, W
        img_path = frame_dir / f"image_{idx:08d}.jpeg"
        if bundle.get('save_frames'):
            im.save(img_path)
        img_paths.append(str(img_path))
        
        # attention map
        _attn_map = overlap_maps[idx] if overlap_maps!=None else None
        
        # NOTE: frm RGB to BGR format, important for cv2
        # input frame and box color are BGR; output video is RGB
        # frame H, W
        frame_only = frame[:,:,[2, 1, 0]].copy()#.transpose(1,0,2)
        frame_with_arrow = frame[:,:,[2, 1, 0]].copy()#.transpose(1,0,2) 
        frame_with_arrow_without_opt_box = frame[:,:,[2, 1, 0]].copy()
        frame_with_arrow_with_attn = frame[:,:,[2, 1, 0]].copy() 

        if _attn_map!=None:
            # upsample
            # please NOTE attn shape is HxW 
            # interpolation requires fp32
            _attn_map = _attn_map.float()
            attn_map = F.interpolate(_attn_map.unsqueeze(0).unsqueeze(0), size=(bundle['height'], bundle['width']), mode='bilinear', align_corners=False)
            attn_map = attn_map.squeeze(0)

            # add attention
            attn_map = attn_map.permute(1,2,0).numpy()
            # scale image to 0-1, add attention and clip values
            frame_with_arrow_with_attn_0_1 = frame_with_arrow_with_attn / 255.0
            
            alpha = 0.7
            frame_with_arrow_with_attn_0_1 = ((1-alpha) * frame_with_arrow_with_attn_0_1) + (alpha * attn_map)
            frame_with_arrow_with_attn_0_1 = np.clip(frame_with_arrow_with_attn_0_1, 0, 1)
            # scale image to 255
            frame_with_arrow_with_attn = (frame_with_arrow_with_attn_0_1 * 255).astype(np.uint8)

        bboxA = bbox_per_frame[idx:idx+1]
        if init_bbox_per_frame!=None:
            init_bboxA = init_bbox_per_frame[idx:idx+1]

        # compute pseudo-2d-orientation between 2 optimized boxes
        if idx < (n_frames-1):
            bboxB = bbox_per_frame[idx+1:idx+2]
            if init_bbox_per_frame!=None:
                init_bboxB = init_bbox_per_frame[idx+1:idx+2]
            
            orient_vec2d, _ = get_2d_orientation(bboxA, bboxB)
            orient_vectors_2d.append(orient_vec2d)

            if bundle.get('viz_orient2d'):
                from_pt = get_bbox_midpoint(bboxA)
                to_pt = from_pt + orient_vec2d

                from_pt_int = convert_pt_to_image_space(from_pt, bundle['width'], bundle['height'])
                to_pt_int = convert_pt_to_image_space(to_pt, bundle['width'], bundle['height'])
                draw_arrow(frame_with_arrow, from_pt_int.squeeze(dim=0).numpy(), to_pt_int.squeeze(dim=0).numpy(), thickness=3, color=red_color_clipped)
                draw_arrow(frame_with_arrow_without_opt_box, from_pt_int.squeeze(dim=0).numpy(), to_pt_int.squeeze(dim=0).numpy(), thickness=3, color=red_color_clipped)
                draw_arrow(frame_with_arrow_with_attn, from_pt_int.squeeze(dim=0).numpy(), to_pt_int.squeeze(dim=0).numpy(), thickness=3, color=red_color_clipped)
                
        bboxA_int = convert_bbox_to_image_space(bboxA, bundle['width'], bundle['height'])
        if init_bbox_per_frame!=None:
            init_bboxA_int = convert_bbox_to_image_space(init_bboxA, bundle['width'], bundle['height'])
        
        # draw optimized box
        orange_bgr = (0, 165, 255) 
        green_color = (0, 255, 0)
        draw_bbox(frame_with_arrow, bboxA_int.squeeze(dim=0).numpy(), thickness=2, color=orange_bgr)
        draw_bbox(frame_with_arrow_with_attn, bboxA_int.squeeze(dim=0).numpy(), thickness=2, color=orange_bgr)
        
        # draw initial user box
        if init_bbox_per_frame!=None:
            # grey_color = (128, 128, 128)
            muted_blue_color = (200, 150, 100)  # In BGR format, due to opencv frame read
            muted_red_bgr = (0, 0, 255) # (80, 80, 150)
            blue_color = (255, 0, 0)
            

            if bundle['user_box_color']=='blue':
                init_box_color=blue_color
            elif bundle['user_box_color']=='red':
                init_box_color=muted_red_bgr
            else: 
                raise NotImplementedError
            
            draw_bbox(frame_with_arrow, init_bboxA_int.squeeze(dim=0).numpy(), thickness=2, color=init_box_color)
            draw_bbox(frame_with_arrow_without_opt_box, init_bboxA_int.squeeze(dim=0).numpy(), thickness=2, color=init_box_color)
            draw_bbox(frame_with_arrow_with_attn, init_bboxA_int.squeeze(dim=0).numpy(), thickness=2, color=init_box_color)
                
        if bundle['width']!=display_w_size:
            assert bundle['width']==bundle['height'], 'you need to handle unequal resolution...'
            frame_only_disp = cv2.resize(frame_only, (display_w_size, display_h_size))
            frame_with_arrow = cv2.resize(frame_with_arrow, (display_w_size, display_h_size))
            frame_with_arrow_without_opt_box = cv2.resize(frame_with_arrow_without_opt_box, (display_w_size, display_h_size))
            frame_with_arrow_with_attn = cv2.resize(frame_with_arrow_with_attn, (display_w_size, display_h_size))
        else:
            frame_only_disp = frame_only.copy()

        if bundle.get('save_frames'):
            
            # save_cv2_image(frame_only, str(frames_with_dir / f"image_{idx:08d}.jpeg"))
            save_cv2_image(frame_with_arrow, str(frame_dir / f"image_{idx:08d}_with_box.jpeg"))
            save_cv2_image(frame_with_arrow_without_opt_box, str(frame_dir / f"image_{idx:08d}_without_opt_box.jpeg"))
            save_cv2_image(frame_with_arrow_with_attn, str(frame_dir / f"image_{idx:08d}_with_box_with_attn.jpeg"))
        
        if bundle.get('save_video'):
            # all stored as HW
            write_video(cv_video_raw, frame_only)
            write_video(cv_video_disp, frame_only_disp)
            write_video(cv_video_with_boxes, frame_with_arrow)
            write_video(cv_video_without_opt_box, frame_with_arrow_without_opt_box)
            write_video(cv_video_with_boxes_with_attn, frame_with_arrow_with_attn)
            
        # log first generated video frame to wandb
        if idx==0:
            if bundle['wandb_log']:
                assert isinstance(frame_with_arrow, np.ndarray), 'expects video frame in numpy array'
                # cast back to rgb for wandb
                rgb_image = cv2.cvtColor(frame_with_arrow, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(rgb_image.astype('uint8'))
                # omitted the wandb logging step here (as frame log is post-iterations)
                wandb.log({f"gen_video/frame_{idx}": wandb.Image(image_pil)})

    # compile video
    if bundle.get('save_video'):
        cv_video_raw.release()
        cv_video_disp.release()
        cv_video_with_boxes.release()
        cv_video_without_opt_box.release()
        cv_video_with_boxes_with_attn.release()

        # mp4 to gif
        v_path_no_boxes = output_video_path.replace('video.', 'video_disp.')
        v_path = output_video_path.replace('video.', 'video_with_bboxes.')
        v_path_without_opt_box = output_video_path.replace('video.', 'video_without_opt_box.')
        v_path_with_attn = output_video_path.replace('video.', 'video_with_bboxes_with_attn.')

        gif_path = f"{v_path.replace('.mp4', '.gif')}"
        gif_path_no_boxes = f"{v_path_no_boxes.replace('.mp4', '.gif')}"
        gif_path_without_opt_box = f"{v_path_without_opt_box.replace('.mp4', '.gif')}"
        gif_path_with_attn = f"{v_path_with_attn.replace('.mp4', '.gif')}"

        video_to_gif(v_path, gif_path, start_time=None, end_time=None, fps=fps)
        video_to_gif(v_path_no_boxes, gif_path_no_boxes, start_time=None, end_time=None, fps=fps)
        video_to_gif(v_path_without_opt_box, gif_path_without_opt_box, start_time=None, end_time=None, fps=fps)
        video_to_gif(v_path_with_attn, gif_path_with_attn, start_time=None, end_time=None, fps=fps)


    # save orientations as well to .pt
    orient_vectors_2d = torch.cat(orient_vectors_2d)

    # save opt bbboxes on white background
    if isinstance(video_opt_bboxes, np.ndarray):
        imageio.mimwrite(output_video_path.replace('video.', 'video_opt_bboxes.'), video_opt_bboxes, fps=fps)

    # store config, low-res latents, bbox data, etc in .pt  
    data = {
        "latents": result.latents.detach().cpu(), # 40, 4, 25, 40, 72 (DDIM steps, ?, n_frames, h, w)
        "bundle": bundle, # video yaml config
        "opt_bbox": result.opt_bbox_per_frame,
        "bbox": result.init_bbox_per_frame,
        "orient": orient_vectors_2d,
        "img_path": img_paths}

    if result.latents_x0_given_xt!=None:
        data["latents_x0_given_xt"] = result.latents_x0_given_xt.detach().cpu()

    latent_path = os.path.splitext(output_video_path)[0] + ".pt"
    torch.save(data, latent_path)
    config_path = os.path.splitext(output_video_path)[0] + ".yaml"
    ConfigIO.config_saver(bundle, config_path)
    run_config_path = os.path.splitext(output_video_path)[0].replace('video.', 'run.') + ".yaml"
    
    ConfigIO.config_saver(bundle, run_config_path)
    # ConfigIO.config_saver(base_bundle, run_config_path)

    log.info(latent_path)
    log.info(output_video_path)
    log.info(config_path)
    log.info(run_config_path)
    log.info("Done")

    seconds = time.time() - per_bundle_start_time
    total_time = time_taken(seconds)
    log.info(f"\n ---------- time_taken PER BUNDLE {total_time} ----------")
    
    return video_frames


def save_multi_video_metadata_and_gif(
    bundle, result, output_video_path: str
) -> None:
    """
    Helper for multi-subject runs driven by CmdTrailBlazerMulti:
    - Assumes the raw mp4 at `output_video_path` has already been written.
    - Creates a GIF next to it.
    - Saves latents, bundle, and bbox_per_frame to a .pt file.
    - Saves the bundle config to a .yaml file.
    """
    # mp4 -> gif
    v_path = output_video_path
    gif_path = f"{v_path.replace('.mp4', '.gif')}"
    video_to_gif(v_path, gif_path, start_time=None, end_time=None, fps=bundle.get("fps", 24))

    # Store config, latents, bbox data, etc in .pt
    data = {
        "latents": result.latents,
        "bundle": bundle,
        "bbox": getattr(result, "bbox_per_frame", None),
    }
    latent_path = os.path.splitext(v_path)[0] + ".pt"
    torch.save(data, latent_path)

    config_path = os.path.splitext(v_path)[0] + ".yaml"
    ConfigIO.config_saver(bundle, config_path)

    log.info(latent_path)
    log.info(v_path)
    log.info(config_path)
    log.info("Done")