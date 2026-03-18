
#!/usr/bin/env pyton
import argparse
import copy
import os
import pdb
import cv2
import glob
import time
import torch
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.pipelines import TextToVideoSDPipeline
from diffusers.utils import export_to_video
from PIL import Image
from pathlib import Path

from TrailBlazer.Misc import ConfigIO
from TrailBlazer.Setting import Keyframe
from TrailBlazer.Misc import Logger as log
from TrailBlazer.Misc import Const


from TrailBlazer.Pipeline.TextToVideoSDPipelineCall_origin import text_to_video_sd_pipeline_call
from TrailBlazer.Pipeline.UNet3DConditionModelCall_origin import unet3d_condition_model_forward

TextToVideoSDPipeline.__call__ = text_to_video_sd_pipeline_call
from diffusers.models.unet_3d_condition import UNet3DConditionModel

unet3d_condition_model_forward_copy = UNet3DConditionModel.forward
UNet3DConditionModel.forward = unet3d_condition_model_forward

import pyrootutils
from bin.utils.plot_helpers import (draw_arrow, draw_bbox, save_cv2_image, 
                                    write_video, delete_images, plot2chk_image)
from bin.utils.misc import (convert_bbox_to_image_space, get_topk_yamls, 
                        drop_invalid_yamls, video_to_gif, convert_video,
                        time_taken)
from bin.CmdTrailBlazer_helpers import cmd_override_final_bundle, overwrite_bundle_box


# root = pyrootutils.setup_root(
#     search_from=__file__,
#     indicator=[".git", "pyproject.toml"],
#     pythonpath=True,
#     dotenv=True,
# )

# # set manual cache directory for stable diffusion
# cache_dir = f"{root}/.cache"
# os.makedirs(cache_dir,exist_ok=True)

# # change cache location to a directory with available space
# os.environ['XDG_CACHE_HOME'] = cache_dir # works

def run(bundle, config, pipe, args, output_folder, per_bundle_start_time):
    # start_time = time.time()
    timestamp = args.timestamp

    # Note: We use Const module in attention processor as well and that's why here
    Const.DEFAULT_HEIGHT = bundle["height"]
    Const.DEFAULT_WIDTH = bundle["width"]
    print(Const.DEFAULT_HEIGHT, Const.DEFAULT_WIDTH)
    num_inference_steps = (
        40
        if not bundle.get("num_inference_steps")
        else bundle.get("num_inference_steps")
    )
    generator = torch.Generator().manual_seed(bundle["seed"])

    # -----------------
    if args.validate:
        task_name = os.path.splitext(config)[0].split("config/")[1]

    else:
        task_name = os.path.splitext(os.path.basename(config))[0]

    if args.validate:
        if args.val_model_name not in ['trailblazer_origin', 'optim', 'peakaboo', 'text2vidzero']:
            raise ValueError(f'{args.val_model_name} is not included.')
        task_name = os.path.join(args.val_model_name, task_name)

    output_path = Path(os.path.join(output_folder, task_name))
    output_opt_viz_path = (output_path / f'{timestamp}/opt_viz')

    # -----------------
    result = pipe(
        bundle=bundle,
        height=Const.DEFAULT_HEIGHT,
        width=Const.DEFAULT_WIDTH,
        generator=generator,
        num_inference_steps=num_inference_steps,
        wandb_log=bundle.get('wandb_log'), 
        output_opt_viz_path=output_opt_viz_path
    )
    video_frames = result.frames
    video_latent = result.latents
    init_bbox_per_frame = torch.tensor(result.bbox_per_frame)
    overlap_maps = result.overlay_attn_maps
    # -----------------
    
    output_path.mkdir(parents=True, exist_ok=True)
    output_video_path = (output_path / f'{timestamp}/video')
    output_video_path.mkdir(parents=True, exist_ok=True)

    # assume its the first, else update the item id
    # output_video_path = output_video_path._str + ".0000.mp4"
    
    video_ext = "mp4" 
    if os.path.exists(output_video_path):
        task_base_name = os.path.splitext(os.path.basename(output_video_path))[0]
        repeated = os.path.join(output_folder, task_name,  timestamp, "video", task_base_name + f"*{video_ext}")
        num_reapts = len(glob.glob(repeated))
        output_video_path = os.path.join(output_folder, task_name, timestamp, "video", task_base_name + f".{num_reapts:04d}.{video_ext}".format(num_reapts))

    # save video frames
    frame_dir = output_path / f'{timestamp}/frames'
    frame_dir.mkdir(parents=True, exist_ok=True)

    if bundle.get('viz_orient2d'):
        frames_with_dir = output_path / f'{timestamp}/frames_with_dir'
        frames_with_dir.mkdir(parents=True, exist_ok=True)

    # video writer definition
    fps = 24
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Choose the codec (codec may vary based on your system and preferences)
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # display_w_size, display_h_size = 512, 512
    display_w_size, display_h_size = 320, 320
    
    if bundle['width']!=display_w_size:
        # assert bundle['width']==bundle['height'], 'you need to handle unequal resolution...'
        if bundle['width']!=bundle['height']:
            # then we cant resize to same W same H
            display_w_size, display_h_size = bundle['width'], bundle['height']

    cv_video_raw = cv2.VideoWriter(output_video_path, fourcc, fps, (bundle['width'], bundle['height']))
    cv_video_disp = cv2.VideoWriter(output_video_path.replace('video.', 'video_disp.'), fourcc, fps, (display_w_size, display_h_size))
    cv_video_with_box = cv2.VideoWriter(output_video_path.replace('video.', 'video_with_bboxes.'), fourcc, fps, (display_w_size, display_h_size))
    cv_video_with_box_with_attn = cv2.VideoWriter(output_video_path.replace('video.', 'video_with_bboxes_with_attn.'), fourcc, fps, (display_w_size, display_h_size))

    # rescale attention maps from low to peak values
    if overlap_maps!=None:
        max_vals, max_indices = overlap_maps.reshape(len(video_frames), -1).max(dim=1)
        overlap_maps = overlap_maps / max_vals[..., None, None]

    for idx, frame in enumerate(video_frames):

        # image/attention map
        im = Image.fromarray(frame) # im W,H <- frame H, W
        _attn_map = overlap_maps[idx] if overlap_maps!=None else None

        # NOTE: convert RGB to BGR for cv2; input frame and box color are BGR, output video is RGB
        frame_only = frame[:,:,[2, 1, 0]].copy() 
        frame_with_init_box = frame[:,:,[2, 1, 0]].copy() 
        frame_with_init_box_with_attn = frame[:,:,[2, 1, 0]].copy() 

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
            frame_with_init_box_with_attn_0_1 = frame_with_init_box_with_attn / 255.0
            
            alpha = 0.7
            frame_with_init_box_with_attn_0_1 = ((1-alpha) * frame_with_init_box_with_attn_0_1) + (alpha * attn_map)
            frame_with_init_box_with_attn_0_1 = np.clip(frame_with_init_box_with_attn_0_1, 0, 1)
            # scale image to 255
            frame_with_init_box_with_attn = (frame_with_init_box_with_attn_0_1 * 255).astype(np.uint8)

        img_path = frame_dir / f"image_{idx:08d}.jpeg"
        if bundle.get('save_frames'):
            im.save(img_path)

        init_bboxA, init_bboxB = init_bbox_per_frame[idx:idx+1], init_bbox_per_frame[idx+1:idx+2]
        init_bboxA_int = convert_bbox_to_image_space(init_bboxA, bundle['width'], bundle['height'])

        if init_bbox_per_frame!=None:
            # grey_color = (128, 128, 128)
            muted_blue_color = (200, 150, 100)  # In BGR format, due to opencv frame read
            muted_red_bgr = (80, 80, 150)
            blue_color = (255, 0, 0)

            if bundle['user_box_color']=='blue':
                init_box_color=blue_color
            elif bundle['user_box_color']=='red':
                init_box_color=muted_red_bgr
            else: 
                raise NotImplementedError

            draw_bbox(frame_with_init_box, init_bboxA_int.squeeze(dim=0).numpy(), thickness=2, color=init_box_color)
            draw_bbox(frame_with_init_box_with_attn, init_bboxA_int.squeeze(dim=0).numpy(), thickness=2, color=init_box_color)
        
        
        if bundle['width']!=display_w_size:
            assert bundle['width']==bundle['height'], 'you need to handle unequal resolution...'
            frame_only_disp = cv2.resize(frame_only, (display_w_size, display_h_size))
            frame_with_init_box = cv2.resize(frame_with_init_box, (display_w_size, display_h_size))
            frame_with_init_box_with_attn = cv2.resize(frame_with_init_box_with_attn, (display_w_size, display_h_size))
        else:
            frame_only_disp = frame_only

        if bundle.get('save_frames'):
            # save_cv2_image(frame_only, str(frames_with_dir / f"image_{idx:08d}.jpeg"))
            save_cv2_image(frame_with_init_box, str(frame_dir / f"image_{idx:08d}_with_box.jpeg"))
            save_cv2_image(frame_with_init_box_with_attn, str(frame_dir / f"image_{idx:08d}_with_box_with_attn.jpeg"))

        if bundle.get('save_video'):
            write_video(cv_video_raw, frame_only)
            write_video(cv_video_disp, frame_only_disp)
            write_video(cv_video_with_box, frame_with_init_box)
            write_video(cv_video_with_box_with_attn, frame_with_init_box_with_attn)
                    
    if bundle.get('save_video'):
        cv_video_raw.release()
        cv_video_disp.release()
        cv_video_with_box.release()
        cv_video_with_box_with_attn.release()

        # mp4 to gif
        v_path_no_boxes = output_video_path.replace('video.', 'video_disp.')
        v_path = output_video_path.replace('video.', 'video_with_bboxes.')
        v_path_with_attn = output_video_path.replace('video.', 'video_with_bboxes_with_attn.')
        
        gif_path = f"{v_path.replace('.mp4', '.gif')}"
        gif_path_no_boxes = f"{v_path_no_boxes.replace('.mp4', '.gif')}"
        gif_path_with_attn = f"{v_path_with_attn.replace('.mp4', '.gif')}"
        
        video_to_gif(v_path, gif_path, start_time=None, end_time=None)
        video_to_gif(v_path_no_boxes, gif_path_no_boxes, start_time=None, end_time=None)
        video_to_gif(v_path_with_attn, gif_path_with_attn, start_time=None, end_time=None)


    data = {
        "latents": result.latents,
        "bundle": bundle,
        "bbox": result.bbox_per_frame,
    }
    latent_path = os.path.splitext(output_video_path)[0] + ".pt"
    torch.save(data, latent_path)
    config_path = os.path.splitext(output_video_path)[0] + ".yaml"
    ConfigIO.config_saver(bundle, config_path)
    log.info(latent_path)
    log.info(output_video_path)
    log.info(config_path)
    log.info("Done")

    seconds = time.time() - per_bundle_start_time
    total_time = time_taken(seconds)
    log.info(f"\n ---------- time_taken PER BUNDLE {total_time} ----------")
    return video_frames

def original_traiblazer(args, cmd_args_bundle):
    
    pre_bundle_start_time = time.time()
    video_frames = None
    
    timestamp = args.timestamp
    # timestamp = bundle['timestamp']

    if args.validate:
        # NOTE: path is linked to /output in optim codebase [high LIKELY]
        output_folder = os.path.join(args.output_path, args.validate_dirname)
        # output_folder = os.path.join(args.output_path, "validate2")
        val_yaml = args.shared_config 
        val_bundle = ConfigIO.config_loader(val_yaml)
        
    else:
        output_folder = os.path.join(args.output_path, "TrailBlazer")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # if args.config_recover:
    #     if not os.path.exists(args.config_recover):
    #         log.info("Path [{}] is invalid.".format(args.config_recover))

    #     data = torch.load(args.config_recover)["bundle"]

    #     filepath = os.path.splitext(os.path.basename(args.config_recover))[0] + ".yaml"
    #     ConfigIO.config_saver(data, filepath=filepath)
    #     log.info("Config recovered in [{}]".format(filepath))

    if args.config:

        experiment_bundles = []
        log.info("Loading config..")
        if os.path.isdir(args.config):
            
            configs = sorted(glob.glob(os.path.join(args.config + "/**/*.yaml"), recursive=True))
            total_configs_len = len(configs)

            # sub-selection
            if args.use_topk_difficult:
                configs = get_topk_yamls()

            if args.drop_excluded_yamls:
                if args.width>320 and args.height>320:
                    print(f'Using high resolution ({args.width}, {args.height}), do you still want to drop invalid yamls?')
                    pdb.set_trace()
                else:
                    configs = drop_invalid_yamls(configs)

            if args.val_start!=None and args.val_stop!=None:
                if args.use_topk_difficult:
                    assert args.val_stop <= len(configs), f'the stop range should be less than or equal to {len(configs)}'
                configs = configs[args.val_start:args.val_stop]

            print(f'\nworking on {len(configs)} out of {total_configs_len} config files ...\n ') # ; time.sleep(0.5)
            if args.use_topk_difficult:
                log.info('using TOPK difficult trajectories')
            assert args.validate, "aren't you validating?"
            
            for cfg in configs:
                log.info(cfg)
                obj_bundle = ConfigIO.config_loader(cfg)
                # ------------------
                if args.use_optim_box:

                    # get corresponding path where otimized box is located
                    pt_dir = os.path.splitext(cfg)[0]
                    pt_dir_drop_config_name  = pt_dir.split('config/')[1]
                    refeed_model = 'optim'
                    optim_box_pt = os.path.join(output_folder, refeed_model, pt_dir_drop_config_name, args.optim_box_timestamp, 'video/video.0000.pt') 
                    assert os.path.isfile(optim_box_pt), f'{optim_box_pt} does not exists'

                    overwrite_bundle_box(obj_bundle, optim_box_pt)
                # ------------------

                if args.validate:
                    bundle = obj_bundle | val_bundle
                else:
                    pdb.set_trace()
                    # TODO: should be uncommented?
                    # bundle = obj_bundle
                
                cmd_override_final_bundle(cmd_args_bundle, bundle)
                experiment_bundles.append([bundle, cfg])
        
        else:
            
            log.info(args.config)
            obj_bundle = ConfigIO.config_loader(args.config)
            
            # ------------------
            if args.use_optim_box:
                optim_box_pt = args.optim_box_pt
                assert os.path.isfile(optim_box_pt), f'{optim_box_pt} does not exists'
                
                overwrite_bundle_box(obj_bundle, optim_box_pt)

            # ------------------
            
            if args.validate:
                # NOTE: the latter overrides the former if duplicate key exists
                bundle = obj_bundle | val_bundle

            cmd_override_final_bundle(cmd_args_bundle, bundle)
            experiment_bundles.append([bundle, args.config])


        model_root = os.environ.get("ZEROSCOPE_MODEL_ROOT")
        assert model_root!="", 'did you export/set the model path on terminal?'

        if not model_root:
            model_root = args.model_root

        model_id = "cerspense/zeroscope_v2_576w"
        model_path = os.path.join(model_root, model_id)
        pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()


        # --------------------------------------------------------------------
        # # create a timestamp now
        # if args.timestamp != '':
        #     timestamp = args.timestamp
        # else:
        #     timestamp = datetime.datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
        
        # **** NOT USED
        if args.search:
            log.info(
                "Searching trailing length by range (-3, 4) of given {}".format(
                    bundle_copy["trailing_length"]
                )
            )
            for i in range(-3, 4):
                bundle = copy.deepcopy(bundle_copy)
                bundle["trailblazer"]["trailing_length"] += i
                # run(bundle)
                run(bundle, config, pipe, args, output_folder)

        else:

            seconds = time.time() - pre_bundle_start_time
            total_time = time_taken(seconds)
            log.info(f"\n ---------- time_taken BEFORE RUN-BUNDLE {total_time} ----------")
            
            
            # ------ MAIN USAGE -------
            for bundle, config in experiment_bundles:
                per_bundle_start_time = time.time()
                
                if not bundle.get("keyframe"):
                    bundle["keyframe"] = Keyframe.get_dyn_keyframe(bundle["prompt"], fix_seed=args.fix_tbl_eval_boxes)
                    # TODO:
                    bundle["trailblazer"]["spatial_strengthen_scale"] = 0.125
                    bundle["trailblazer"]["temp_strengthen_scale"] = 0.125
                    bundle["trailblazer"]["trailing_length"] = 15

                if not bundle.get("trailblazer"):
                    log.warn("No [trailblazer] field found in the config file. Abort.")
                    continue


                if args.create_tbl_eval_boxes: 
                    assert args.val_model_name=="trailblazer_origin", 'TIMED yamls are created once using ORIGIN model call'
                    metric_dir = Path(f'config/Metric_timed/{args.timestamp}')
                    metric_dir.mkdir(parents=True, exist_ok=True)

                    save_config_path = metric_dir / os.path.basename(config)
                    ConfigIO.config_saver(bundle, save_config_path)
                
                else:
                    # video_frames = run(bundle, config)
                    video_frames = run(bundle, config, pipe, args, output_folder, per_bundle_start_time)
                    
            
            # exit program once timed yamls are created
            if args.create_tbl_eval_boxes:
                log.info("Timed/versioned metric yamls has been created")
                exit()

    # end program
    quit()



