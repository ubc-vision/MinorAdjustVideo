#!/usr/bin/env python

import os
import cv2
import pdb
import glob
import json
import time
import copy
import torch
import argparse
import random

import wandb
import imageio
import platform
import datetime
import numpy as np
import pyrootutils
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt


from bin.utils.plot_helpers import (draw_arrow, draw_bbox, save_cv2_image, write_video, delete_images,
                                plot2chk_image)
from bin.utils.misc import (get_bbox_midpoint, get_2d_orientation, convert_pt_to_image_space, 
                        convert_bbox_to_image_space, convert_to_int, args_to_str, time_taken,
                        get_topk_yamls, drop_invalid_yamls)

from bin.CmdTrailBlazer_helpers import (
    run_bundle,
    initialize_wandb,
    cmd_override_final_bundle,
    overwrite_bundle_box,
    get_unique_seed_from_config_path,
    resolve_set_global_deterministic,
)


project_dir = os.getcwd()

machine_name = os.environ.get("TRAILBLAZER_MACHINE", "UNK")

def set_global_determinism(seed: int | None) -> None:
    if seed is None:
        return

    # Map seed to valid NumPy range (0 to 2**32-1)
    # Small seeds (< 2**32) are preserved exactly for backward compatibility
    # Large seeds are clamped via modulo to avoid ValueError
    numpy_seed = seed % (2**32)
    
    # Python / NumPy / PyTorch seeds
    random.seed(seed)
    np.random.seed(numpy_seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Optional but recommended for determinism in matmul
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    # Strict deterministic behavior (may slow things down)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Disable TF32 so A6000 behaves more like V100 numerically
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

# Set the timeout to 1 second (or any value you prefer) to reduce vscode timeout errors
os.environ['PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT'] = '1.0'

# set manual cache directory for stable diffusion
cache_dir = f"{root}/.cache"
os.makedirs(cache_dir,exist_ok=True)

# change cache location to a directory with available space
os.environ['XDG_CACHE_HOME'] = cache_dir # works

from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
from PIL import Image

from TrailBlazer.Misc import ConfigIO
from TrailBlazer.Setting import Keyframe
from TrailBlazer.Misc import Logger as log
from TrailBlazer.Misc import Const



def get_args():
    """args parsing
    Args:
    Returns:
    """
    parser = argparse.ArgumentParser(description="Directed Video Diffusion")
    # parser.add_argument('--foobar', action='store_true')
    parser.add_argument("-c", "--config", help="Input config file", required=True, type=str)
    parser.add_argument("-mr", "--model-root", help="Model root directory", default=".cache", type=str)
    # parser.add_argument("-cr", "--config-recover", help="Input saved data path", type=str)

    parser.add_argument("--run_config", help="run config file", default="", type=str)
    parser.add_argument("--eval_video", help="evaluate generated video", action="store_true")
    parser.add_argument("--zeroscope_xl", help="run base zeroscope model", action="store_true")
    parser.add_argument("--generate_data", help="generate data", action="store_true")
    parser.add_argument("--wandb_log", help="[acts as override] whether to log data to wandb", action="store_true")
    parser.add_argument("--wandb_name", help="name of wandb run", default=None, type=str)
    parser.add_argument("--timestamp", help="timestamp", default="", type=str)
    parser.add_argument("--validate", help="validate", action="store_true")
    parser.add_argument("--val_start", help="validation start idx", default=None, type=int)
    parser.add_argument("--val_stop", help="validation stop idx", default=None, type=int)
    parser.add_argument("--fix_tbl_eval_boxes", help="fix trailblazer eval boxes used for tbl paper evaluation", action="store_true")
    parser.add_argument("--create_tbl_eval_boxes", help="create a version of trailblazer eval boxes (randomly sampled) for tbl paper evaluation", action="store_true")
    parser.add_argument("--use_topk_difficult", help="use diverse top-k `low mIOU per object` trajectories", action="store_true")
    parser.add_argument("--drop_excluded_yamls", help="drop yamls with invalid boxes e.g wrongly-shaped box due to resolution", action="store_true")

    parser.add_argument("--fps", help="[acts as override] output video fps (mp4 writers + gif export)", default=None, type=int)

    parser.add_argument("--width", help="[acts as override] width of video generation", default=None, type=int) 
    parser.add_argument("--height", help="[acts as override] height of video generation", default=None, type=int) 

    parser.add_argument("--n_opt_iterations", help="[acts as override] no of optimization iterations", default=None, type=int)
    parser.add_argument("--bb_deviate_lambda", help="[acts as override] how much of box deviation", default=None, type=float)
    parser.add_argument("--lr", help="[acts as override] learning rate", default=None, type=float)
    parser.add_argument("--outside_bbox_loss_scale", help="[acts as override] scale for outside bbox loss", default=None, type=float)
    parser.add_argument("--inside_bbox_attn_loss_scale", help="[acts as override] scale for inside bbox loss", default=None, type=float)
    parser.add_argument("--box_temp_smooth_scale", help="[acts as override] scale for box temporal smoothness loss", default=None, type=float)
    parser.add_argument("--num_dd_spatial_steps", help="[acts as override] no of spatial steps to edit", default=None, type=int)
    parser.add_argument("--num_dd_temporal_steps", help="[acts as override] no of temporal steps to edit", default=None, type=int)
    parser.add_argument("--num_inference_steps", help="[acts as override] no of inference denoising steps", default=None, type=int)

    parser.add_argument("--spatial_strengthen_scale", help="[acts as override] scale for spatial strengthen_ map", default=None, type=float)
    parser.add_argument("--spatial_weaken_scale", help="[acts as override] scale for spatial weaken map", default=None, type=float)
    parser.add_argument("--temp_strengthen_scale", help="[acts as override] scale for temporal strengthen_ map", default=None, type=float)
    parser.add_argument("--temp_weaken_scale", help="[acts as override] scale for temporal weaken map", default=None, type=float)
    parser.add_argument("--sigma_strength", help="[acts as override] fraction of diagonal length to determine amount of smooth edge transition", default=None, type=float)
    parser.add_argument(
        "--trailing_length",
        help="[acts as override] number of trailing frames for trailblazer",
        default=None,
        type=int,
    )
    
    parser.add_argument("--off_normalize_gauss", help="turn OFF normalizing gaussian [note, normalization maintains consistent peak attention independent of layer resolution]", 
                        action="store_true")
    parser.add_argument("--on_clip_box_values", help="turn ON clipping for box values used to create differentiable box map", action="store_true")
    parser.add_argument("--use_scale_local_foreground", help="turn ON (dynamic) local scaling for box map", action="store_true")
    parser.add_argument("--edit_before_softmax", help="edit attention map before softmax normalization operation", action="store_true")

    parser.add_argument("--use_optim_box", help="use/re-use already optimized box", action="store_true")
    parser.add_argument("--optim_box_pt", help="path to specific pt file containing already optimized box ", default="", type=str)
    parser.add_argument("--optim_box_timestamp", help="timestamp to access pt file(s) containing already optimized box ", default="", type=str)
    parser.add_argument("--store_latent_xo_xt", help="store `jump` latents computed using xo given xt", action="store_true")
    
    # use tri-state behaviour flags here to override
    parser.add_argument(
        "--set_global_deterministic",
        help="[tri-state] Enable strict global determinism. "
        "If passed with no value, defaults to true. "
        "If omitted, falls back to shared_config key 'set_global_deterministic' (bool) or False. "
        "Requires integer 'seed' in the main config YAML. May slow down.",
        nargs="?",
        const="true",
        choices=["true", "false"],
        default=None,
        type=str,
    )
    parser.add_argument(
        "--vis_layer",
        help="Comma-separated layer ids to visualize, e.g. 'up.2.2.0' or 'up.2.2.0,down.0.0.0'",
        default="up.2.2.0",
        type=str,
    )
    parser.add_argument(
        "--vis_maps",
        help="[acts as override] turn map visualizations on/off (config default unless set)",
        choices=["true", "false"],
        default=None,
        type=str,
    )
    parser.add_argument(
        "--overlay_maps",
        help="[acts as override] turn overlay map generation on/off (config default unless set)",
        choices=["true", "false"],
        default=None,
        type=str,
    )
    parser.add_argument("--off_normalize_mask", help="turn OFF normalizing boundary masks [xxx]", action="store_true")
    parser.add_argument("--temp_edit_at_low_res", help="do temporal edit for low-resolution generation", action="store_true")
    parser.add_argument("--box_temp_smooth_loss", help="penalize optimized boxes to be temporal smooth", action="store_true")
    parser.add_argument("--init_bbox_area_loss", help="penalize optimized boxes that deviate from user control in terms of area", action="store_true")

    parser.add_argument("--validate_dirname", help="validate directory name", default="validate", type=str)
    parser.add_argument("--shared_config", help="shared config file for comparisons", default="config/common_shared.yaml", type=str)
    parser.add_argument("--val_model_name", help="model to run validation on \
                        [trailblazer_origin, trailblazer_diff, optim, peakaboo, text2vidzero]", default="optim", type=str)
    
    parser.add_argument(
        "-s",
        "--search",
        help="Search parameter based on the number of trailing attention",
        action="store_true")
    parser.add_argument(
        "--output-path",
        type=str,
        default="output",
        help="Path to save the generated videos",
    )
    parser.add_argument("-xl", "--zeroscope-xl", help="Search parameter", action="store_true")
    return parser.parse_args()


def main():
    """
    The entry point to execute this program
    Args:
    Returns:
    """
    pre_bundle_start_time = time.time()
    args = get_args()
    cmd_args_bundle = vars(args)

    # -------------------------------------------------------------------------
    # Tri-state global determinism:
    # - CLI: --set_global_deterministic {true,false} overrides everything
    # - CLI: --set_global_deterministic (no value) => true
    # - Else: shared_config key set_global_deterministic (bool) if present
    # - Else: False
    det = resolve_set_global_deterministic(args.set_global_deterministic, args.shared_config)

    if det:
        # Strict determinism once at program start.
        # Supports config path being either a YAML file or a directory of YAMLs.
        seed = get_unique_seed_from_config_path(args.config)
        set_global_determinism(seed)

    # create a timestamp now
    if args.timestamp != '':
        timestamp = args.timestamp
    else:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
        args.timestamp = timestamp

    # -------------------------------------------------------------------------------------
    if cmd_args_bundle['val_model_name']=="trailblazer_origin" or args.run_config=='':

        from bin.Origin_CmdTrailBlazer import original_traiblazer
        original_traiblazer(args, cmd_args_bundle)

    elif cmd_args_bundle['val_model_name']=="trailblazer_diff" or cmd_args_bundle['val_model_name']=="optim":
        # custom
        # from TrailBlazer.Pipeline.Attention_class import Attention_custom

        # original classes
        from diffusers.pipelines import TextToVideoSDPipeline
        from diffusers.models.attention_processor import Attention
        from diffusers.models.unet_3d_condition import UNet3DConditionModel
        from diffusers.models.transformer_temporal import TransformerTemporalModel
        from diffusers.models.attention import BasicTransformerBlock
        from diffusers.models.unet_3d_blocks import (
            CrossAttnDownBlock3D,
            CrossAttnUpBlock3D,
            DownBlock3D,
            UNetMidBlock3DCrossAttn,
            UpBlock3D
        )
        from diffusers.models.transformer_2d import Transformer2DModel
        # custom calls
        from TrailBlazer.Pipeline.TextToVideoSDPipelineCall import text_to_video_sd_pipeline_call
        from TrailBlazer.Pipeline.UNet3DConditionModelCall import unet3d_condition_model_forward
        from TrailBlazer.Pipeline.blocks.TransformerTemporalModelCall import transformer_temporal_model_forward
        from TrailBlazer.Pipeline.blocks.BasicTransformerBlockCall import basic_transformer_block_forward 
        from TrailBlazer.Pipeline.blocks.Transformer2DModelCall import transformer2d_forward
        from TrailBlazer.Pipeline.blocks.Unet3DBlocksCall import (
            cross_attn_downblock3D_forward, downblock3D_forward, unet_mid_block3D_cross_attn_forward,
            cross_attn_upblock3D_forward, upblock3D_forward)
        # from TrailBlazer.Pipeline.Attention_class import Attention_custom 
        
        # note: extract and override original forward function/call with custom
        # 1.
        TextToVideoSDPipeline.__call__ = text_to_video_sd_pipeline_call
        # 2.
        unet3d_condition_model_forward_copy = UNet3DConditionModel.forward
        UNet3DConditionModel.forward = unet3d_condition_model_forward
        # 3.
        TransformerTemporalModel.forward = transformer_temporal_model_forward
        BasicTransformerBlock.forward = basic_transformer_block_forward
        # 4.
        Transformer2DModel.forward = transformer2d_forward
        CrossAttnDownBlock3D.forward = cross_attn_downblock3D_forward
        DownBlock3D.forward = downblock3D_forward
        # 5. 
        UNetMidBlock3DCrossAttn.forward = unet_mid_block3D_cross_attn_forward
        # 6. 
        CrossAttnUpBlock3D.forward = cross_attn_upblock3D_forward
        UpBlock3D.forward = upblock3D_forward
        # 7.
        # Override Attention class globally in diffusers models
        # import diffusers
        # diffusers.models.attention_processor.Attention = Attention_custom

    else:
        raise NotImplementedError
    
    # -------------------------------------------------------------------------------------
    
    video_frames = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_yaml = args.run_config # "config/run.yaml"
    base_bundle = ConfigIO.config_loader(base_yaml)

    if args.generate_data:
        output_folder = os.path.join(args.output_path, "TrailBlazer")
    if args.validate:
        output_folder = os.path.join(args.output_path, args.validate_dirname)
        val_yaml = args.shared_config 
        val_bundle = ConfigIO.config_loader(val_yaml)
    else:
        output_folder = os.path.join(args.output_path, "models")
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    

    if args.config:
        # note: still uses `experiment_bundles` even if its a single bundle
        experiment_bundles = []

        log.info("Loading config..")
        if os.path.isdir(args.config):
            configs = sorted(glob.glob(os.path.join(args.config + f"/**/*.yaml"), recursive=True))
            init_total_configs_len = len(configs)

            # sub-selection
            if args.use_topk_difficult and 'validate' in args.validate_dirname:
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

            print(f'\nworking on {len(configs)} out of {init_total_configs_len} config files ...\n ') # ; time.sleep(0.5)
            if args.use_topk_difficult:
                log.info('using TOPK difficult trajectories')
            assert args.validate or args.generate_data, "expected --validate or --generate_data"
            
            for c_args in configs:
                log.info(c_args)
                obj_bundle = ConfigIO.config_loader(c_args)
                # ------------------
                if args.use_optim_box:

                    # get corresponding path where otimized box is located
                    pt_dir = os.path.splitext(c_args)[0]
                    pt_dir_drop_config_name  = pt_dir.split('config/')[1]
                    refeed_model = 'optim'
                    optim_box_pt = os.path.join(output_folder, refeed_model, pt_dir_drop_config_name, args.optim_box_timestamp, 'video/video.0000.pt') 
                    assert os.path.isfile(optim_box_pt), f'{optim_box_pt} does not exists'

                    overwrite_bundle_box(obj_bundle, optim_box_pt)
                # ------------------
                
                # `|` merges two dictionaries
                # NOTE: the latter overrides the former if duplicate key exists
                bundle = obj_bundle | base_bundle  
                if args.validate:
                    bundle = bundle | val_bundle
                
                cmd_override_final_bundle(cmd_args_bundle, bundle)
                experiment_bundles.append([bundle, c_args])
                del bundle
        
        else:

            log.info(args.config)
            obj_bundle = ConfigIO.config_loader(args.config)
            # ------------------
            if args.use_optim_box:
                optim_box_pt = args.optim_box_pt
                assert os.path.isfile(optim_box_pt), f'{optim_box_pt} does not exists'

                overwrite_bundle_box(obj_bundle, optim_box_pt)
            # ------------------

            bundle = obj_bundle | base_bundle 
            if args.validate:
                bundle = bundle | val_bundle
            cmd_override_final_bundle(cmd_args_bundle, bundle)
            experiment_bundles.append([bundle, args.config])
            del bundle
        
        model_root = os.environ.get("ZEROSCOPE_MODEL_ROOT")
        assert model_root!="", 'did you export/set the model path on terminal?'

        if not model_root:
            model_root = args.model_root

        
        model_id = "cerspense/zeroscope_v2_576w"
        model_path = os.path.join(model_root, model_id)

        # note: weights in fp16 is less numerically precise but saves memory
        # fp16 - precise only up to 3-4 decimal points
        pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        
        # Move pipeline components to the correct device 
        # note: unet takes ~4GB GPU memory
        pipe.unet.to(device)
        pipe.text_encoder.to(device)
        pipe.vae.to(device)

        # --------------------------------------------------------------------


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
                run_bundle(bundle)

        else:   

            seconds = time.time() - pre_bundle_start_time
            total_time = time_taken(seconds)
            log.info(f"\n ---------- time_taken BEFORE RUN-BUNDLE {total_time} ----------")
            
            
            # ------ MAIN USAGE -------
            for bundle, config in experiment_bundles:
                per_bundle_start_time = time.time()

                # note: if keyframe not available
                if not bundle.get("keyframe"):

                    bundle["keyframe"] = Keyframe.get_dyn_keyframe(bundle["prompt"], fix_seed=args.fix_tbl_eval_boxes)
                    # TODO:
                    bundle["trailblazer"]["spatial_strengthen_scale"] = 0.125
                    bundle["trailblazer"]["temp_strengthen_scale"] = 0.125
                    if "trailing_length" not in bundle["trailblazer"]:
                        bundle["trailblazer"]["trailing_length"] = 15

                if not bundle.get("trailblazer"):
                    log.warn("No [trailblazer] field found in the config file. Abort.")
                    continue
                    
                if args.create_tbl_eval_boxes: 
                    assert args.val_model_name=="trailblazer_origin", 'TIMED yamls are created once using ORIGIN model call'

                bundle['machine_name'] = machine_name
                initialize_wandb(bundle)
                video_frames = run_bundle(bundle, config, pipe, args, output_folder, per_bundle_start_time)

                # after each YAML
                del video_frames
                import gc
                gc.collect()
                torch.cuda.empty_cache()



if __name__ == "__main__":
    main()


