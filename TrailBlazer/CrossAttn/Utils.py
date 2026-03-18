import enum
import math
import torch
import torchvision
import numpy as np

from ..Setting import Config
from ..Misc import Logger as log
from ..Misc.BBox import BoundingBox, compute_bbox_LRTB_HW

import matplotlib
from pathlib import Path
import matplotlib.pyplot as plt

# To avoid plt.imshow crash
matplotlib.use("Agg")

import os
import pdb
import math
import wandb
import imageio
from glob import glob
import matplotlib.pyplot as plt
from torchvision.io import write_video
from PIL import Image, ImageDraw, ImageFont
from bin.utils.misc import resize_image_to_macro_block
from bin.utils.misc import delete_videos, get_bbox_midpoint
from bin.utils.plot_helpers import convert_to_numpy, plot2chk_image, fig2img
from moviepy.editor import VideoFileClip, CompositeVideoClip, ImageClip, concatenate_videoclips


INJECTION_SCALE = 1.0
KERNEL_DIVISION = 3.

def process_temporal1(modified_copy_temporal_attn, dim_x, dim_y, num_frames, attn_heads):
    # torch.Size([8, 64, 64, 24, 24])
    modified_copy_temporal_attn = torch.transpose(modified_copy_temporal_attn, 1, 2)
    modified_copy_temporal_attn = reshape_fortran(modified_copy_temporal_attn,
        (attn_heads * dim_y * dim_x, num_frames, num_frames))
    return modified_copy_temporal_attn

def process_temporal2(modified_copy_temp_attn, attention_probs, n, device):
    # differentiable operation
    mask = torch.ones_like(attention_probs)
    mask[:n] = 0
    bottom_part_0s = torch.zeros_like(attention_probs[n:])
    assert modified_copy_temp_attn.shape[0] == n, f'modified copy needs correct shape {n}'
    new_attention_probs = (attention_probs * mask) + torch.cat([modified_copy_temp_attn, bottom_part_0s]).to(device)
    return new_attention_probs

def get_edge_coords(left,right,top,bottom, map_h, map_w, allow_edge_margin=False, edge_margin=2):
    if allow_edge_margin:
        top_a, bottom_a = max(math.floor(top-edge_margin), 0), min(math.ceil(bottom+edge_margin), map_h) 
        left_a, right_a = max(math.floor(left-edge_margin), 0) , min(math.ceil(right+edge_margin), map_w)
    else:
        # allowed small round-off margin
        top_a, bottom_a, left_a, right_a = math.floor(top), math.ceil(bottom), math.floor(left),  math.ceil(right)
        # top_a, bottom_a, left_a, right_a = int(top), min(int(bottom)+1, map_h), int(left),  min(int(right)+1, map_w)
    return left_a, right_a, top_a, bottom_a

def reshape_fortran(x, shape):
    """ Reshape a tensor in the fortran index. See
    https://stackoverflow.com/a/63964246
    """
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))

def compute_sigma(height, width, fraction):
    # Compute the diagonal length of the bbox/image
    diagonal = torch.sqrt(height**2 + width**2)
    # Compute sigma as a fraction e.g 3%, of the diagonal length
    sigma = fraction * diagonal
    return sigma

def create_diff_bbox_heatmap_gaussian_2(height, width, top, bottom, left, right, 
                                        sigma_x=10.0, sigma_y=10.0, mask_sigma=1.0, act=torch.sigmoid,
                                        use_high_box_only=False, eps=1e-32, scale_local_foreground=False,
                                        local_scale=1,
                                        inverse_gauss=False, normalize_gauss=True, normalize_mask=True, index=None):
    """
    Create a differentiable (version of prev authors) heatmap with a Gaussian centered within the bounding box.

    Parameters:
    bbox (tuple): Bounding box defined as (x_min, y_min, x_max, y_max).
    heatmap_size (tuple): Size of the heatmap (height, width).
    sigma (float): Standard deviation of the Gaussian.
    mask_sigma (float): Standard deviation for the soft mask transition. Higher value allows the gaussian to spread beyond the box


    Returns:
    torch.Tensor: Differentiable heatmap.
    """
    # x_min, top, x_max, bottom = bbox
    # height, width = heatmap_size

    # Center of the bounding box
    x_center = (left + right) / 2
    y_center = (top + bottom) / 2

    # Create a 2D grid (`switched` repeat is done for compatible) 
    # NOTE: all u`s at dim HxW
    u = torch.arange(0, width).view(1, -1).repeat(height, 1).float().to(left.device)
    # NOTE: all v`s at dim HxW
    v = torch.arange(0, height).view(-1, 1).repeat(1, width).float().to(left.device)

    
    # Calculate the Gaussian within the bounding box
    # sigma controls the maximum value and spread of the gaussian
    gauss = torch.exp(-( ((u - x_center)**2) / (2 * sigma_x**2)  + ((v - y_center)**2 ) / (2 * sigma_y**2)))

    # challenge: small res e.g spatial mid-block creates a small peak value, that explodes the loss value due to small-value division (normalization) by the atten mask sum
    # normalize gaussian to have consistent peak value of 1 (regardless of resolution)
    '''update latex: maintain numerical stability, ensures consistency, and attention mechanism
    - remain effective across different resolutions'''


    # old_gauss = gauss.clone()
    # note: max normalization does not break gaussian kernel point-wise equivalence 
    if normalize_gauss:
        gauss_max = gauss.max()
        gauss = gauss / (gauss_max + eps)
    
    # if gauss.isnan().any():

    # plot2chk_image(test_bbox_map, filename="test_bbox_map2")
    #  mask_x = (act((x - left) / mask_sigma) * act((right - x) / mask_sigma))
    # mask_y = (act((y - top) / mask_sigma) * act((bottom - y) / mask_sigma))

    # Create soft masks for the bounding box edges
    mask_x = (act((u - left) / (mask_sigma) + eps) * 
              act((right - u) / (mask_sigma) + eps))
    mask_y = (act((v - top) / (mask_sigma) + eps) * # <------ small resolution issue
              act((bottom - v) / (mask_sigma) + eps))
    
    # if height == width == 5:
    
    if normalize_mask:
        mask_x = mask_x / (mask_x.max() + eps)
        mask_y = mask_y / (mask_y.max() + eps)
    
    mask = mask_x * mask_y
    # if mask.isnan().any():

    if use_high_box_only:
        gauss = mask

    # Apply the soft mask to the Gaussian to smoothly drop to zeros at the edges
    if inverse_gauss:
        pdb.set_trace()
        'TO BE DROPPED - as scaling it down is more principled, than engineering'
        # inv_gauss = gauss.max() - gauss
        # img = inv_gauss * mask
        # plot2chk_image(mask, filename="mask")
    else:
        # if index==0:
        #     orig_gauss = gauss.clone()
        # scale_local_foreground (following trailblazer)
        if scale_local_foreground:
            gauss = gauss * local_scale

        img = gauss * mask
        # if index==0:
        #     print(f'orig_gauss max {orig_gauss.max()}| scale_local_foreground {scale_local_foreground}| local_scale {local_scale} | scaled orig_gauss max {gauss.max()} | masked gauss max {img.max()}')

    # plot2chk_image(img, filename=f"img_norm_mask_and_gauss_mask_sigma_{mask_sigma}")
    # plot2chk_image(img_inv, filename="img_inv")
    # plot2chk_image(gauss, filename="gauss")

    return img

def create_diff_bbox_heatmap(height, width, top, bottom, left, right, sigma=1.0, act=torch.sigmoid):
    """
    Create a differentiable heatmap for a bounding box.
    
    Parameters:
    height (int): Height of the output heatmap.
    width (int): Width of the output heatmap.
    top (float): Top coordinate of the bounding box.
    bottom (float): Bottom coordinate of the bounding box.
    left (float): Left coordinate of the bounding box.
    right (float): Right coordinate of the bounding box.
    sigma (float): Standard deviation for Gaussian blur (default: 1.0).
    
    Returns:
    torch.Tensor: Heatmap of shape (1, height, width).
    """

    'it introduces sigmoid in between layers + the attention is not gaussian, can lead to unstable training'
    # Create a coordinate grid
    y_coords = torch.arange(0, height, dtype=torch.float32).view(-1, 1).expand(height, width).to(top.device)
    x_coords = torch.arange(0, width, dtype=torch.float32).view(1, -1).expand(height, width).to(top.device)

    # Create masks for the bounding box
    mask_top = act((y_coords - top) / sigma)
    mask_bottom = act((bottom - y_coords) / sigma)
    mask_left = act((x_coords - left) / sigma)
    mask_right = act((right - x_coords) / sigma)

    # Combine the masks to get the final bounding box mask
    heatmap = mask_top * mask_bottom * mask_left * mask_right

    return heatmap

def gaussian_2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):
    """ 2d Gaussian weight function
    """
    gaussian_map = (
        1
        / (2 * math.pi * sx * sy)
        * torch.exp(-((x - mx) ** 2 / (2 * sx**2) + (y - my) ** 2 / (2 * sy**2)))
    )
    # note: max normalization does not break gaussian kernel point-wise equivalence 
    gaussian_map = gaussian_map / gaussian_map.max()
    # gaussian_map.div_(gaussian_map.max())
    return gaussian_map

# def gaussian_2d_modified(height, width, x=0, y=0, mx=0, my=0, sx=1, sy=1):
#     """ 2d Gaussian weight function
#     """
#     gaussian_map = (
#         1
#         / (2 * math.pi * sx * sy)
#         * torch.exp(-((x - mx) ** 2 / (2 * sx**2) + (y - my) ** 2 / (2 * sy**2)))
#     )
#     gaussian_map = gaussian_map / gaussian_map.max()
#     # gaussian_map.div_(gaussian_map.max())
#     return gaussian_map

def get_patch(bbox_at_frame, i, j, bbox_per_frame, attention_probs_5d, dim_x, dim_y, sigma_strength, 
              minimize_bkgd=False, no_opt=True, use_high_box_only=False,  normalize_gauss=False,
              normalize_mask=False, clip_box_values = False, 
              scale_local_foreground = False, local_scale = 1,
              gauss_only=True, allow_edge_margin=False, edge_margin=2
              ):
    
    if no_opt==False:
        assert bbox_per_frame.requires_grad==True, 'the boxes are not optimized.'
    
    bbox_ratios = bbox_at_frame
    n_c, map_h, map_w, n_frames_i, n_frames_j = attention_probs_5d.shape
    
    'REMOVE int'
    try:
        # clip_box_values = True
        left, right, top, bottom, bbox_h, bbox_w = compute_bbox_LRTB_HW(dim_x, dim_y, bbox_ratios, use_int=clip_box_values, margin=0.01)
    except:
        pdb.set_trace()
    # print(f"left {left} right {right} top {top} bottom {bottom} bbox_h {bbox_h} bbox_w {bbox_w}")
    bbox_h, bbox_w  = bbox_h.detach(), bbox_w.detach()
    
    'da note: .half can be removed once bbox is differentiable?'
    if gauss_only:
        sigma_x = bbox_w / KERNEL_DIVISION
        sigma_y = bbox_h / KERNEL_DIVISION 
        mask_sigma = compute_sigma(bbox_h, bbox_w, fraction=sigma_strength)
        # mask_sigma=0.01
        
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
                ).unsqueeze(0).repeat(n_c, 1, 1).to(attention_probs_5d.device).half()
        
        # test_bbox_map = create_diff_bbox_heatmap_gaussian_2(height=map_h, width=map_w, top=top, bottom=bottom, left=left, right=right, sigma=sigma, mask_sigma=mask_sigma, act=torch.sigmoid)
        
        # 'we apply soft masks to inverse bbox map'
        # inv_bbox_map = create_diff_bbox_heatmap_gaussian_2(
        #     height=map_h,
        #         width=map_w,
        #         top=top, 
        #         bottom=bottom, 
        #         left=left, 
        #         right=right, 
        #         sigma=sigma, # if higher diff2 ~= diff1
        #         mask_sigma=mask_sigma,
        #         act=torch.sigmoid,
        #         inverse_gauss=True,
        # normalize_gauss=normalize_gauss,
        # normalize_mask=normalize_mask
        #         ).unsqueeze(0).repeat(n_c, 1, 1).to(attention_probs_5d.device).half()
        
    else:
        pdb.set_trace()
        sigma = compute_sigma(bbox_h, bbox_w, fraction=sigma_strength)
        bbox_map = create_diff_bbox_heatmap(
                height=map_h,
                width=map_w,
                top=top, 
                bottom=bottom, 
                left=left, 
                right=right, 
                sigma=sigma, 
                act=torch.sigmoid
                ).unsqueeze(0).repeat(n_c, 1, 1).to(attention_probs_5d.device).half()
    
    'removing local-scale boosting, as it affects outside bbox values, hence affects weaken map creation'
    # scale = attention_probs_5d.max() * INJECTION_SCALE
    # bbox_map = bbox_map * scale.detach()

    # if i==0 and j==0:
    #     print(f'final bbox_map max {bbox_map.max()} | attn max {attention_probs_5d.max()}\n')

    # box_map_dist = create_diff_bbox_heatmap(height=map_h, width=map_w,top=top, bottom=bottom,left=left,right=right,sigma=sigma_,act=torch.sigmoid)

    # NEW TESTING
    # max_values, _ = torch.max(bbox_map.view(n_c, -1), dim=1)
    # diff_inv_bbox = max_values[..., None, None] - bbox_map
    # diff_inv_bbox = bbox_map.max() - bbox_map 

    # OLD
    # inv_bbox_map = bbox_map - bbox_map.max() # .detach()
    
    # OLD: zero-out introduced negatives values outside box area in `inv_bbox_map`
    # bkgd_zero = torch.zeros_like(bbox_map).to(attention_probs_5d.device).half()
    # non_zero_indices = torch.where(bbox_map.detach() != 0) 
    # bkgd_zero[non_zero_indices] = 1

    # if minimize_bkgd:
    #     bkgd_one = torch.ones_like(bbox_map).to(attention_probs_5d.device).half()
        
    # left_a, right_a, top_a, bottom_a = get_edge_coords(left,right,top,bottom, map_h, map_w, allow_edge_margin=allow_edge_margin, edge_margin=edge_margin)
    # bkgd_zero[:, top_a:bottom_a,left_a:right_a] = 1
    
    # if minimize_bkgd:
    #     bkgd_one[:, top_a:bottom_a,left_a:right_a] = 0
    #     bkgd_map = bbox_map * bkgd_one
    #     inv_bkgd_map = bkgd_map - bkgd_map.max()

    'zero-out introduced values outside box area, focus is on the bbox area'
    # core_inv_bbox_map = inv_bbox_map * bkgd_zero

    # if minimize_bkgd:
    #     # zero-out introduced negatives values inside box area in `inv_bkgd_map`
    #     core_inv_bkgd_map = inv_bkgd_map * bkgd_one

    'NOTE: `bbox_map` not masked to keep smooth transition on the boundaries and get well-behaved derivaties'
    # bbox_map = bbox_map * bkgd_zero

    # note: normalized distance using bbox mid point location
    # NOTE: optimized bbox_per_frame if fp32, convert to fp16 when used
    mid_j = get_bbox_midpoint(bbox_per_frame[[j]]).half()
    mid_i = get_bbox_midpoint(bbox_per_frame[[i]]).half()
    dist = torch.norm(mid_j - mid_i, dim=1)

    # OLD
    # dist = (float(abs(j - i))) / (len(bbox_per_frame)-1)
 
    'NEW - simple scale down, rather than using discontinous-engineered inverse map'
    weight = (1. - dist)
    final_patch_bbox = bbox_map * weight

    # final_patch_bbox = inv_bbox_map * dist + bbox_map * (1. - dist)
    # if minimize_bkgd:
    #     final_patch_bkgd = core_inv_bkgd_map * dist + bkgd_map * (1. - dist)

    # TODO: challenge - bkgd map is 0s (has no value) since bbox map is a gaussian. Is this good? oe even okay and why? 
    # because we would still optimize?

    # if i==0 and j==4:
    #     plot2chk_image(bbox_map[0, :,:], filename=f"bbox_map_temporal_i{i}_j{j}")
    #     plot2chk_image(final_patch_bbox[0, :,:], filename=f"final_patch_bbox_temporal_i{i}_j{j}")

    return final_patch_bbox, dist, weight

def get_patch_old(bbox_at_frame, i, j, bbox_per_frame, attention_probs_5d, dim_x, dim_y, INJECTION_SCALE):
    bbox = BoundingBox(dim_x, dim_y, bbox_at_frame)
    # print(f"left {bbox.left} right {bbox.right} top {bbox.top} bottom {bbox.bottom} bbox_h {bbox.height} bbox_w {bbox.width}")

    # Generating the gaussian distribution map patch
    x = torch.linspace(0, bbox.height, bbox.height)
    y = torch.linspace(0, bbox.width, bbox.width)
    x, y = torch.meshgrid(x, y, indexing="ij")

    'da note: .half can be removed once bbox is differentiable?'
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

    # scale = attention_probs_5d.max() * INJECTION_SCALE
    # noise_patch = noise_patch * scale

    inv_noise_patch = noise_patch - noise_patch.max()
    # note: normalized distance
    dist = (float(abs(j - i))) / len(bbox_per_frame)
    final_patch = inv_noise_patch * dist + noise_patch * (1. - dist)
    #final_patch = noise_patch * (1. - dist)
    #final_patch = inv_noise_patch * dist
    return final_patch, bbox

class CAttnProcChoice(enum.Enum):
    INVALID = -1
    BASIC = 0

def time_taken(seconds, tag=""):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    print(f' {tag}: {int(d)} days :{int(h)} hrs :{int(m):02d} mins :{int(s):02d} secs\n') 

def create_last_map(t, attn_type):
    text = f"Timestep_{t:04d}_{attn_type}"
    ...

def create_motion_timestep_video(attn_video_path, caption, t, opt_idx, n_edit_steps, attn_type, inject=True):
    # stack videos in motion-timestep

    text = f"opt_idx_{opt_idx:04d}_timestep_{t:04d}_{attn_type}"

    video_urls = sorted(glob(os.path.join(attn_video_path,  f"{caption}_*.mp4")))
    try:
        final_clip = concatenate_videoclips([movie_prepare_video(url, text) for url in video_urls])
    except:
        pdb.set_trace()
        # steps_folder = attn_video_path.replace("layers", "steps")
        # created_videos = sorted(glob(os.path.join(steps_folder,  f"*.mp4")))
        # assert len(created_videos) < n_edit_steps, f'we expect video from {n_edit_steps} editing steps'

    step_local_folder = "steps/with_inject_at_edit" if inject else "steps/no_inject_at_edit"
    new_attn_video_path = Path(str(attn_video_path).replace("layers", step_local_folder))
    new_attn_video_path.mkdir(exist_ok=True)
    try:
        final_clip.write_videofile(os.path.join(str(new_attn_video_path),  f"{text}.mp4"))
    except:
        pdb.set_trace()

    # delete single videos
    delete_videos(video_urls)


def add_timestep_to_video_array(attn_video_path, caption, t, opt_idx, no_opt=True, time_bf_motion=False):
    # add timestep to video array files

    if no_opt==False and time_bf_motion==True:
        # find only npy that has not been timestep renamed
        array_urls = sorted(glob(os.path.join(str(attn_video_path).replace("layers", "arrays"),  f"{caption}_*.attn2.npy")), reverse=True)
        if len(array_urls)!=0:
            print(f'------------------------ caption {caption} t {t} opt_idx {opt_idx}')
        # rename npy files to contain actual timestep when they were created
        for url in array_urls:
            assert os.path.isfile(url), f'file: {url} does not exist.'
            os.rename(url, url.replace('.npy', f'_opt_idx_{opt_idx:04d}_timestep_{t:04d}.npy'))

def draw_text_on_np_array(video_array, text, fontsize=22):
    video_array_with_text = video_array.copy()

    # Load a font
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf" 
    font = ImageFont.truetype(font_path, fontsize)

    # Iterate over each frame in the video array
    for i in range(video_array.shape[0]):
        # Convert the NumPy array frame to a PIL image
        pil_image = Image.fromarray(video_array[i])
        # Initialize ImageDraw
        draw = ImageDraw.Draw(pil_image)
        # Define text position (e.g., center of the image)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        position = ((video_array.shape[2] - text_width) // 2, 100)
        # Add text in black to image - keep alpha (last) channel to 255 to make color non-transparent but fully opaque
        draw.text(position, text, (0, 0, 0, 255), font=font)
        # Convert the PIL image back to a NumPy array
        video_array_with_text[i] = np.array(pil_image)
    
    return video_array_with_text

def create_timestep_motion_video(attn_video_path, caption, t, opt_idx, n_opt_iterations, num_dd_spatial_steps, 
                                 attn_type, no_opt=True, time_bf_motion=False, focus_1timestep_1channel = False, 
                                 focus_1channel_only=False, fps=24):
    # text = f"opt_idx_{opt_idx:04d}_{attn_type}"

    video_opt_iter_timestep_motion = []
    for opt_idx in range(n_opt_iterations):
        # switching order to denoising timestep first before motion, for easy debugging
        opt_idx_array_urls = sorted(glob(os.path.join(str(attn_video_path).replace("layers", "arrays"),  f"{caption}*_opt_idx_{opt_idx:04d}_*.npy")), reverse=True)
        # array_urls = sorted(glob(os.path.join(str(attn_video_path).replace("layers", "arrays"),  f"{caption}*_opt_idx_{opt_idx:04d}_*.npy")), reverse=True)
        # assert len(array_urls) == (n_opt_iterations * num_dd_spatial_steps)

        if len(opt_idx_array_urls)==0:
            print(f'No opt idx {opt_idx} data found for {attn_type}...')
            return 

        video_timestep_motion = []
        for iv, video_array_url in enumerate(opt_idx_array_urls):
            video_array = np.load(video_array_url)
            n_channels = video_array.shape[0]

            if focus_1timestep_1channel:
                if iv==0:
                    video_array = video_array[n_channels-1:n_channels, ...]
                    # video_array = video_array[0:1, ...]
                else:
                    continue
            elif focus_1channel_only:
                video_array = video_array[0:1, ...]


            denoise_timestep = os.path.splitext(os.path.basename(video_array_url))[0].split('timestep_')[1]
            text = f'opt iter: {opt_idx} denoise_timestep: {denoise_timestep}'
            # encode opt iter and denoising timestep data here
            video_array_with_text = draw_text_on_np_array(video_array, text=text) 
            video_timestep_motion.append(video_array_with_text)
        
        # KEY: making `denoising timestep, motion` order
        timestep_motion = np.stack(video_timestep_motion, axis=1)
        video_opt_iter_timestep_motion.append(timestep_motion)
    
    # KEY: making `opt iter, denoising timestep, motion` order
    opt_iter_timestep_motion =  np.stack(video_opt_iter_timestep_motion, axis=2)
    video = opt_iter_timestep_motion.reshape(-1, 1008, 1008, 4)

    # TODO:
    # create video over timesteps before motion (extend over opt iter later)
    # try diff optimization objective to meet achieve task 1 | try log as well | confirm changes in opt visualizations

    filename = os.path.join(str(attn_video_path).replace("layers", "time_bf_motion"),  f'{caption}.mp4')
    # filename = os.path.join(str(attn_video_path).replace("layers", "time_bf_motion"),  f'{caption}_opt_idx_{opt_idx:04d}.mp4')
    # TODO: write text information over images
    imageio.mimwrite(filename, video, fps=fps)

    # delete single npy files
    _array_urls = sorted(glob(os.path.join(str(attn_video_path).replace("layers", "arrays"),  f"{caption}*.npy")), reverse=True)
    delete_videos(_array_urls, ext=".npy")

def movie_prepare_video(url, text, text_color="black", fontsize=22):
    """prepare and add text to text to existing videos"""
    video = VideoFileClip(url)

    # Create a mini-image for the text
    image = Image.new('RGB', (video.w, 100), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf" 
    font = ImageFont.truetype(font_path, fontsize)
    bbox = draw.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((video.w - w) / 2, (100 - h) / 2), text, font=font, fill="black")
    
    # Convert the Pillow image to a NumPy array
    image_np = np.array(image)
    # Create an ImageClip from the NumPy array
    text_image = ImageClip(image_np).set_duration(video.duration).set_position(('center', 'top'))

    # Overlay the text clip on the video clip
    video_with_text = CompositeVideoClip([video, text_image])
    return video_with_text

# custom attn layer ids (16 spatial multi-resolution layers)
spatial_module_names = {'down_blocks.0.attentions.0.transformer_blocks.0.attn2': 0,
                        'down_blocks.0.attentions.1.transformer_blocks.0.attn2': 1,
                        'down_blocks.1.attentions.0.transformer_blocks.0.attn2': 2,
                        'down_blocks.1.attentions.1.transformer_blocks.0.attn2': 3,
                        'down_blocks.2.attentions.0.transformer_blocks.0.attn2': 4,
                        'down_blocks.2.attentions.1.transformer_blocks.0.attn2': 5,
                        'mid_block.attentions.0.transformer_blocks.0.attn2': 6,
                        'up_blocks.1.attentions.0.transformer_blocks.0.attn2': 7,
                        'up_blocks.1.attentions.1.transformer_blocks.0.attn2': 8,
                        'up_blocks.1.attentions.2.transformer_blocks.0.attn2': 9,
                        'up_blocks.2.attentions.0.transformer_blocks.0.attn2': 10,
                        'up_blocks.2.attentions.1.transformer_blocks.0.attn2': 11,
                        'up_blocks.2.attentions.2.transformer_blocks.0.attn2': 12,
                        'up_blocks.3.attentions.0.transformer_blocks.0.attn2': 13,
                        'up_blocks.3.attentions.1.transformer_blocks.0.attn2': 14,
                        'up_blocks.3.attentions.2.transformer_blocks.0.attn2': 15,
                        }
# 1 temporal single layer
temporal_module_names = {'transformer_in.transformer_blocks.0.attn2': 0}

def get_layer_id_info(block_name):
    xx = block_name.split('.')
    layer_info = list(filter(lambda x:x.isdigit(), xx))
    layer_id = ".".join(layer_info)
    yy = [xx[0]] + layer_info
    layer_id_info = ".".join(yy)
    return layer_id_info

def create_attention_video(attention_probs, attention_probs_copied=None, token_inds=None, attn_type="", block_name="", 
                           frame_i=0, frame_j=0, no_opt=True, time_bf_motion=False, show_colorbar=True,
                        #    interval=20, 
                            first_title="", second_title="", timestep=None,
                           fontsize=16):

    # visual inspection
    token_vis_id = token_inds[0] if len(token_inds) == 1 else token_inds[-1]

    # dim_vis_id = 0
    comb_name = f'{attn_type}_{block_name}'
    filename = f"output/attn_viz/videos/layers/attn_orig_vs_edited_{comb_name}.mp4"
    # print(block_name)
    layer_id_info = get_layer_id_info(block_name)

    if no_opt==False:
        # manual selection: high focus cross-attention layer ('up_blocks.2.2.0')
        # 1 for now
        vis_layers = ['up_blocks.2.2.0'] # 'up_blocks.2.2.0' 'down_blocks.0.0.0'
        if layer_id_info in vis_layers:
            filename = filename.replace('attn_viz', 'opt_viz')
        else:
            return 
   
    if attn_type.startswith("spatial"):
        # layer_id = spatial_module_names[block_name]
        layer_desc = f"Spatial atten layer: {layer_id_info}"
        fps = 24
    elif attn_type.startswith("temporal"):
        # layer_id = temporal_module_names[block_name]
        layer_desc = f"Temporal atten layer: {layer_id_info}"
        fps = 24

    hmaps_motion_clips = []
    horizontal_len = 1
    att_probs = attention_probs.detach().cpu().numpy()
    if attention_probs_copied!=None:
        att_probs_copied = attention_probs_copied.detach().cpu().numpy()
        horizontal_len = 2
    
    if attn_type.startswith("spatial"):    
        interval = 20 # total 120 channels (5 channels per frame) - sparsely select e.g 6/120 of them
    elif attn_type.startswith("temporal"):
        interval = 1 # total 8 channels

    if attn_type.endswith("_cross_frame"):
        assert attention_probs.shape[-2] == attention_probs.shape[-1], 'n_frames x nframes dont match. Is this really a temporal cross-frame?'
        range_max = attention_probs.shape[-1] # vis frames cross frame attention (only for temporal)
    else:
        range_max = attention_probs.shape[0] # vis layer channel attentions

    for dim_vis_id in range(0, range_max, interval):
        fig, axs = plt.subplots(1, horizontal_len, figsize=(10, 10))
        
        if first_title=="":
            first_title = "attention_map"
        if second_title=="":
            second_title = "attention_map_copied_and_edited"

        if attention_probs_copied!=None:
            if attn_type == "spatial":
                img0 = axs[0].imshow(att_probs[dim_vis_id,:,:,token_vis_id])
                axs[0].set_title(first_title, fontsize=fontsize)
                if show_colorbar:
                    # use fraction to adjust the height of the color bar relative with the plot
                    # use pad to adjust the space between colorbar and the plot
                    colorbar = plt.colorbar(img0, ax=axs[0], fraction=0.046, pad=0.04)

                img1 = axs[1].imshow(att_probs_copied[dim_vis_id,:,:,token_vis_id])
                axs[1].set_title(second_title, fontsize=fontsize)
                if show_colorbar:
                    colorbar = plt.colorbar(img1, ax=axs[1], fraction=0.046, pad=0.04)
                layer_desc_up = layer_desc + f" | channel: {dim_vis_id}"

            elif attn_type == "temporal": # primarily for `self` frame, but can be used for others
                img0 = axs[0].imshow(att_probs[dim_vis_id,:,:,frame_i, frame_j])
                axs[0].set_title(first_title, fontsize=fontsize)
                if show_colorbar:
                    colorbar = plt.colorbar(img0, ax=axs[0], fraction=0.046, pad=0.04)

                img1 = axs[1].imshow(att_probs_copied[dim_vis_id,:,:,frame_i, frame_j])
                axs[1].set_title(second_title, fontsize=fontsize)
                if show_colorbar:
                    colorbar = plt.colorbar(img1, ax=axs[1], fraction=0.046, pad=0.04)

                layer_desc_up = layer_desc + f" | channel: {dim_vis_id}"
            
            elif attn_type.endswith("_cross_frame"):
                last_channel = att_probs.shape[0]-1
                frame_j = dim_vis_id
                
                try:
                    img0 = axs[0].imshow(att_probs[last_channel,:,:,frame_i, frame_j])
                    axs[0].set_title(first_title, fontsize=fontsize)
                    if show_colorbar:
                        colorbar = plt.colorbar(img0, ax=axs[0], fraction=0.046, pad=0.04)
                        
                    img1 = axs[1].imshow(att_probs_copied[last_channel,:,:,frame_i, frame_j])
                    axs[1].set_title(second_title, fontsize=fontsize)
                    if show_colorbar:
                        colorbar = plt.colorbar(img1, ax=axs[1], fraction=0.046, pad=0.04)

                    layer_desc_up = layer_desc + f" | channel: {last_channel}"
                except:
                    pdb.set_trace()
        else:
        
            if attn_type == "spatial":
                img0 = axs.imshow(att_probs[dim_vis_id,:,:,token_vis_id])
                axs.set_title("attention_map", fontsize=fontsize)
                if show_colorbar:
                    colorbar = plt.colorbar(img0, ax=axs, fraction=0.046, pad=0.04)

                layer_desc_up = layer_desc + f" | channel: {dim_vis_id}"

            elif attn_type == "temporal":
                img0 = axs.imshow(att_probs[dim_vis_id,:,:,frame_i, frame_j])
                axs.set_title("attention_map", fontsize=fontsize)
                if show_colorbar:
                    colorbar = plt.colorbar(img0, ax=axs, fraction=0.046, pad=0.04)

                layer_desc_up = layer_desc + f" | channel: {dim_vis_id}"
            
            elif attn_type.endswith("_cross_frame"):
                last_channel = 7
                frame_j = dim_vis_id
                img0 = axs.imshow(att_probs[last_channel,:,:,frame_i, frame_j])
                axs.set_title("attention_map", fontsize=fontsize)
                if show_colorbar:
                    colorbar = plt.colorbar(img0, ax=axs, fraction=0.046, pad=0.04)

                layer_desc_up = layer_desc + f" | channel: {last_channel}"
            
            layer_desc_up = "NO EDIT - " + layer_desc_up


        if attn_type.startswith("temporal"):
            layer_desc_up = layer_desc_up + f" | frame i: {frame_i}, frame j: {frame_j}"
 
        # Add overall title
        fig.suptitle(f"{layer_desc_up}", y=0.8, fontsize=fontsize) # Adjust y to move the title down
        if attention_probs_copied!=None:
            fig.tight_layout()
        else:
            plt.subplots_adjust(left=0.25, right=0.75, top=0.65, bottom=0.25)

        image_ = np.array(fig2img(fig))
        resized_image_ = resize_image_to_macro_block(image_)
        hmaps_motion_clips.append(resized_image_)
        plt.close('all')
    
    hmaps_video = np.stack(hmaps_motion_clips, axis=0)
    if no_opt==False and time_bf_motion:
        arr_filename = filename.replace('layers', 'arrays').replace('mp4', 'npy')
        np.save(arr_filename, hmaps_video)
    else:
        imageio.mimwrite(filename, hmaps_video, fps=fps)
    # print("spatial video attended...")

def plot_activations(cross_attn, prompt, plot_with_trailings=False):
    num_frames = cross_attn.shape[0]
    cross_attn = cross_attn.cpu()
    for i in range(num_frames):
        filename = "/tmp/out.{:04d}.jpg".format(i)
        plot_activation(cross_attn[i], prompt, filename, plot_with_trailings)


def plot_activation(cross_attn, prompt, filepath="", plot_with_trailings=False):

    splitted_prompt = prompt.split(" ")
    n = len(splitted_prompt)
    start = 0
    arrs = []
    if plot_with_trailings:
        for j in range(5):
            arr = []
            for i in range(start, start + n):
                cross_attn_sliced = cross_attn[..., i + 1]
                arr.append(cross_attn_sliced.T)
            start += n
            arr = np.hstack(arr)
            arrs.append(arr)
        arrs = np.vstack(arrs).T
    else:
        arr = []
        for i in range(start, start + n):
            print(i)
            cross_attn_sliced = cross_attn[..., i + 1]
            arr.append(cross_attn_sliced)
        arrs = np.hstack(arr).astype(np.float32)
    plt.clf()

    v_min = arrs.min()
    v_max = arrs.max()
    n_min = 0.0
    n_max = 1

    arrs = (arrs - v_min) / (v_max - v_min)
    arrs = (arrs * (n_max - n_min)) + n_min

    plt.imshow(arrs, cmap="jet")
    plt.title(prompt)
    plt.colorbar(orientation="horizontal", pad=0.2)
    if filepath:
        plt.savefig(filepath)
        log.info(f"Saved [{filepath}]")
    else:
        plt.show()


def get_cross_attn(
    unet,
    resolution=32,
    target_size=64,
):
    """To get the cross attention map softmax(QK^T) from Unet.
    Args:
        unet (UNet2DConditionModel): unet
        resolution (int): the cross attention map with specific resolution. It only supports 64, 32, 16, and 8
        target_size (int): the target resolution for resizing the cross attention map
    Returns:
        (torch.tensor): a tensor with shape (target_size, target_size, 77)
    """
    attns = []
    check = [8, 16, 32, 64]
    if resolution not in check:
        raise ValueError(
            "The cross attention resolution only support 8x8, 16x16, 32x32, and 64x64. "
            "The given resolution {}x{} is not in the list. Abort.".format(
                resolution, resolution
            )
        )
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        # NOTE: attn2 is for cross-attention while attn1 is self-attention
        dim = resolution * resolution
        if not hasattr(module, "processor"):
            continue
        if hasattr(module.processor, "cross_attention_map"):
            attn = module.processor.cross_attention_map[None, ...]
            attns.append(attn)

    if not attns:
        print("Err: Quried attns size [{}]".format(len(attns)))
        return
    attns = torch.cat(attns, dim=0)
    attns = torch.sum(attns, dim=0)
    # resized = torch.zeros([target_size, target_size, 77])
    # f = torchvision.transforms.Resize(size=(64, 64))
    # dim = attns.shape[1]
    # print(attns.shape)
    # for i in range(77):
    #     attn_slice = attns[..., i].view(1, dim, dim)
    #     resized[..., i] = f(attn_slice)[0]
    return attns


def get_avg_cross_attn(unet, resolutions, resize):
    """To get the average cross attention map across its resolutions.
    Args:
        unet (UNet2DConditionModel): unet
        resolution (list): a list of specific resolution. It only supports 64, 32, 16, and 8
        target_size (int): the target resolution for resizing the cross attention map
    Returns:
        (torch.tensor): a tensor with shape (target_size, target_size, 77)
    """
    cross_attns = []
    for resolution in resolutions:
        try:
            cross_attns.append(get_cross_attn(unet, resolution, resize))
        except:
            log.warn(f"No cross-attention map with resolution [{resolution}]")
    if cross_attns:
        cross_attns = torch.stack(cross_attns).mean(0)
    return cross_attns


def save_cross_attn(unet):
    """TODO: to save cross attn"""
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn2" in name:
            folder = "/tmp"
            filepath = os.path.join(folder, name + ".pt")
            torch.save(module.attn, filepath)
            print(filepath)


def use_dd(unet, use=True):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn2" in name:
            module.processor.use_dd = use


def use_dd_temporal(unet, use=True):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn2" in name:
            module.processor.use_dd_temporal = use


def get_loss(unet):
    loss = 0
    total = 0
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "Attention" and "attn2" in name:
            loss += module.processor.loss
            total += 1
    return loss / total


def get_params(unet):
    parameters = []
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "Attention" and "attn2" in name:
            parameters.append(module.processor.parameters)
    return parameters
