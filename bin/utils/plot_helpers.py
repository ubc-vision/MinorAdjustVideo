import os
import io
import cv2
import pdb
import glob
import wandb
import torch
import subprocess
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from bin.utils.misc import convert_bbox_to_image_space


def plot_opt_temporal_maps(temporal_attn, diff_bboxmap_str, modified_copy_temporal_attn_str, 
                           modified_copy_temporal_attn_wk, n_opt_iterations="", output_opt_viz_path="",
                            opt_id="", timestep="", wandb_log=False, plot_local=True, mini_label=""):                                                             
    
    if wandb_log or plot_local:
        first_id, frame_i, frame_j = 0,2,12
        fontsize = 10

        if opt_id==0:
            fig, axs = plt.subplots(1, 4, figsize=(10, 10)) 
            
            img0 = axs[0].imshow(temporal_attn[first_id,:,:,frame_i, frame_j].detach().cpu()) 
            axs[0].set_title("bottom_attention_map", fontsize=fontsize)
            axs[0].axis("off")
            colorbar = plt.colorbar(img0, ax=axs[0], fraction=0.046, pad=0.04)


            img1 = axs[1].imshow(diff_bboxmap_str[first_id,:,:,frame_i, frame_j].detach().cpu()) 
            axs[1].set_title(f"mask_bbox_str [t:{timestep} opt_id:{opt_id}]", fontsize=fontsize)
            axs[1].axis("off")
            colorbar = plt.colorbar(img1, ax=axs[1], fraction=0.046, pad=0.04)

            img2 = axs[2].imshow(modified_copy_temporal_attn_str[first_id,:,:,frame_i, frame_j].detach().cpu()) 
            axs[2].set_title(f"edited_attention_map_str", fontsize=fontsize)
            axs[2].axis("off")
            colorbar = plt.colorbar(img2, ax=axs[2], fraction=0.046, pad=0.04)

            img3 = axs[3].imshow(modified_copy_temporal_attn_wk[first_id,:,:,frame_i, frame_j].detach().cpu()) 
            axs[3].set_title(f"edited_attention_map_wk", fontsize=fontsize)
            axs[3].axis("off")
            colorbar = plt.colorbar(img3, ax=axs[3], fraction=0.046, pad=0.04)

            if wandb_log:
                log_step = (timestep * n_opt_iterations) + opt_id
                wandb.log({"temporal/attn_map-diff_boxmap-edited_attn_map": plt}, step=log_step)

            fig.tight_layout()
            save_dir = output_opt_viz_path / 'temporal'
            # save_dir = Path('output/opt_viz/temporal')
            save_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(f"{str(save_dir)}/maps_diff_map-attn_map_temporal_t_{timestep:04d}_opt_idx_{opt_id:04d}.png", bbox_inches='tight')

def plot_single_spatial_out_map(bottom_attention_copy, 
                                n_frames=None, 
                                n_opt_iterations="",
                                # scale_box_map=None,
                                opt_id="", timestep="", all_tokens_inds=None, wandb_log=False, 
                                output_opt_viz_path= "", plot_local=False,
                                mini_label=""):


    if wandb_log or plot_local:
        first_token = 0
        fontsize = 10
        frame_size = bottom_attention_copy.shape[0] // n_frames

        # layout format (Trailblazer backbone): e.g up_blocks.2.2.0 layer -> edited lower channel 240, 20, 20, 77 (see attention map table in appendix)
        # 240 for 24 frames -> 10 up-2.2.0 layers per frame
        # --- Frame 0: Layer 0 - 9
        # --- Frame 3: Layer 30 - 39
        # --- Frame 15: Layer 150 - 159
        # --- Frame 23: Layer 230 - 239

        for frame_idx in range(0, n_frames):
            
            # choice frames
            if frame_idx in [0, 3, 15, 23]:
                first_id = frame_idx * frame_size
                fig, axs = plt.subplots(1, 1, figsize=(10, 10)) 

                img0 = axs.imshow(bottom_attention_copy[first_id,:,:,all_tokens_inds][..., first_token].detach().cpu()) 
                axs.axis("off")

                if wandb_log:
                    log_step = (timestep * n_opt_iterations) + opt_id
                    wandb.log({f"spatial/{mini_label}_attn_map-diff_boxmap-edited_attn_map": plt}, step=log_step)

                if plot_local:
                    fig.tight_layout()
                    save_dir = output_opt_viz_path / 'spatial'
                    save_dir.mkdir(parents=True, exist_ok=True)
                    fig.savefig(f"{str(save_dir)}/{mini_label}_maps_diff_map-attn_map_spatial_t_{timestep:04d}_opt_idx_{opt_id:04d}_frame_idx_{frame_idx:04d}.png", bbox_inches='tight')


def plot_opt_spatial_maps(bottom_attention_copy, diff_bboxmap_str, modified_bottom_attention_copy_str, 
                          modified_bottom_attention_copy_wk, n_opt_iterations="",
                          scale_box_map=None,
                            opt_id="", timestep="", all_tokens_inds=None, wandb_log=False, 
                            output_opt_viz_path= "", plot_local=True,
                            mini_label="",
                            use_color_minmax: bool = False):

    """
    bottom_attention_copy: original raw (pre-edit) attention map
    diff_bboxmap_str: unscaled strength map
    scale_box_map: scaled strength map
    modified_bottom_attention_copy_str: edited by both weaken + strengthen terms (see exact aggregation strategy `aggregate_str`, e.g add)
    modified_bottom_attention_copy_wk: edited by only scaled weaken map
    """
    
    if wandb_log or plot_local:
        first_id, first_token = 0,0

        title_fontsize = 25
        cbar_ticksize = 20
        dpi = 300
        
        # Wider layout for 1x3 panels
        fig, axs = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True) 

        # -------------

        A_map = bottom_attention_copy[first_id,:,:,all_tokens_inds][..., first_token].detach().cpu()
        S_map = diff_bboxmap_str[first_id,:,:,all_tokens_inds][..., first_token].detach().cpu()
        A_map_edited = modified_bottom_attention_copy_str[first_id,:,:,all_tokens_inds][..., first_token].detach().cpu()

        # Keep attention panels comparable by using shared vmin/vmax for attention maps
        if use_color_minmax:
            vmin_A = float(min(A_map.min(), A_map_edited.min()))
            vmax_A = float(max(A_map.max(), A_map_edited.max()))
            img0 = axs[0].imshow(A_map, vmin=vmin_A, vmax=vmax_A)
        else:
            img0 = axs[0].imshow(A_map)
        axs[0].set_title("Original attention map\n", fontsize=title_fontsize, fontweight="bold")
        axs[0].axis("off")
        colorbar = plt.colorbar(img0, ax=axs[0], fraction=0.046, pad=0.04)
        colorbar.ax.tick_params(labelsize=cbar_ticksize)

        img1 = axs[1].imshow(S_map) 
        axs[1].set_title(f"Differential edit map \n(step {timestep}, opt id {opt_id})", fontsize=title_fontsize, fontweight="bold")
        axs[1].axis("off")
        colorbar = plt.colorbar(img1, ax=axs[1], fraction=0.046, pad=0.04)
        colorbar.ax.tick_params(labelsize=cbar_ticksize)

        if use_color_minmax:
            img2 = axs[2].imshow(A_map_edited, vmin=vmin_A, vmax=vmax_A)
        else:
            img2 = axs[2].imshow(A_map_edited)
        axs[2].set_title(f"Edited attention map\n", fontsize=title_fontsize, fontweight="bold")
        axs[2].axis("off")
        colorbar = plt.colorbar(img2, ax=axs[2], fraction=0.046, pad=0.04)
        colorbar.ax.tick_params(labelsize=cbar_ticksize)


        if wandb_log:
            # if timestep==1 and opt_id==2:
            log_step = (timestep * n_opt_iterations) + opt_id
            wandb.log({f"spatial/{mini_label}_attn_map-diff_boxmap-edited_attn_map": plt}, step=log_step)

        if plot_local:
            save_dir = output_opt_viz_path  / 'spatial'
            save_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                f"{str(save_dir)}/{mini_label}_maps_diff_map-attn_map_spatial_t_{timestep:04d}_opt_idx_{opt_id:04d}.png",
                bbox_inches='tight',
                dpi=dpi,
            )
        
        # Important: close figures in loops to avoid memory growth
        plt.close(fig)



def plot_bbox_gradients(bboxes_ratios, opt_idx, t):
    gradients = {
        'left': bboxes_ratios.grad[:,0].detach().cpu(),
        'top': bboxes_ratios.grad[:,1].detach().cpu(),
        'right': bboxes_ratios.grad[:,2].detach().cpu(),
        'bottom': bboxes_ratios.grad[:,3].detach().cpu(),
    }

    # Plot histograms
    plt.figure(figsize=(12, 8))

    for i, (coord, grad) in enumerate(gradients.items()):
        plt.subplot(2, 2, i + 1)
        plt.hist(grad.flatten(), bins=50, alpha=0.75, color='blue', edgecolor='black')
        plt.title(f'Histogram of {coord} Gradients')
        plt.xlabel('Gradient Value')
        plt.ylabel('Frequency')
    

    # Add an overall title
    plt.suptitle(f'Gradients of bbox coords opt_idx: {opt_idx:04d} timestep: {t:04d}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for the title
    save_dir = Path('output/opt_viz/gradients')
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{str(save_dir)}/hist_gradients_opt_idx_{opt_idx:04d}_timestep_{t:04d}.png")

def plot_bboxes(bbox_ratios, image, display_size=512,
                opt_iter="", abs_denoise_step="", rel_denoise_step="", total_rel_denoise_step="", path2save=""):
    
    width, height, _ = image.shape
    bbox_locs = convert_bbox_to_image_space(bbox_ratios, width=width, height=height)
    bbox_locs = bbox_locs.numpy()

    # Extract bbox coordinates
    left = bbox_locs[:, 0]
    top = bbox_locs[:, 1]
    right = bbox_locs[:, 2]
    bottom = bbox_locs[:, 3]
    
    # Vectorized operation using OpenCV
    # KEY: use a single for-loop with zip operation
    for l, t, r, b in zip(left, top, right, bottom):
        cv2.rectangle(image, (l, t), (r, b), (0, 255, 0), 1)

    # TODO
    # simply keep in original resolution even if small?
    # current challenge: CV2 use WH, PIL in `draw_text` uses HW
    if width!=display_size:
        # assert width==height, 'you need to handle unequal resolution...'
        image = cv2.resize(image, (width, height))
    # else:
    #     image = cv2.resize(image, (display_size, display_size))

    text = f"opt-iter {opt_iter:03d} | denoise iter {rel_denoise_step}/{total_rel_denoise_step} step ({abs_denoise_step:03d})"
    image = draw_text(image, text, text_position=(50, 10))
    return image

def plot_simple_bboxes(bbox_ratios, image, width, height, path2save=""):
    
    bbox_locs = convert_bbox_to_image_space(bbox_ratios, width=width, height=height)
    bbox_locs = bbox_locs.numpy()

    # Extract bbox coordinates
    left = bbox_locs[:, 0]
    top = bbox_locs[:, 1]
    right = bbox_locs[:, 2]
    bottom = bbox_locs[:, 3]
    
    # Vectorized operation using OpenCV
    # KEY: use a single for-loop with zip operation
    for l, t, r, b in zip(left, top, right, bottom):
        cv2.rectangle(image, (l, t), (r, b), (0, 255, 0), 1)

    text = f""
    image = draw_text(image, text, text_position=(50, 10))
    return image, bbox_locs


def draw_text(image, text, text_position=(120, 150), font_size=20, text_color = (0, 0, 0)):
    # Define the font, scale, color, and thickness of the text
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # font_scale = 0.1
    
    # text_thickness = 0.5
    # Put the text on the image
    # cv2.putText(image, text, text_position, font, font_scale, text_color, text_thickness)

    # Convert the OpenCV image to a PIL image
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # Load the DejaVuSans font
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf" 
    font = ImageFont.truetype(font_path, font_size)

    # Create a drawing context
    draw = ImageDraw.Draw(pil_image)
    # Draw the text on the image
    draw.text(text_position, text, font=font, fill=text_color)
    # Convert the PIL image back to an OpenCV image
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return image

def draw_arrow(image, start_point, end_point, thickness=2, color=(0, 255, 0)):
    cv2.arrowedLine(image, start_point, end_point, color, thickness, tipLength=0.5)

def draw_bbox(image, bbox, thickness=2, color=(0, 255, 0)):
    xmin, ymin, xmax, ymax = bbox
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)

def save_cv2_image(image, save_url):
    # expects BGR format and saves in BGR. Else, you see color flip 
    cv2.imwrite(save_url, image)

def delete_images(image_dir, img_ext=".jpeg"):
    [os.remove(f) for f in glob.glob(f'{image_dir}/*{img_ext}') if os.path.isfile(f)]

def write_video(cv_video, image):
    cv_video.write(image.astype('uint8'))

def save_image(image, filename="chk_img", folder="", title="", show_colorbar=True):
    """temporary image"""
    save_path = "checkers/imgs" if folder=="" else f"{folder}"

    if not isinstance(image, np.ndarray):
        np_image = convert_to_numpy(image)
        assert isinstance(np_image, np.ndarray), "array is not in numpy format"
    else:
        np_image = image

    if len(np_image.shape)==3 and (np_image.shape[0]<np_image.shape[2]):
        np_image = np.transpose(np_image, (1,2,0)) # make channel last

    img = plt.imshow(np_image)
    plt.title(title, fontsize=20)
    plt.axis("off")
    if show_colorbar:
        colorbar = plt.colorbar(img)
    plt.savefig(f"{save_path}/{filename}.png")
    plt.close()

def plot2chk_image(image, filename="chk_img", 
                    folder="checkers/imgs", 
                    title="", show_colorbar=True, 
                    cmap='viridis', off_axis=True,
                    figsize=(10, 10)):
    """temporary image
    - expect and works with RGB
    """

    np_image = convert_to_numpy(image)
    # isinstance is better, as strict type check can sometimes fail for things it cant allow
    assert isinstance(np_image, np.ndarray), "array is not in numpy format"

    if len(np_image.shape)==3 and (np_image.shape[0]<np_image.shape[2]):
        np_image = np.transpose(np_image, (1,2,0)) # make channel last

    plt.figure(figsize=figsize)
    img = plt.imshow(np_image, cmap=cmap)
    plt.title(title, fontsize=20)
    
    if off_axis:
        plt.axis("off")
    if show_colorbar:
        colorbar = plt.colorbar(img)
    
    plt.tight_layout()
    plt.savefig(f"{folder}/{filename}.png")
    plt.close('all')

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

"""
Helper functions adapted from https://github.com/ShirAmir/dino-vit-features.
"""
def convert_to_numpy(arr):
    if not isinstance(arr, np.ndarray):
        if arr.is_cuda:
            arr = arr.detach().cpu()
        arr = arr.numpy()
    return arr


def convert_video_to_view(input_path):

    # Verify the input path exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Create a temporary output file path
    temp_output_path = input_path + "_temp.mp4"

    # Run FF/mpeg to convert the video
    command = [
        'ffmpeg',
        '-i', input_path,
        '-codec:v', 'libx264',  # Video codec
        '-codec:a', 'aac',      # Audio codec
        '-strict', 'experimental',  # Allow experimental codecs
        temp_output_path,
        '-y' # overwrite
    ]

    # command = f'ffmpeg -i {input_path} -codec:v libx264 -codec:a aac -strict experimental {temp_output_path} -y'
    # os.system(command)

    subprocess.run(command, check=True, 
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE  # suppress ffmpeg output
                   )
    
    # Check if the temp output file was created successfully
    if os.path.exists(temp_output_path):
        # Remove the original file and rename the temporary file to the original name
        os.remove(input_path)
        os.rename(temp_output_path, input_path)
    else:
        print("Temporary output file was not created.")
        raise FileNotFoundError("Temporary output file was not created.")