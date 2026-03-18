
import os
import cv2
import pdb
import sys
import json
import tqdm
import torch
import imageio
import contextlib
import subprocess
import numpy as np
import humanize as read_numbers
import matplotlib.pyplot as plt
import moviepy.editor as mp


top_k_yaml_dir = "Data/AnimalKingdom/video_grounding/top_k_yamls"

# Define a context manager to suppress stdout i.e console prinouts
@contextlib.contextmanager
def suppress_printouts():
    with open('/dev/null', 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            
def get_average_box_gap_apart(bboxes_ratios):
    horizontal_gap = bboxes_ratios[:, 2] - bboxes_ratios[:, 0]
    vertical_gap = bboxes_ratios[:, 3] - bboxes_ratios[:, 1]
    gap_2d = torch.stack([horizontal_gap, vertical_gap], dim=-1)
    gap = gap_2d.mean()
    return gap

def plot_check_trajectories(clip_scene_cut_frame_data_set, img_descr="", 
                            add_object_name=False, object_name="",
                            video_id=""):
    for fi, cut_f_data in enumerate(clip_scene_cut_frame_data_set):
        first_location=0
        HW_f_data = cut_f_data[first_location]['HW']
        

        bbox_percents = torch.tensor(list(map(lambda x:x['bbox_percents'] , cut_f_data)))
        bbox_int = convert_bbox_to_image_space(bbox_percents, width=HW_f_data[1], height=HW_f_data[0])
        pt_img_space_mid_pts = get_bbox_midpoint(bbox_int)
        

        init_traj_filename = f'checkers/imgs/{img_descr}_traj_{fi:04d}_{pt_img_space_mid_pts.shape[0]}_box_pts_{video_id}.png'
        
        # TODO: confirm note below
        'NOTE: our ratio format (left, top, right, bottom) assumes HxW for the bboxes and the image'
        image = np.ones((int(HW_f_data[0]), int(HW_f_data[1]), 3), dtype=np.uint8) * 255
    
        plot_trajectory_on_image(image, pt_img_space_mid_pts, add_object_name=add_object_name, 
                                    object_name=object_name, filename=init_traj_filename)

def group_clip_to_scenecut_trajectories(frames_data, all_scene_bounds):
    clip_scene_cut_trajs = [] # keeps all
    clip_scene_cut_trajs_set = [] # drops empty trajectories

    for id in range(len(all_scene_bounds) - 1): 
        # each cut is defined by lower and upper bound
        lower_bound = all_scene_bounds[id]
        upper_bound = all_scene_bounds[id+1]

        temp = []
        temp_set = []
        for fi, f_data in enumerate(frames_data):
            if lower_bound <= f_data['frame_id'] <= upper_bound:
                temp.append(frames_data[fi])
                temp_set.append(frames_data[fi])
        
        # note, some cuts might be empty
        clip_scene_cut_trajs.append(temp)
        if len(temp_set)!=0:
            clip_scene_cut_trajs_set.append(temp)

    return clip_scene_cut_trajs, clip_scene_cut_trajs_set

def sort_based_on_frame_id(frames_data):
    sorted_frames_data = sorted(frames_data, key=lambda x: x['frame_id'])
    return sorted_frames_data

def plot_trajectory_on_image(image, most_continous_mid_pts, add_object_name=False, object_name="", filename=""):
    plt.imshow(image) 
    # xy = torch.stack(most_continous_mid_pts)
    
    
    xy = most_continous_mid_pts

    # amt=30
    plt.plot(xy[:, 0], xy[:, 1])
    plt.scatter(xy[1:-1, 0], xy[1:-1, 1], s=5, color="green", label="anchor points")
    
    # plots start and end points later, for final overlay
    plt.scatter(xy[0:1, 0], xy[0:1, 1], s=10, color="blue", label="start point")
    plt.scatter(xy[-1:, 0], xy[-1:, 1], s=10, color="red", label="end point")
    
    plt.legend()
    title = f'most_continous_locations_({xy.shape[0]} mid_pts)'
    if add_object_name:
        title = title + f'_[{object_name}]' 
    plt.title(title)

    if add_object_name:
        name, ext = os.path.splitext(filename)
        filename = name + f'_{object_name}' + ext
    plt.savefig(filename)
    plt.close('all')

    # if 'ABASTCVX_556_765' in filename:

def fill_with_interp_data(frames_data, detected_frame_ids, start_frame, stop_frame):
    # ref: https://github.com/hohonu-vicml/TrailBlazer/blob/de2696ef50537a473ab5afc57de89a5edc97c00b/TrailBlazer/Pipeline/Utils.py#L126C5-L126C68
    interp = lambda start, end, index: (1 - index) * start + index * end
    
    # extract start and stop frame item
    start_item_ = list(filter(lambda x:x['frame_id']==start_frame, frames_data))
    stop_item_ = list(filter(lambda x:x['frame_id']==stop_frame, frames_data))
    assert len(start_item_)==1, f'start_item is either empty or found more than one. FOUND {len(start_item)}'
    assert len(stop_item_)==1, f'stop_item is either empty or found more than one. FOUND {len(stop_item)}'

    # remove list
    start_item = start_item_[0]
    stop_item = stop_item_[0]
    
    start_bbox = start_item["bbox_percents"]
    end_bbox = stop_item["bbox_percents"]
    HW = start_item["HW"]
    
    clip_length = stop_frame - start_frame + 1
    for fr in range(clip_length):
        index = float(fr) / (clip_length - 1)
        bbox = []
        for j in range(4):
            bbox.append(interp(start_bbox[j], end_bbox[j], index))

        interp_frame_id = start_frame + fr
        # skip, avoid overwriting available frames i.e first and last - the interpolation anchor frames
        if interp_frame_id not in detected_frame_ids:                
            created_interp_data = {
                            "frame_id": interp_frame_id,
                            "bbox_percents": bbox,
                            "bbox_conf_score": 'interpolated',
                            "HW": HW
                        }
            frames_data.append(created_interp_data)

def get_closest_scene_bound(frame_id, bounds):
    'assumes bounds in numpy'
    offsets = bounds - frame_id
    mask = offsets >= 0 
    closest_bound = bounds[mask][0]
    return closest_bound

def check_resolution_order(arr, expect="HW"):
    if expect=="HW":
        assert arr.shape[1] <= arr.shape[2], f'for HW order, expects height should be less than width - current shape {arr.shape}'

def is_it_normalized(data, min, max, margin=1):
    print(f"min {min}, max {max}", min-margin, max+margin)
    assert not (min-margin <= data.min()) and (data.max() <= max+margin), f"you are trying to re-normalize a normalized data that is already scaled to [0-1] +/- margin of {margin}. min {data.min()} max {data.max()}"

def convert_video(input_path):

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
    try:
        subprocess.run(command, check=True, 
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE  # suppress ffmpeg output
                    )
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e.stderr}")
        raise
    
    # Check if the temp output file was created successfully
    if os.path.exists(temp_output_path):
        # Remove the original file and rename the temporary file to the original name
        os.remove(input_path)
        os.rename(temp_output_path, input_path)
    else:
        print("Temporary output file was not created.")
        raise FileNotFoundError("Temporary output file was not created.")
    
def check_readbable(video_path):
    try:
        vid = imageio.get_reader(video_path)
    except Exception as e:
        print(f'{e} - {video_path}')

def video_to_gif(video_path, gif_path, start_time=None, end_time=None, off_printouts=False, fps=24):
    # Load the video
    video_clip = mp.VideoFileClip(video_path)
    
    # Trim the video to the specified duration
    if start_time!=None and end_time!=None:
        video_clip = video_clip.subclip(start_time, end_time)
    
    # Write the video clip to a GIF file
    if off_printouts:
        with suppress_printouts():
            video_clip.write_gif(gif_path, fps=fps)
    else:
        video_clip.write_gif(gif_path, fps=fps)  # Adjust fps as needed

def get_topk_yamls():
    # load top k difficult examples 
    with open(f'{top_k_yaml_dir}/top_k_yamls_origin.json') as file:
        yamls = json.load(file)
        topk_yamls = list(yamls.keys())

    return topk_yamls

def drop_invalid_yamls(configs):
    # load noted invalid examples 
    invalid_yaml_dir = f"{top_k_yaml_dir.replace('top_k_yamls', 'invalid_yamls')}/invalid_yamls.json"
    with open(invalid_yaml_dir) as file:
        invalid_yamls = json.load(file)
        configs_left = [cfg for cfg in configs if cfg not in invalid_yamls]
        # configs_left = set(configs) - set(invalid_yamls)
    return configs_left

# Function to generate random integers with a specific seed
def generate_randint_with_seed(low, high, size, seed=42):
    # Save the current random state
    rng_state = torch.get_rng_state()
    # Set the seed for reproducibility
    if seed!=None:
        torch.manual_seed(seed)
    # Generate random integers
    result = torch.randint(low, high, (size,)) # tuple
    # Restore the previous random state
    torch.set_rng_state(rng_state)
    return result

def subselect_indices_max_spatial(total_frames, n_frames_per_video):

    max_start_idxs = total_frames - n_frames_per_video
    assert max_start_idxs > 0, 'negative maximum starting value is not allowed'

    # generate an initial set of candidate ids (i.e random starts)
    # indices = torch.randint(0, max_start_idxs, (1,)).tolist()
    rnd_start = generate_randint_with_seed(0, max_start_idxs, 1, seed=7).tolist()

    if len(rnd_start) == 1:
        indices = list(range(rnd_start[0], rnd_start[0]+n_frames_per_video))
    else:
        raise NotImplementedError

    return indices

def subselect_indices(total_frames, n_frames_per_video):

    # total_frames = frames.shape[0]
    quotient = total_frames / n_frames_per_video
    frame_skips = int(np_floor_or_ceil(quotient))  # Ensure it is an integer
    
    # Calculate indices
    indices = np.arange(0, total_frames, frame_skips)
    # Adjust if we have more than the required number of frames
    if len(indices) > n_frames_per_video:
        indices = indices[:n_frames_per_video]
    # Adjust if we have fewer than the required number of frames
    elif len(indices) < n_frames_per_video:
        indices = np.linspace(0, total_frames - 1, n_frames_per_video).astype(int)
    
    return indices

# def get_maximum_pt_euc_dist(continous_mid_pts):
def get_maximum_spatial_extent(continous_mid_pts):
    
    'revised version v2'
    # Compute pairwise Euclidean distances between all points
    pairwise_distances = torch.cdist(continous_mid_pts, continous_mid_pts, p=2)
    # Find the maximum distance
    maximum_euc_dist = pairwise_distances.max()

    """
    # find most minimum and maximum values across each dimension - but does not 
    ncessarily come from any two pair of points in the trajectory
    min_vals = continous_mid_pts.min(dim=0)[0]
    max_vals = continous_mid_pts.max(dim=0)[0]
    maximum_euc_dist = torch.norm(max_vals - min_vals)
   
    # min_x, min_y = torch.stack(continous_mid_pts)[:, 0].min(), torch.stack(continous_mid_pts)[:, 1].min()
    # max_x, max_y = torch.stack(continous_mid_pts)[:, 0].max(), torch.stack(continous_mid_pts)[:, 1].max()
    # maximum_euc_dist = torch.norm(torch.stack([max_x-min_x, max_y-min_y]))
    """
    return maximum_euc_dist

def time_taken(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    print(f'{int(d)} days :{int(h)} hrs :{int(m):02d} mins :{int(s):02d} secs\n') 

def args_to_str(args, exp_name):
    """Convert cmd line args into a logdir string for experiment logging"""
    exp_name += '_{}'.format(args.timestamp)
    exp_name += '_{}'.format(args.pose_model)
    return exp_name

def delete_videos(video_urls, ext=".mp4"):
    """Delete files given their URLs and extension (file paths)."""
    for video_url in video_urls:
        if os.path.exists(video_url):
            assert video_url.endswith(f"{ext}"), f"are you deleting a {ext}?"
            os.remove(video_url)

def resize_image_to_macro_block(image, macro_block_size=16):
    # make resolution marcro-block compatible
    height, width = image.shape[:2]
    new_height = (height + macro_block_size - 1) // macro_block_size * macro_block_size
    new_width = (width + macro_block_size - 1) // macro_block_size * macro_block_size
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image


def get_bbox_midpoint(bbox):
    # get bbox mid-location (x,y)
    xmin_A, ymin_A, xmax_A, ymax_A = bbox[:,0:1], bbox[:,1:2], bbox[:,2:3], bbox[:,3:4]
    bbox_wh = (xmax_A-xmin_A, ymax_A-ymin_A)
    midpointA = torch.concat([xmin_A+(bbox_wh[0]/2), ymin_A+(bbox_wh[1]/2)], dim=1)
    return midpointA

def get_2d_orientation(bboxA, bboxB):
    # compute 2D orientation given two bboxes
    assert 0.0 <= bboxA.mean() <= 1.0, f"bbox A needs to normalized to unit length!, you have mean of {bboxA.mean()}"
    assert 0.0 <= bboxB.mean() <= 1.0, f"bbox B needs to normalized to unit length!, you have mean of {bboxB.mean()}"

    midpointA = get_bbox_midpoint(bboxA)
    midpointB = get_bbox_midpoint(bboxB)
    orient_vec2d = midpointB - midpointA

    norm_orient_vec2d = orient_vec2d/torch.norm(orient_vec2d, dim=1, keepdim=True)
    # norm_orient_vec2d = orient_vec2d/np.linalg.norm(orient_vec2d)
    return orient_vec2d, norm_orient_vec2d

def convert_loc_to_percent(box, height, width):
    assert box[0] < box[2] and box[1] < box[3], 'asssumes left,top,right,bottom format'
    
    box_pt = box.clone() 
    box_pt[0] = box_pt[0]/width
    box_pt[1] = box_pt[1]/height
    box_pt[2] = box_pt[2]/width
    box_pt[3] = box_pt[3]/height
    return box_pt

def convert_to_int(pt):
    # floor or ceil point appropriately
    pt_int = pt.clone()
    pt_int = torch.where(pt % 1 < 0.5, torch.floor(pt), torch.ceil(pt)).int()
    return pt_int

def convert_pt_to_image_space(pt, width, height):
    # convert bbox mid point ratio to image coordinates
    'args: bbox mid pt (:, wh)'
    pt_img = pt.clone()
    pt_img[:, 0] = pt[:, 0] * width
    pt_img[:, 1] = pt[:, 1] * height

    pt_int = convert_to_int(pt_img)
    # pt_int = np.where(pt % 1 < 0.5, np.floor(pt), np.ceil(pt)).astype(int)
    return pt_int

def convert_bbox_to_image_space(bbox_ratio, width, height):
    # convert bbox ratios to image coordinates
    'args: bbox_ratio (:, left, top, right, bottom)'
    bbox = bbox_ratio.clone()
    bbox[:, 0] = bbox_ratio[:, 0] * width
    bbox[:, 2] = bbox_ratio[:, 2] * width
    bbox[:, 1] = bbox_ratio[:, 1] * height
    bbox[:, 3] = bbox_ratio[:, 3] * height

    # floor or ceil appropriately
    bbox_int = torch_floor_or_ceil(bbox)
    # bbox_int = torch.where(bbox % 1 < 0.5, torch.floor(bbox), torch.ceil(bbox)).int()
    # bbox_int = np.where(bbox % 1 < 0.5, np.floor(bbox), np.ceil(bbox)).astype(int)
    return bbox_int

def torch_floor_or_ceil(val):
    return torch.where(val % 1 < 0.5, torch.floor(val), torch.ceil(val)).int()

def np_floor_or_ceil(val):
    return np.where(val % 1 < 0.5, np.floor(val), np.ceil(val)).astype(int)

def inverse_normalize_image(image):
    # Undo normalization (C,H,W)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if len(image.shape)==4:
        mean = torch.tensor(mean).reshape(1, 3, 1, 1)
        std = torch.tensor(std).reshape(1, 3, 1, 1)
    elif len(image.shape)==3:
        mean = torch.tensor(mean).reshape(3, 1, 1)
        std = torch.tensor(std).reshape(3, 1, 1)
    else:
        raise NotImplementedError
    return image * std + mean

def args_to_str(args, exp_name):
    """Convert cmd line args into a logdir string for experiment logging"""
    exp_name += '_{}'.format(args.get('timestamp'))
    return exp_name

def time_taken(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    print(f'{int(d)} days :{int(h)} hrs :{int(m):02d} mins :{int(s):02d} secs\n') 

def count_params(model, name=""):
    # Count the number of trainable parameters
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Convert to gigabytes (GB)
    total_trainable_params_gb = total_trainable_params / 1_073_741_824  # 1,073,741,824 bytes in a gigabyte
    print(f"Total trainable parameters in the {name} model: {read_numbers.intword(str(total_trainable_params))} - [{total_trainable_params}]")


def save_net(model, optimizer, path, global_step, iter_stats=None, scheduler=None, 
            run_id="", key_name=None, best_val_error=None):
    """Save all model state dict."""
    if key_name == None: key_name = "net_state_dict"
    dict_ = {'global_step': global_step, 
             'iter_stats': iter_stats, 
                f'{key_name}': model.state_dict(), 
                'optimizer_state_dict': optimizer.state_dict(), 
                'run_id': run_id
                }
    if scheduler is not None:
        dict_['scheduler_state_dict'] = scheduler.state_dict()
    if best_val_error != None:
        dict_['best_val_error'] = best_val_error

    torch.save(dict_, path)
    print('Saved weights at', path)
    

def load_net(checkpt_path, model, optimizer=None, scheduler=None, global_step=None, iter_stats=None, 
             key_name=None, best_val_error=float("inf")):
    
    checkpoint = torch.load(checkpt_path)
    if key_name == None: key_name = "net_state_dict"
    model.load_state_dict(checkpoint[key_name])
    if optimizer != None: # not eval 
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        global_step = checkpoint['global_step']

    if "best_val_error" in checkpoint:
        best_val_error = checkpoint['best_val_error']

    if "iter_stats" in checkpoint:
        iter_stats = checkpoint['iter_stats']
        
    if 'scheduler' in checkpoint and scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print(f"Reloaded weights at {checkpt_path} \n")
    return model, optimizer, global_step, iter_stats, scheduler, best_val_error
