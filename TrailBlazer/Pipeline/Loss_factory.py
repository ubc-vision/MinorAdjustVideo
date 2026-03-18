
import pdb
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F

# from TrailBlazer.Pipeline.Utils import geometric_mean
from bin.utils.plot_helpers import plot2chk_image
from bin.utils.misc import get_average_box_gap_apart
                            

CLIP_N_TOKENS = 77

def compute_loss(cross_attn_loss, bundle, bboxes_ratios, 
                 init_bboxes_ratios=None, loss_dict={},
                 t_idx=None, t=None, opt_idx=None, n_opt_iterations=None,
                 strengthen_scale=None, weaken_scale=None, 
                 outside_bbox_loss_scale=1.0, box_temp_smooth_scale=1.0,
                 inside_bbox_attn_loss_scale=1.0,
                 use_mean=True, box_flip_thresh=0.1, 
                 box_flip_thresh_scale=10,
                wandb_log=False):
    
    neg_loss, shrink_loss, diff_loss = 0, 0, 0
    maximize_loss, minimize_loss, init_bbox_coords_loss, init_bbox_area_loss = 0, 0, 0, 0
    outside_bbox_loss, diff_loss = 0, 0
    inside_bbox_attn_loss, out_in_loss = 0, 0 


    if bundle.get('opt_str_scale'):
        str_scale_loss = zero_one_penalty_loss(strengthen_scale, lambda_=1, use_mean=use_mean)
    if bundle.get('opt_wk_scale'):
        wk_scale_loss = zero_one_penalty_loss(weaken_scale, lambda_=1, use_mean=use_mean)

    bb_deviate_lambda = bundle['bb_deviate_lambda']   

    # for statistics log
    init_area = get_box_areas(init_bboxes_ratios)
    opt_area = get_box_areas(bboxes_ratios)
    # keep the sign (+ve for over-growth, -ve for under-growth)
    area_delta = torch.mean(opt_area - init_area)
    optimized_box_mean_area = torch.mean(opt_area)

    if bundle.get('init_bbox_coords_loss'): # in fp32? 
        init_bbox_coords_loss = get_bbox_coords_deviation_loss(init_bboxes_ratios, bboxes_ratios,  
                                                 width=bundle['width'], height=bundle['height'], 
                                                 lambda_=bb_deviate_lambda, 
                                                 use_mean=use_mean)
        # loss agnostic: basic components
        minimize_loss = minimize_loss + init_bbox_coords_loss # + box_flip_loss

    if bundle.get('init_bbox_area_loss'): # in fp32? 
        init_bbox_area_loss = get_bbox_area_deviation_loss(init_bboxes_ratios, bboxes_ratios,  
                                                 width=bundle['width'], height=bundle['height'], 
                                                 lambda_=bb_deviate_lambda, 
                                                 use_mean=use_mean)
        # loss agnostic: basic components
        minimize_loss = minimize_loss + init_bbox_area_loss # + box_flip_loss

    if bundle.get('box_temp_smooth_loss'):
        box_temp_sm_loss = get_box_temp_smooth_loss(bboxes_ratios)
        minimize_loss = minimize_loss + (box_temp_sm_loss * box_temp_smooth_scale)

    if bundle.get('opt_str_scale'):
        minimize_loss = minimize_loss + str_scale_loss
    if bundle.get('opt_wk_scale'):
        minimize_loss = minimize_loss + wk_scale_loss

    outside_bbox_loss, inside_bbox_attn_loss, outside_bbox_agg_attn_value, inside_bbox_agg_attn_value, diff_loss, out_in_loss = cross_attn_loss 

    outside_bbox_loss = outside_bbox_loss * outside_bbox_loss_scale 
    inside_bbox_attn_loss = inside_bbox_attn_loss * inside_bbox_attn_loss_scale

    if loss_dict['diff_loss']:
        minimize_loss = minimize_loss + diff_loss # + shrink_loss 

    if out_in_loss:
        minimize_loss = minimize_loss + out_in_loss # + shrink_loss 
        # loss = out_in_loss + max(0, minimize_loss) # + shrink_loss 

    if loss_dict['max_cross_loss']:

        if use_mean:
            assert 0 <= inside_bbox_attn_loss <= 1, 'are the attention map values not 0-1?'
            minimize_loss = minimize_loss + inside_bbox_attn_loss + outside_bbox_loss 
        else:
            raise NotImplementedError


    'final collation/muster point'
    loss =  minimize_loss
    
    if wandb_log:
        loss_dict = {
        "losses/final_loss": loss,
        "losses/diff_loss": diff_loss,
        "losses/shrink_loss": shrink_loss,
        "losses/signed_bbox_area_growth_change": area_delta,
        "losses/bkgd_agg_attn_value": outside_bbox_agg_attn_value,
        "losses/fg_agg_attn_value": inside_bbox_agg_attn_value,
        "losses/maximize_loss": maximize_loss,
        "losses/init_bbox_coords_deviation_loss": init_bbox_coords_loss,
        "losses/init_bbox_area_deviation_loss": init_bbox_area_loss,
        "losses/minimize_loss": minimize_loss,
        "losses/fg_attn_loss": inside_bbox_attn_loss,
        "losses/bkgd_loss": outside_bbox_loss,
        "losses/neg_loss": neg_loss,
        "ratios/box_area_change-2-bg_loss_ratio": area_delta/outside_bbox_loss,
        "ratios/opt_box_mean_area-2-bg_loss_ratio": optimized_box_mean_area/outside_bbox_loss,
        "ratios/bg_attn_value-2-bg_loss_ratio": outside_bbox_agg_attn_value/outside_bbox_loss,
        "stats/denoising step (local)": t_idx,
        "stats/denoising step (global)": t,
        "stats/opt_idx": opt_idx
        }

        if bundle.get('opt_str_scale'):
            loss_dict['scale/strengthen_scale'] = strengthen_scale
            loss_dict['scale/strengthen_scale_grad'] = strengthen_scale.grad
        if bundle.get('opt_wk_scale'):
            loss_dict['scale/weaken_scale'] = weaken_scale
            loss_dict['scale/weaken_scale_grad'] = weaken_scale.grad

        log_step = (t_idx * n_opt_iterations) + opt_idx
        wandb.log(loss_dict, step=log_step)

    return loss

def l2_loss(target_, pred_, use_mean=True, eps=1e-32):
    # cast tensors to fp32 
    target, pred = target_.to(torch.float32), pred_.to(torch.float32)
    # calculate squared differences
    diff = (target - pred)
    diff = diff**2

    if diff.isinf().any() or diff.isnan().any():
        pdb.set_trace()

    # '''
    b = diff.shape[0]
    # a)
    mask = (target**2).reshape(b, -1).sum(dim=-1) # squared L2 norm -> sum squared diff
    diff = diff.reshape(b,-1).sum(dim=-1)  # squared L2 norm
    diff_value = diff / (mask + eps)
    
    loss = torch.mean(diff_value)
    loss = loss.to(torch.float16)

    return loss

def get_shrink_threshold(bbox_ratio, gap_ratio=0.50):
    # args: x (24, 4) - (:, left, top, right, bottom); gap_ratio: higher ratio keeps bbox maximum
    horizontal_gap = bbox_ratio[:, 2] - bbox_ratio[:, 0]
    vertical_gap = bbox_ratio[:, 3] - bbox_ratio[:, 1]
    # threshold
    horizontal_threshold = horizontal_gap * gap_ratio
    vertical_threshold = vertical_gap * gap_ratio
    
    shrink_threshold = torch.stack([horizontal_threshold, vertical_threshold], dim=-1)
    return shrink_threshold

def get_shrink_loss(bbox_ratio, shrink_thresh, lambda_, use_mean=True):
    horizontal_gap = bbox_ratio[:, 2] - bbox_ratio[:, 0]
    vertical_gap = bbox_ratio[:, 3] - bbox_ratio[:, 1]
    # compute difference to threshold
    horz_diff2thresh = horizontal_gap - shrink_thresh
    vert_diff2thresh = vertical_gap - shrink_thresh[:,1]

    diff2thresh = torch.stack([horz_diff2thresh, vert_diff2thresh], dim=-1)
    # compute shrink penalty
    shrink_loss = bound_squared_penalty_term(diff2thresh, lambda_=lambda_, use_mean=use_mean)
    return shrink_loss

def get_box_flip_loss(bbox_ratio, box_flip_thresh, lambda_, use_mean=True):
    horizontal_gap = bbox_ratio[:, 2] - bbox_ratio[:, 0]
    vertical_gap = bbox_ratio[:, 3] - bbox_ratio[:, 1]

    box_flip_threshold = torch.tensor([box_flip_thresh]).to( bbox_ratio.device)
    # compute difference to threshold
    horz_diff2thresh = horizontal_gap - box_flip_threshold
    vert_diff2thresh = vertical_gap - box_flip_threshold

    diff2thresh = torch.stack([horz_diff2thresh, vert_diff2thresh], dim=-1)
    # compute flip penalty
    box_flip_loss = bound_squared_penalty_term(diff2thresh, lambda_=lambda_, use_mean=use_mean)
    return box_flip_loss

def geometric_mean(width, height):
    # get balanced representation that takes both dimensions into account
    return torch.sqrt(width * height)

def get_box_temp_smooth_loss(bboxes):
    # ref: https://github.com/danielajisafe/Mirror-Aware-Neural-Humans/blob/6bf65b080a5a091ede20bf7758ce6d57045ddeda/core_mirror/loss_factory.py#L23
    bboxes_vel = bboxes[1:] - bboxes[:-1]
    bboxes_acel = bboxes_vel[1:] - bboxes_vel[:-1]

    # the natural bboxes are inherently non-smooth (because they were detected)
    # hence make sm_loss less sensitive to large changes in the box-parameter dimension
    # i.e use mean in 2nd dim, while being smooth in overall motion
    bboxes_acel = (bboxes_acel).pow(2.).mean(dim=1)
    sm_loss = bboxes_acel.sum().pow(0.5)
    return sm_loss

def get_bbox_coords_deviation_loss(init_bbox, opt_bbox, width, height, lambda_, use_mean=True):

    '''
    # left, top, right, bottom
    box_img_resolutions = torch.tensor([[width, height, width, height]]).to(init_bbox_.device)
    # penalize deviation in pixel space
    init_bbox = init_bbox_ * box_img_resolutions
    opt_bbox = opt_bbox_ * box_img_resolutions
    '''
    v_res = geometric_mean(torch.tensor(width), torch.tensor(height))

    if use_mean:
        loss = lambda_ * v_res * torch.mean((init_bbox-opt_bbox)**2)
    else:
        loss = lambda_ * v_res * torch.sum((init_bbox-opt_bbox)**2)
    return loss

def get_box_areas(bboxes):
    'left, top, right, bottom'
    width = bboxes[:,2] - bboxes[:,0]
    height = bboxes[:,3] - bboxes[:,1]
    areas = width * height
    return areas

def get_bbox_area_deviation_loss(init_bbox, opt_bbox, width, height, lambda_, use_mean=True):

    v_res = geometric_mean(torch.tensor(width), torch.tensor(height))
    init_area = get_box_areas(init_bbox)
    opt_area = get_box_areas(opt_bbox)

    if use_mean:
        loss = lambda_ * v_res * torch.mean((init_area-opt_area)**2)
    else:
        loss = lambda_ * v_res * torch.sum((init_area-opt_area)**2)
    return loss

def zero_one_penalty_loss(x, lambda_, use_mean=True):
    zero_loss = bound_squared_penalty_term(x, lambda_, bound=0, use_mean=use_mean)
    one_loss = bound_squared_penalty_term(x, lambda_, bound=1, use_mean=use_mean)
    return zero_loss + one_loss

def negative_loss(x, lambda_, use_mean=True):
    return bound_squared_penalty_term(x, lambda_, bound=0, use_mean=use_mean)


def bound_squared_penalty_term(x, lambda_, bound=0, use_mean=True):
    if bound==0:
        x_val = bound-x
    elif bound==1:
        x_val = x-bound

    # relu equates to using a maximum function
    if use_mean:
        penalty = lambda_ * torch.mean(torch.relu(x_val)**2)
    else:
        penalty = lambda_ * torch.sum(torch.relu(x_val)**2)
    return penalty

def get_attention_loss(unet, act="sigmoid", aggregate=torch.mean, num_frames=None):
    loss = 0
    total = 0

    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "Attention" and "attn2" in name:

            if "temp_attentions" in name:
                # note: found to be empty
                extracted_attention_map = module.processor.cross_attention_map
                if extracted_attention_map!=None:
                    intd_temp_loss = activation_attention_loss(extracted_attention_map, act=act, aggregate=aggregate, num_frames=num_frames)
                    print(f"intended temp loss {intd_temp_loss}")
                    loss += intd_temp_loss
                    total += 1
            else:
                extracted_attention_map = module.processor.cross_attention_map
                if extracted_attention_map!=None:
                    loss += extracted_attention_map.mean()
                    total += 1


    return loss / total

def get_loss(unet):
    loss = 0
    total = 0
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "Attention" and "attn2" in name:
            loss += module.processor.loss
            total += 1
    return loss / total

# log_softmax
def activation_attention_loss(attention_map, bounding_box=None, act="sigmoid", aggregate=torch.mean, num_frames=None):
    # TODO: too much workload in this function, please can split workload to other functions?

    if bounding_box!=None:
        x1, y1, x2, y2 = bounding_box.int()
        attention = attention_map[y1:y2, x1:x2]
    else:
        attention = attention_map

    activation_attention = process_act(act, attention, num_frames)
    
    # Convert to float32 before sum calculation to avoid overflow
    # if activation_attention.dtype==torch.float16:
    #     activation_attention = activation_attention.to(torch.float32)

    # using mean to avoid overflow (due to large tensors) + provide stable/scaled-down value | 64x64x24x24= 2_359_296
    loss = -aggregate(activation_attention)  # Maximize attention within the bounding box
    return loss 

def process_act(act, attention, num_frames):
    if act=="sigmoid":
        activation_attention = torch.sigmoid(attention)
    elif act=="softmax":
        if attention.shape[-1] == num_frames: # 64, 64, 24, 24
            w, h = attention.shape[0], attention.shape[1]
            flat_attn = attention.permute(2,3,0,1).reshape(num_frames, num_frames, -1)
            soft_attn = F.softmax(flat_attn, dim=-1)
            activation_attention = soft_attn.reshape(num_frames, num_frames, w, h).permute(2,3,0,1)
        elif attention.shape[-1] == CLIP_N_TOKENS: # 120, 64, 64, 77
            first_dim, w, h, n_tokens = attention.shape
            flat_attn = attention.permute(0,3,1,2).reshape(first_dim, n_tokens, -1)
            soft_attn = F.softmax(flat_attn, dim=-1)
            activation_attention = soft_attn.reshape(first_dim, n_tokens, w, h).permute(0,2,3,1)
    return activation_attention

# def log_attention_loss(attention_map, bounding_box=None, act="sigmoid", aggregate=torch.mean):
#     if bounding_box!=None:
#         x1, y1, x2, y2 = bounding_box.int()
#         attention = attention_map[y1:y2, x1:x2]
#     else:
#         attention = attention_map # 2,1,3
    
#     # TODO: 
#     # add preprocess here
#     # shouldnt we be using logsoftmax?

#     log_attention = torch.log(activation_attention + 1e-8)  # Adding a small epsilon for numerical stability
#     # Convert to float32 before sum calculation to avoid overflow
#     if activation_attention.dtype==torch.float16:
#         log_attention = log_attention.to(torch.float32)
#     loss = -aggregate(log_attention)  # Maximize attention within the bounding box
#     return loss 

# def sigmoid_attention_loss(attention_maps, bounding_boxes):
#     loss = 0
#     for attention_map, bbox in zip(attention_maps, bounding_boxes):
#         x1, y1, x2, y2 = bbox.int()
#         bbox_attention = attention_map[y1:y2, x1:x2]
#         sigmoid_attention = torch.sigmoid(bbox_attention)
#         loss += -torch.sum(sigmoid_attention)  # Maximize attention within the bounding box
#     return loss / len(bounding_boxes)

# def log_attention_loss(attention_maps, bounding_boxes):
#     loss = 0
#     for attention_map, bbox in zip(attention_maps, bounding_boxes):
#         x1, y1, x2, y2 = bbox.int()
#         bbox_attention = attention_map[y1:y2, x1:x2]
#         sigmoid_attention = torch.sigmoid(bbox_attention)
#         log_attention = torch.log(sigmoid_attention + 1e-8)  # Adding a small epsilon for numerical stability
#         loss += -torch.sum(log_attention)  # Maximize attention within the bounding box
#     return loss / len(bounding_boxes)