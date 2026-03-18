# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# NOTICE: This file is modified from diffuser source (version diffusers-0.21.4)

import pdb
import torch
from torch import nn

from diffusers.models.resnet import Downsample2D, ResnetBlock2D, TemporalConvLayer, Upsample2D
from diffusers.models.transformer_2d import Transformer2DModel
from diffusers.models.transformer_temporal import TransformerTemporalModel

from torch.utils.checkpoint import checkpoint


def unet_mid_block3D_cross_attn_forward(
    self,
    hidden_states,
    temb=None,
    encoder_hidden_states=None,
    attention_mask=None,
    num_frames=1,
    cross_attention_kwargs=None,
    use_grad_chkpt=False,
    use_reentrant=True,
):
   
    if use_grad_chkpt:
        hidden_states = checkpoint(self.resnets[0], hidden_states, temb, use_reentrant=use_reentrant)
        hidden_states = checkpoint(self.temp_convs[0], hidden_states, num_frames, use_reentrant=use_reentrant)
    else:
        hidden_states = self.resnets[0](hidden_states, temb)
        hidden_states = self.temp_convs[0](hidden_states, num_frames=num_frames)

    for attn, temp_attn, resnet, temp_conv in zip(
        self.attentions, self.temp_attentions, self.resnets[1:], self.temp_convs[1:]
    ):
        
        hidden_states = attn(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            cross_attention_kwargs=cross_attention_kwargs,
            return_dict=False,
            use_grad_chkpt=use_grad_chkpt,
            use_reentrant=use_reentrant,
        )[0]

        hidden_states = temp_attn(
            hidden_states, num_frames=num_frames, cross_attention_kwargs=cross_attention_kwargs, return_dict=False,
            use_grad_chkpt=use_grad_chkpt,
            use_reentrant=use_reentrant,
        )[0]

        if use_grad_chkpt:
            hidden_states = checkpoint(resnet, hidden_states, temb, use_reentrant=use_reentrant)
            hidden_states = checkpoint(temp_conv, hidden_states, num_frames, use_reentrant=use_reentrant)
        else:
            hidden_states = resnet(hidden_states, temb)
            hidden_states = temp_conv(hidden_states, num_frames=num_frames)

    return hidden_states


def cross_attn_downblock3D_forward(
    self,
    hidden_states,
    temb=None,
    encoder_hidden_states=None,
    attention_mask=None,
    num_frames=1,
    cross_attention_kwargs=None,
    use_grad_chkpt=False,
    use_reentrant=True,
):  
    
    # TODO(Patrick, William) - attention mask is not used
    output_states = ()

    for resnet, temp_conv, attn, temp_attn in zip(
        self.resnets, self.temp_convs, self.attentions, self.temp_attentions
    ):
        if use_grad_chkpt:
            # note: most args here have been pre-defined, hence no need for partial or extra function definition
            hidden_states = checkpoint(resnet, hidden_states, temb, use_reentrant=use_reentrant)
            hidden_states = checkpoint(temp_conv, hidden_states, num_frames, use_reentrant=use_reentrant)
        else:
            hidden_states = resnet(hidden_states, temb)
            hidden_states = temp_conv(hidden_states, num_frames=num_frames)

        hidden_states = attn(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            cross_attention_kwargs=cross_attention_kwargs,
            return_dict=False,
            use_grad_chkpt=use_grad_chkpt,
            use_reentrant=use_reentrant,
        )[0]
        
        
        hidden_states = temp_attn(
            hidden_states, num_frames=num_frames, cross_attention_kwargs=cross_attention_kwargs, return_dict=False,
            use_grad_chkpt=use_grad_chkpt,
            use_reentrant=use_reentrant
        )[0]

        output_states += (hidden_states,)

    if self.downsamplers is not None:
        # TODO: checkpoint downsampler
        for downsampler in self.downsamplers:
            if use_grad_chkpt:
                hidden_states = checkpoint(downsampler, hidden_states, use_reentrant=use_reentrant)
            else:
                hidden_states = downsampler(hidden_states)

        output_states += (hidden_states,)

    return hidden_states, output_states


def downblock3D_forward(self, hidden_states, temb=None, num_frames=1,
                        use_grad_chkpt=False,
                        use_reentrant=True):
    output_states = ()

    for resnet, temp_conv in zip(self.resnets, self.temp_convs):

        if use_grad_chkpt:
            hidden_states = checkpoint(resnet, hidden_states, temb, use_reentrant=use_reentrant)
            hidden_states = checkpoint(temp_conv, hidden_states, num_frames, use_reentrant=use_reentrant)
        else:
            hidden_states = resnet(hidden_states, temb)
            hidden_states = temp_conv(hidden_states, num_frames=num_frames)

        output_states += (hidden_states,)

    if self.downsamplers is not None:
        for downsampler in self.downsamplers:
            if use_grad_chkpt:
                hidden_states = checkpoint(downsampler, hidden_states, use_reentrant=use_reentrant)
            else:
                hidden_states = downsampler(hidden_states)

        output_states += (hidden_states,)

    return hidden_states, output_states


def cross_attn_upblock3D_forward(
    self,
    hidden_states,
    res_hidden_states_tuple,
    temb=None,
    encoder_hidden_states=None,
    upsample_size=None,
    attention_mask=None,
    num_frames=1,
    cross_attention_kwargs=None,
    use_grad_chkpt=False,
    use_reentrant=True,
):  

    # TODO(Patrick, William) - attention mask is not used
    for resnet, temp_conv, attn, temp_attn in zip(
        self.resnets, self.temp_convs, self.attentions, self.temp_attentions
    ):
        # pop res hidden states
        res_hidden_states = res_hidden_states_tuple[-1]
        res_hidden_states_tuple = res_hidden_states_tuple[:-1]
        hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

        if use_grad_chkpt:
            hidden_states = checkpoint(resnet, hidden_states, temb, use_reentrant=use_reentrant)
            hidden_states = checkpoint(temp_conv, hidden_states, num_frames, use_reentrant=use_reentrant)
        else:
            hidden_states = resnet(hidden_states, temb)
            hidden_states = temp_conv(hidden_states, num_frames=num_frames)
        
        hidden_states = attn(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            cross_attention_kwargs=cross_attention_kwargs,
            return_dict=False,
            use_grad_chkpt=use_grad_chkpt,
            use_reentrant=use_reentrant
        )[0]

        hidden_states = temp_attn(
            hidden_states, num_frames=num_frames, cross_attention_kwargs=cross_attention_kwargs, return_dict=False,
            use_grad_chkpt=use_grad_chkpt,
            use_reentrant=use_reentrant
        )[0]

    if self.upsamplers is not None:
        for upsampler in self.upsamplers:
            if use_grad_chkpt:
                hidden_states = checkpoint(upsampler, hidden_states, upsample_size, use_reentrant=use_reentrant)
            else:
                hidden_states = upsampler(hidden_states, upsample_size)

    return hidden_states


def upblock3D_forward(self, hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None, num_frames=1,
                    use_grad_chkpt=False,
                    use_reentrant=True):

    for resnet, temp_conv in zip(self.resnets, self.temp_convs):
        # pop res hidden states
        res_hidden_states = res_hidden_states_tuple[-1]
        res_hidden_states_tuple = res_hidden_states_tuple[:-1]
        hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

        if use_grad_chkpt:
            hidden_states = checkpoint(resnet, hidden_states, temb, use_reentrant=use_reentrant)
            hidden_states = checkpoint(temp_conv, hidden_states, num_frames, use_reentrant=use_reentrant)
        else:
            hidden_states = resnet(hidden_states, temb)
            hidden_states = temp_conv(hidden_states, num_frames=num_frames)

    if self.upsamplers is not None:
        for upsampler in self.upsamplers:
            if use_grad_chkpt:
                hidden_states = checkpoint(upsampler, hidden_states, upsample_size, use_reentrant=use_reentrant)
            else:
                hidden_states = upsampler(hidden_states, upsample_size)

    return hidden_states