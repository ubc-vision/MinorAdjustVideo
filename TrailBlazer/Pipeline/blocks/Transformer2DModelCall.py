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

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pdb
import torch
import torch.nn.functional as F

from diffusers.utils import BaseOutput
from torch.utils.checkpoint import checkpoint

@dataclass
class Transformer2DModelOutput(BaseOutput):
    """
    The output of [`Transformer2DModel`].

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` or `(batch size, num_vector_embeds - 1, num_latent_pixels)` if [`Transformer2DModel`] is discrete):
            The hidden states output conditioned on the `encoder_hidden_states` input. If discrete, returns probability
            distributions for the unnoised latent pixels.
    """

    sample: torch.FloatTensor

def transformer2d_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        use_grad_chkpt=False,
        use_reentrant=True,
    ):
        """
        The [`Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.LongTensor` of shape `(batch size, num latent pixels)` if discrete, `torch.FloatTensor` of shape `(batch size, channel, height, width)` if continuous):
                Input `hidden_states`.
            encoder_hidden_states ( `torch.FloatTensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `torch.LongTensor`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            class_labels ( `torch.LongTensor` of shape `(batch size, num classes)`, *optional*):
                Used to indicate class labels conditioning. Optional class labels to be applied as an embedding in
                `AdaLayerZeroNorm`.
            encoder_attention_mask ( `torch.Tensor`, *optional*):
                Cross-attention mask applied to `encoder_hidden_states`. Two formats supported:

                    * Mask `(batch, sequence_length)` True = keep, False = discard.
                    * Bias `(batch, 1, sequence_length)` 0 = keep, -10000 = discard.

                If `ndim == 2`: will be interpreted as a mask, then converted into a bias consistent with the format
                above. This bias will be added to the cross-attention scores.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None and attention_mask.ndim == 2:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # Retrieve lora scale.
        lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

        # 1. Input
        if self.is_input_continuous:
            batch, _, height, width = hidden_states.shape
            residual = hidden_states

            if use_grad_chkpt:
                hidden_states = checkpoint(self.norm, hidden_states, use_reentrant=use_reentrant)
            else:
                hidden_states = self.norm(hidden_states)

            if use_grad_chkpt:
                def proj_in_forward(hidden_states, scale):
                    result = self.proj_in(hidden_states, scale)
                    return result
            
            if not self.use_linear_projection: 
                if use_grad_chkpt:    
                    hidden_states = checkpoint(proj_in_forward, hidden_states, lora_scale, use_reentrant=use_reentrant)
                else:
                    hidden_states = self.proj_in(hidden_states, lora_scale)

                inner_dim = hidden_states.shape[1]
                hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
            else:
                inner_dim = hidden_states.shape[1]
                hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
                if use_grad_chkpt:
                    hidden_states = checkpoint(proj_in_forward, hidden_states, lora_scale, use_reentrant=use_reentrant)
                else:
                    hidden_states = self.proj_in(hidden_states, scale=lora_scale)

        elif self.is_input_vectorized:
            if use_grad_chkpt:
                hidden_states = checkpoint(self.latent_image_embedding, hidden_states, use_reentrant=use_reentrant)
            else:
                hidden_states = self.latent_image_embedding(hidden_states)
        elif self.is_input_patches:
            if use_grad_chkpt:
                hidden_states = checkpoint(self.pos_embed, hidden_states, use_reentrant=use_reentrant)
            else:
                hidden_states = self.pos_embed(hidden_states)

        # 2. Blocks
        for block in self.transformer_blocks:
            if self.training and self.gradient_checkpointing:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    block,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    cross_attention_kwargs,
                    class_labels,
                    use_reentrant=False,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                    use_grad_chkpt=use_grad_chkpt,
                    use_reentrant=use_reentrant
                )

        # 3. Output
        if self.is_input_continuous:
            if use_grad_chkpt:
                def proj_out_forward(hidden_states, scale):
                    result = self.proj_out(hidden_states, scale)
                    return result
                
            if not self.use_linear_projection:
                hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
                if use_grad_chkpt:
                    hidden_states = checkpoint(proj_out_forward, hidden_states, lora_scale, use_reentrant=use_reentrant)
                else:
                    hidden_states = self.proj_out(hidden_states, scale=lora_scale)
            else:
                if use_grad_chkpt:
                    hidden_states = checkpoint(proj_out_forward, hidden_states, lora_scale, use_reentrant=use_reentrant)
                else:
                    hidden_states = self.proj_out(hidden_states, scale=lora_scale)
                hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()

            output = hidden_states + residual

        elif self.is_input_vectorized:
            if use_grad_chkpt:
                hidden_states = checkpoint(self.norm_out, hidden_states, use_reentrant=use_reentrant)
                logits = checkpoint(self.out, hidden_states, use_reentrant=use_reentrant)
            else:
                hidden_states = self.norm_out(hidden_states)
                logits = self.out(hidden_states)
            # (batch, self.num_vector_embeds - 1, self.num_latent_pixels)
            logits = logits.permute(0, 2, 1)

            # log(p(x_0))
            output = F.log_softmax(logits.double(), dim=1).float()

        elif self.is_input_patches:
            # TODO: cleanup!
            if use_grad_chkpt:
                def transformer_blocks_0_norm1_emb_forward(timestep, class_labels, hidden_dtype):
                    result = self.transformer_blocks[0].norm1.emb(timestep, class_labels, hidden_dtype)
                    return result
                conditioning = checkpoint(transformer_blocks_0_norm1_emb_forward,
                                    timestep, class_labels, hidden_states.dtype
                                    )
                shift, scale = checkpoint(self.proj_out_1, F.silu(conditioning), use_reentrant=use_reentrant).chunk(2, dim=1)
                hidden_states = checkpoint(self.norm_out, hidden_states, use_reentrant=use_reentrant) * (1 + scale[:, None]) + shift[:, None]
                hidden_states = checkpoint(self.proj_out_2, hidden_states, use_reentrant=use_reentrant)
            else:
                conditioning = self.transformer_blocks[0].norm1.emb(
                    timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
                shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
                hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
                hidden_states = self.proj_out_2(hidden_states)

            # unpatchify
            height = width = int(hidden_states.shape[1] ** 0.5)
            hidden_states = hidden_states.reshape(
                shape=(-1, height, width, self.patch_size, self.patch_size, self.out_channels)
            )
            hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
            output = hidden_states.reshape(
                shape=(-1, self.out_channels, height * self.patch_size, width * self.patch_size)
            )

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)



