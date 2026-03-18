# Copyright 2024 The HuggingFace Team. All rights reserved.
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

from typing import Any, Dict, Optional

import pdb
import torch
from torch.utils.checkpoint import checkpoint

def basic_transformer_block_forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
        use_grad_chkpt: bool = False,
        use_reentrant: bool = True,
    ):

        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        if self.use_ada_layer_norm:
            if use_grad_chkpt:
                def norm1_forward(hidden_states, timestep):
                    result = self.norm1(hidden_states, 
                                    timestep
                                    )
                    return result

                norm_hidden_states = checkpoint(norm1_forward,
                                    hidden_states,
                                    timestep,
                                    use_reentrant=use_reentrant
                                    )
            else:
                norm_hidden_states = self.norm1(hidden_states, timestep)

        elif self.use_ada_layer_norm_zero:
            if use_grad_chkpt:
                def norm1_forward(hidden_states, timestep, class_labels, hidden_dtype):
                    result = self.norm1(hidden_states, timestep, class_labels, hidden_dtype)
                    return result

                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = checkpoint(norm1_forward,
                                    hidden_states, timestep, class_labels, hidden_states.dtype,
                                    use_reentrant=use_reentrant
                                    )
            else:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
        else:
            if use_grad_chkpt:
                norm_hidden_states = checkpoint(self.norm1, hidden_states, use_reentrant=use_reentrant)
            else:
                norm_hidden_states = self.norm1(hidden_states)

        # 1. Retrieve lora scale.
        lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

        # 2. Prepare GLIGEN inputs
        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

        # Self-Attention
        if use_grad_chkpt:
            
            def attn1_forward(norm_hidden_states, encoder_hidden_states_, attention_mask, cross_attention_kwargs):
                result = self.attn1(
                                        norm_hidden_states,
                                        encoder_hidden_states_,
                                        attention_mask,
                                        **cross_attention_kwargs,
                                        )
                return result

            encoder_hidden_states_=encoder_hidden_states if self.only_cross_attention else None
            attn_output = checkpoint(attn1_forward,
                                norm_hidden_states,
                                encoder_hidden_states_,
                                attention_mask,
                                cross_attention_kwargs,
                                use_reentrant=use_reentrant
                                )

        else:
            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
       

        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = attn_output + hidden_states

        # 2.5 GLIGEN Control
        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])
        # 2.5 ends
        
        # 3. Cross-Attention
        if self.attn2 is not None:

            if self.use_ada_layer_norm:
                if use_grad_chkpt:
                    def norm2_forward(hidden_states, timestep):
                        result = self.norm2(hidden_states, 
                                        timestep
                                        )
                        return result

                    norm_hidden_states = (checkpoint(norm2_forward,
                                        hidden_states,
                                        timestep,
                                        use_reentrant=use_reentrant
                                        ))
                else:
                    norm_hidden_states = (self.norm2(hidden_states, timestep))
            else:
                if use_grad_chkpt:
                    norm_hidden_states = (checkpoint(self.norm2, hidden_states, use_reentrant=use_reentrant))
                else:
                    norm_hidden_states = (self.norm2(hidden_states))
            
            # adding no block-level gradient checkpointing on the cross-attention module - apply linear-level internally
            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        if use_grad_chkpt:
            norm_hidden_states = checkpoint(self.norm3, hidden_states, use_reentrant=use_reentrant)
        else:
            norm_hidden_states = self.norm3(hidden_states)

        if self.use_ada_layer_norm_zero:
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            if norm_hidden_states.shape[self._chunk_dim] % self._chunk_size != 0:
                raise ValueError(
                    f"`hidden_states` dimension to be chunked: {norm_hidden_states.shape[self._chunk_dim]} has to be divisible by chunk size: {self._chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
                )

            num_chunks = norm_hidden_states.shape[self._chunk_dim] // self._chunk_size

            if use_grad_chkpt:
                def ff_forward(hid_slice, scale):
                    result = self.ff(hid_slice, 
                                     scale
                                     )
                    return result

                ff_output = torch.cat(
                    [
                        checkpoint(ff_forward, hid_slice, lora_scale, use_reentrant=use_reentrant)
                        for hid_slice in norm_hidden_states.chunk(num_chunks, dim=self._chunk_dim)
                    ],
                    dim=self._chunk_dim,
                )

            else:
                ff_output = torch.cat(
                    [
                        self.ff(hid_slice, scale=lora_scale)
                        for hid_slice in norm_hidden_states.chunk(num_chunks, dim=self._chunk_dim)
                    ],
                    dim=self._chunk_dim,
                )
        else:
            if use_grad_chkpt:
                def ff_forward(norm_hidden_states, scale):
                    result = self.ff(norm_hidden_states, 
                                     scale
                                     )
                    return result

                ff_output = checkpoint(ff_forward,
                                    norm_hidden_states,
                                    lora_scale,
                                    use_reentrant=use_reentrant
                                    )
            else:
                ff_output = self.ff(norm_hidden_states, scale=lora_scale)

        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = ff_output + hidden_states

        return hidden_states