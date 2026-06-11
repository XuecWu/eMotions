\
\
\
\


import torch
import torch.nn as nn
from einops import rearrange, reduce


class CrossAttentionFusion_Layer(nn.Module):
\
\
\

    def __init__(self, input_dim = 512):
        super(CrossAttentionFusion_Layer, self).__init__()
        self.linear_v         = nn.Linear(input_dim, input_dim)
        self.linear_a         = nn.Linear(input_dim, input_dim)

    def forward(self, A, V):
\
\


        attn_output_a, _ = self.attention_module(query=A, key=V, value=V)
        attn_output_v, _ = self.attention_module(query=V, key=A, value=A)


        enhanced_a = self.linear_a(attn_output_a) + A
        enhanced_v = self.linear_v(attn_output_v) + V


        enhanced_a = rearrange(enhanced_a, 's bs c -> bs c s')
        enhanced_v = rearrange(enhanced_v, 's bs c -> bs c s')

        enhanced_a = reduce(enhanced_a, 'bs c s -> bs c', 'mean')
        enhanced_v = reduce(enhanced_v, 'bs c s -> bs c', 'mean')

        return enhanced_a, enhanced_v
