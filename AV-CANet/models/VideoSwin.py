\
\
\


import torch
import torch.nn as nn
from .Video_Swin_Transformer import SwinTransformer3D


def VideoSwinTransformer(pretrained_video_path=None, pretrained2d=True):
\
\
\
\
\
\
\
\

    model = SwinTransformer3D(pretrained=pretrained_video_path, pretrained2d=pretrained2d, patch_size=(2,4,4), in_chans=3, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=(8,7,7), mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 patch_norm=True, frozen_stages=-1, use_checkpoint=False)


    if pretrained_video_path:
        model.init_weights(pretrained=pretrained_video_path)

    model = model.cuda()
    return model
