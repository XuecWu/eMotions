\
\
\
\


import torch
import torch.nn as nn
import torch.nn.functional as F
from models.visual_stream import VisualStream
from .fusion_modules import SumFusion, ConcatFusion, FiLM, GatedFusion
from .audio_net import resnet18, resnet34, resnet50
from .MPAM import MultimodalPyramidAttentionalModule as MMP
from .CrossAttentionFusion_Module import CrossAttentionFusion_Layer
from einops import rearrange, reduce


class AVNet(VisualStream):
\
\
\

    def __init__(self, snippet_duration=16, sample_size=112, n_classes=8, seq_len=10,
                 pretrained_video_path='', audio_embed_size=256, audio_n_segments=16,
                 audio_pretrained=True):
        super(AVNet, self).__init__(snippet_duration=snippet_duration, sample_size=sample_size, n_classes=n_classes, seq_len=seq_len, pretrained_video_path=pretrained_video_path)


        self.fusion_module    = ConcatFusion(output_dim=n_classes)
        self.audio_n_segments = audio_n_segments
        self.audio_embed_size = audio_embed_size
        self.dropout          = nn.Dropout(p=0.5)


        self.a_resnet                   = resnet34(pretrained=audio_pretrained)
        self.CrossAttentionFusion_Layer = CrossAttentionFusion_Layer(input_dim=512)


        self.a_fc             = nn.Sequential(
            nn.Linear(512, self.audio_embed_size),
            nn.BatchNorm1d(self.audio_embed_size),
            nn.Tanh()
        )
        self.v_fc             = nn.Linear(768, 512)

    def forward(self, visual: torch.Tensor, audio: torch.Tensor):
\
\

        visual = visual.transpose(0, 1).contiguous()
        visual.div_(self.NORM_VALUE).sub_(self.MEAN)


        seq_len, batch, nc, snippet_duration, sample_size, _ = visual.size()
        visual = visual.view(seq_len * batch, nc, snippet_duration, sample_size, sample_size).contiguous()
        output = self.VideoSwinT(visual)
        output = self.avgpool(output)


        output = self.dropout(output)

        output = torch.flatten(output, start_dim=2)
        output = torch.mean(output, dim=2)
        output = output.view(seq_len, batch, 768)
        output = output.permute(1, 0, 2).contiguous()
        output = self.v_fc(output)


        bs    = audio.size(0)
        audio = audio.transpose(0, 1).contiguous()
        audio = audio.chunk(self.audio_n_segments, dim=0)


        audio = torch.stack(audio, dim=0).contiguous()
        audio = audio.transpose(1, 2).contiguous()
        audio = torch.flatten(audio, start_dim=0, end_dim=1)


        audio = torch.unsqueeze(audio, dim=1)
        audio = self.a_resnet(audio)
        audio = torch.flatten(audio, start_dim=1).contiguous()
        audio = self.a_fc(audio)
        audio = audio.view(self.audio_n_segments, bs, self.audio_embed_size).contiguous()
        audio = audio.permute(1, 0, 2).contiguous()


        output, audio   = self.MMP_Module(output, audio)

        output          = rearrange(output, 'bs s c -> s bs c')
        audio           = rearrange(audio, 'bs s c -> s bs c')

        audio, output   = self.CrossAttentionFusion_Layer(audio, output)
        a, v, out       = self.fusion_module(audio, output)

        return a, v, out
