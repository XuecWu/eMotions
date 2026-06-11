import math
import torch
import torch.nn as nn
from models.VideoSwin import VideoSwinTransformer


class VisualStream(nn.Module):
    """
    Constructs the Visual stream in the overall model
    """
    def __init__(self, snippet_duration, sample_size, n_classes, seq_len, pretrained_video_path):
        super(VisualStream, self).__init__()
        self.snippet_duration          = snippet_duration
        self.sample_size               = sample_size
        self.n_classes                 = n_classes
        self.seq_len                   = seq_len
        self.pretrained_video_path     = pretrained_video_path
        self.fc                        = nn.Linear(768, self.n_classes)
        self._init_norm_val()
        self._init_encoder()

    def _init_norm_val(self):
        self.NORM_VALUE = 255.0
        self.MEAN       = 100.0 / self.NORM_VALUE

    def _init_encoder(self):
        last_duration   = int(math.ceil(self.snippet_duration / 16))
        last_size       = int(math.ceil(self.sample_size / 32))
        self.avgpool    = nn.AvgPool3d((last_duration, last_size, last_size), stride=1)
        self.VideoSwinT = VideoSwinTransformer(pretrained_video_path=self.pretrained_video_path)


    def forward(self, input: torch.Tensor):
        input = input.transpose(0, 1).contiguous()
        input.div_(self.NORM_VALUE).sub_(self.MEAN)

        seq_len, batch, nc, snippet_duration, sample_size, _ = input.size()
        input  = input.view(seq_len * batch, nc, snippet_duration, sample_size, sample_size)

        output = self.VideoSwinT(input)
        output = self.avgpool(output)
        output = torch.flatten(output)
        output = torch.mean(output, dim=2)
        output = self.fc(output)
        return output
