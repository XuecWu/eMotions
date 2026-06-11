\
\
\
\


import torch
import torch.nn as nn
import copy
from models.Transformer import (IntegrateAttentionBlock, TransformerLayer, MultiHeadAttention, PositionwiseFeedForward)


class TemporalBlock(nn.Module):
\
\

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation):
        super(TemporalBlock, self).__init__()


        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=dilation, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, 1)

        self.net        = nn.Sequential(self.conv1, self.relu1, self.conv2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu       = nn.ReLU()


    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class SemanticCaptureModule(nn.Module):
\
\
\

    def __init__(self, num_inputs, ffn_dim, num_channels, kernel_size=3, dropout=0.2, nhead=8):
        super(SemanticCaptureModule, self).__init__()


        v_layers   = []
        a_layers   = []
        msa_layers = []


        num_levels = len(num_channels)
        c          = copy.deepcopy


        self.multiheadattn = MultiHeadAttention(nhead, num_inputs)
        self.feedforward   = PositionwiseFeedForward(num_inputs, ffn_dim, dropout=dropout)


        for i in range(num_levels):


            dilation_size = 2 ** i
            in_channels   = num_inputs if i == 0 else num_channels[i - 1]
            out_channels  = num_channels[i]


            v_layers     += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size)]
            a_layers     += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size)]


            msa_layers   += [IntegrateAttentionBlock(TransformerLayer(num_inputs, MultiHeadAttention(nhead, num_inputs, masksize=dilation_size * 2), c(self.feedforward), dropout),
                                                   TransformerLayer(num_inputs, MultiHeadAttention(nhead, num_inputs, masksize=dilation_size * 2), c(self.feedforward), dropout),
                                                   TransformerLayer(num_inputs, MultiHeadAttention(nhead, num_inputs, masksize=dilation_size * 2), c(self.feedforward), dropout),
                                                   num_inputs)]


        self.vnetwork = nn.Sequential(*v_layers)
        self.anetwork = nn.Sequential(*a_layers)
        self.msa      = nn.Sequential(*msa_layers)

    def forward(self, v, a):
\
\

        v_stage_list = []
        a_stage_list = []

        v_stage = v
        a_stage = a


        for i in range(len(self.vnetwork)):


            v_msa, a_msa = self.msa[i](v_stage, a_stage)
            v_msa = v_msa.permute(0, 2, 1).contiguous()
            a_msa = a_msa.permute(0, 2, 1).contiguous()


            v_tcn = self.vnetwork[i](v_msa)
            a_tcn = self.anetwork[i](a_msa)


            v_stage = v_tcn.permute(0, 2, 1).contiguous()
            a_stage = a_tcn.permute(0, 2, 1).contiguous()


            v_stage_list.append(v_stage)
            a_stage_list.append(a_stage)


        v_stage = torch.stack(v_stage_list, dim=2)
        a_stage = torch.stack(a_stage_list, dim=2)


        v_stage = v_stage.view(-1, v_stage.size(2), v_stage.size(3))
        a_stage = a_stage.view(-1, a_stage.size(2), a_stage.size(3))

        return v_stage, a_stage


class SemanticFusionModule(nn.Module):
\
\

    def __init__(self, num_inputs):
        super(SemanticFusionModule, self).__init__()
        c                         = copy.deepcopy
        self.num_inputs           = num_inputs


        self.vselectfusion        = nn.Linear(num_inputs, 1)
        self.aselectfusion        = nn.Linear(num_inputs, 1)


        self.multiheadattn_weight        = MultiHeadAttention(1, 1)
        self.feedforward_weight          = PositionwiseFeedForward(1, 1, dropout=0)

        self.weight_interaction          = TransformerLayer(1, c(self.multiheadattn_weight), c(self.feedforward_weight), 0)

    def forward(self, v_stage, a_stage):
\
\


        v_weight   = torch.sigmoid(self.vselectfusion(v_stage))
        a_weight   = torch.sigmoid(self.aselectfusion(a_stage))


        enhanced_v_weight  = self.weight_interaction(v_weight, a_weight, a_weight)
        enhanced_a_weight  = self.weight_interaction(a_weight, v_weight, v_weight)

        v_interact = v_stage.permute(0, 2, 1).contiguous()
        a_interact = a_stage.permute(0, 2, 1).contiguous()


        v_out      = torch.bmm(v_interact, enhanced_v_weight).view(-1, 16, self.num_inputs)
        a_out      = torch.bmm(a_interact, enhanced_a_weight).view(-1, 16, self.num_inputs)

        return v_out, a_out


class MultimodalPyramidAttentionalModule(nn.Module):
\
\

    def __init__(self, num_inputs, ffn_dim, num_channels, kernel_size=3, dropout=0.5, nhead=8):
        super(MultimodalPyramidAttentionalModule, self).__init__()
        self.capture = SemanticCaptureModule(num_inputs, ffn_dim, num_channels, kernel_size, dropout, nhead)
        self.fusion  = SemanticFusionModule(num_inputs)

    def forward(self, video, audio):
\
\


        v_cap, a_cap = self.capture(video, audio)
        v_out, a_out = self.fusion(v_cap, a_cap)

        return v_out, a_out
