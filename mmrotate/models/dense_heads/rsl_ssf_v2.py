import torch
from mmcv.cnn import ConvModule, kaiming_init, constant_init
from torch import nn
from einops import rearrange


class SemanticAlignmentFusion_v2(nn.Module):
    def __init__(self, in_channel=256, group_num=3, shuffle=True, ratio=1. / 4.):
        super(SemanticAlignmentFusion_v2, self).__init__()
        self.ch = in_channel
        self.group_num = group_num
        self.is_shuffle = shuffle
        self.ratio = ratio
        self.semantic_align = nn.ModuleList([GlobalContextModule(self.ch * 3, self.ratio, self.group_num)
                                             for _ in range(4)])

    def channel_shuffle(self, group_feat, group_num):
        batch, channels, height, width = group_feat.size()
        group_feat = torch.reshape(group_feat, [batch, group_num, channels // group_num, height, width])
        group_feat = group_feat.permute(0, 2, 1, 3, 4)
        group_feat = torch.reshape(group_feat, [batch, channels, height, width])
        return group_feat

    def forward(self, feats):
        feats_ssf = []
        for i in range(len(feats) - 1):
            # Step1: channel shuffle
            if self.is_shuffle:
                group_feat = self.channel_shuffle(torch.concat(feats[i], 1), self.group_num)  # [B, 3C/g * g, H, W]
            else:
                group_feat = torch.concat(feats[i], 1)
            # Step2: extract global context on each group
            out = self.semantic_align[i](group_feat)
            feats_ssf.append(out)
        feats_ssf.append(feats[-1])
        return feats_ssf


class GlobalContextModule(nn.Module):
    def __init__(self, in_channel, ratio, group):
        super(GlobalContextModule, self).__init__()
        self.ratio = ratio
        self.group = group  # 类似于 multi-head
        self.gch = in_channel // group
        self.hid_ch = int(self.gch * ratio)

        self.fc_qk = nn.Linear(self.gch, 1)
        self.softmax = nn.Softmax(dim=2)

        self.ctx_transform = nn.Sequential(
            nn.Linear(self.gch, self.hid_ch),  # [B, g, gch/r]
            nn.LayerNorm(self.hid_ch),
            nn.ReLU(inplace=True),
            nn.Linear(self.hid_ch, self.gch)  # [B, g, gch]
        )

        self.lvl_weight = nn.Sequential(
            nn.Linear(in_channel, in_channel),
            nn.LayerNorm(in_channel),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel, 3),
            nn.Sigmoid()
        )

        self._init_weight()

    def _init_weight(self):
        kaiming_init(self.fc_qk, mode='fan_in')
        self.fc_qk.inited = True
        constant_init(self.ctx_transform[-1], val=0)
    def forward(self, x):
        out = rearrange(x, 'b (g c) h w -> b g c h w', g=self.group)
        # global context
        value = rearrange(x, 'b (g c) h w -> b g c (h w)', g=self.group)  # [B, g, gch, HW]
        qk = rearrange(x, 'b (g c) h w -> b g (h w) c', g=self.group)
        qk = self.fc_qk(qk)  # [B, g, HW, 1]
        qk = self.softmax(qk)  # norm
        context = torch.matmul(value, qk).squeeze(-1)  # [B, g, gch]
        lvl_context = rearrange(context, 'b g c -> b (g c)')

        # SE transformation
        context = self.ctx_transform(context)

        # level re-weight
        lvl_weight = self.lvl_weight(lvl_context)  # [B, g]
        context = context * lvl_weight.unsqueeze(-1)  # [B, g, gch]

        # add fusion
        context = context.unsqueeze(-1).unsqueeze(-1)  # [B, g, gch, 1, 1]
        out = context + out
        out = torch.sum(out, dim=1)
        return out
