import torch
from mmcv.cnn import ConvModule, build_norm_layer, Scale
from mmcv.ops import ModulatedDeformConv2d
from .rsl_utils import DCNv2_SA
from torch import nn


class SpatialAlignmentFusion_v6(nn.Module):
    def __init__(self,
                 in_channel=256,
                 conv_cfg=dict(
                     kernel_size=[3, 3, 3],
                     stride=[1, 1, 1],
                     padding=[1, 2, 5],
                     dilation=[1, 2, 5]),
                 is_sa_block=True,
                 simple_fuse=False):
        super(SpatialAlignmentFusion_v6, self).__init__()
        self.ch = in_channel
        self.simple_fuse = simple_fuse
        self.is_sa_block = is_sa_block

        self.kernel_size = conv_cfg['kernel_size'][0]
        self.stage_num = len(conv_cfg['kernel_size'])
        self.conv_cfg = []
        for i in range(self.stage_num):
            self.conv_cfg.append(dict(
                kernel_size=conv_cfg['kernel_size'][i],
                stride=conv_cfg['stride'][i],
                padding=conv_cfg['padding'][i],
                dilation=conv_cfg['dilation'][i],
            ))

        self.sa_block = nn.ModuleList()
        self.axisconv_angle = nn.ModuleList([
            nn.Conv2d(self.ch, 1, kernel_size=3, padding=1, bias=True)  # 同stage下，angle 层内层间统一，stage 越深 angle 越 fine
            for _ in range(self.stage_num)
        ])
        self.axisconv_mask = nn.ModuleList()
        self.axisconv_down = nn.ModuleList()
        self.axisconv_up = nn.ModuleList()
        for i in range(4):
            for stage in range(self.stage_num):
                if self.is_sa_block:
                    self.sa_block.append(SpatialAwareBlock(self.ch, self.ch))
                else:
                    self.sa_block.append(ConvModule(self.ch, self.ch, kernel_size=3, padding=1, norm_cfg=dict(type='BN')))
                self.axisconv_mask.append(  # 3 层合并预测一个 mask
                    nn.Conv2d(self.ch, self.kernel_size * self.kernel_size, kernel_size=3, padding=1, bias=True))
                self.axisconv_down.append(AxisAlignConv(self.ch, self.ch, **self.conv_cfg[stage]))
                self.axisconv_up.append(AxisAlignConv(self.ch, self.ch, **self.conv_cfg[stage]))
        self._init_weight()

    def _init_weight(self):
        for i in range(self.stage_num):
            nn.init.constant_(self.axisconv_angle[i].weight, 0.)
            nn.init.constant_(self.axisconv_angle[i].bias, 0.)
        for i in range(4 * self.stage_num):
            nn.init.constant_(self.axisconv_mask[i].weight, 0.)
            nn.init.constant_(self.axisconv_mask[i].bias, 0.)

    def forward(self, feats):
        feats_sa = []
        for i in range(len(feats) - 1):
            x, x_down, x_up = feats[i]  # coarse align features
            x_full = x + x_down + x_up  # add / concat
            for stage in range(self.stage_num):
                index = i * self.stage_num + stage
                x_full = self.sa_block[index](x_full)  # spatial aware

                angle = self.axisconv_angle[stage](x_full)  # predict angle for each sampling location
                mask = self.axisconv_mask[index](x_full).sigmoid()  # predict mask sigmoid
                x_down = self.axisconv_down[index](x_down, angle, mask)  # fine align features
                x_up = self.axisconv_up[index](x_up, angle, mask)  # share offset & mask
            if self.simple_fuse:
                feats_sa.append(x + x_down + x_up)
            else:
                feats_sa.append([x, x_down, x_up])
        feats_sa.append(feats[-1])  # p7
        return feats_sa


class SpatialAwareBlock(nn.Module):
    """
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, in_channel, out_channel):
        super(SpatialAwareBlock, self).__init__()
        self.in_ch = in_channel
        self.out_ch = out_channel
        self.dwconv = nn.Conv2d(self.in_ch, self.in_ch, kernel_size=7, padding=3, groups=self.in_ch)
        self.norm = nn.LayerNorm(self.in_ch)
        self.pwconv1 = nn.Linear(self.in_ch, 4 * self.out_ch)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * self.out_ch, self.out_ch)

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)  # point-wise
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = residual + x
        return x


class AxisAlignConv(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size=3, stride=1, padding=1, dilation=1, bias=False,
                 norm_cfg=dict(type='GN', num_groups=8, requires_grad=True)):
        super(AxisAlignConv, self).__init__()
        self.in_ch = in_channel
        self.out_ch = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        self.dcn = ModulatedDeformConv2d(self.in_ch, self.out_ch,
                                         self.kernel_size, self.stride,
                                         self.padding, self.dilation, bias=self.bias)
        self.norm = build_norm_layer(norm_cfg, out_channel)[1]
        self.act = nn.ReLU(inplace=True)

    def gen_offset(self, b, h, w, angle):
        ks = self.dilation * (self.kernel_size - 1) + 1
        x_v = (ks - 1) // 2
        y_v = (ks - 1) // 2
        x_axis = torch.arange(-x_v, x_v + 1, self.dilation)
        y_axis = torch.arange(-y_v, y_v + 1, self.dilation)
        x_coor, y_coor = torch.meshgrid(x_axis, y_axis)
        x_coor = x_coor.float().contiguous().view(-1, 1)
        y_coor = y_coor.float().contiguous().view(-1, 1)
        coor = torch.cat((x_coor, y_coor), dim=1).unsqueeze(2).cuda()
        oH = (h + 2 * self.padding - ks) // self.stride + 1
        oW = (w + 2 * self.padding - ks) // self.stride + 1
        sH = ks // 2 - self.padding
        sW = ks // 2 - self.padding
        angle = angle[:, :, sH:sH + oH, sW:sW + oW]
        cos_theta = torch.cos(angle).unsqueeze(-1)
        sin_theta = torch.sin(angle).unsqueeze(-1)
        rot_theta = torch.cat((cos_theta - 1, sin_theta, -sin_theta, cos_theta - 1), dim=-1)
        rot_theta = rot_theta.contiguous().view(-1, 1, 2, 2)
        offset = torch.matmul(rot_theta, coor)
        offset = offset.reshape(b, oH, oW, -1)
        offset = offset.permute(0, 3, 1, 2).contiguous()
        return offset

    def forward(self, x, angle, mask):
        b, c, h, w = x.size()
        offset = self.gen_offset(b, h, w, angle)
        return self.act(self.norm(self.dcn(x, offset, mask)))
