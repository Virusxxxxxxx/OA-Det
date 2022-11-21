from torch import nn
from mmrotate.models.dense_heads.rsl_utils import DCNv2_SA


class MultiScaleModule_v2(nn.Module):
    def __init__(self, in_channel=256, simple_fuse=False):
        super(MultiScaleModule_v2, self).__init__()
        self.ch = in_channel
        self.simple_fuse = simple_fuse

        self.p2_down = DCNv2_SA(self.ch, self.ch, stride=2)
        self.p3_down = DCNv2_SA(self.ch, self.ch, stride=2)
        self.p4_up = DCNv2_SA(self.ch, self.ch)
        self.p4_down = DCNv2_SA(self.ch, self.ch, stride=2)
        self.p5_up = DCNv2_SA(self.ch, self.ch)
        self.p5_down = DCNv2_SA(self.ch, self.ch, stride=2)
        self.p6_up = DCNv2_SA(self.ch, self.ch)
        self.p7_up = DCNv2_SA(self.ch, self.ch)

        self.dcn_offset = nn.ModuleList()
        self.dcn_mask = nn.ModuleList()
        for i in range(4):
            self.dcn_offset.append(
                nn.Conv2d(self.ch, 2 * 3 * 3, kernel_size=3, padding=1, bias=True))
            self.dcn_mask.append(
                nn.Conv2d(self.ch, 1 * 3 * 3, kernel_size=3, padding=1, bias=True))

        self._init_weight()

    def _init_weight(self):
        for i in range(4):
            nn.init.constant_(self.dcn_offset[i].weight, 0.)
            nn.init.constant_(self.dcn_offset[i].bias, 0.)
            nn.init.constant_(self.dcn_mask[i].weight, 0.)
            nn.init.constant_(self.dcn_mask[i].bias, 0.)

    def forward(self, x):
        # multi level
        p2, p3, p4, p5, p6, p7 = x

        # Step1: get local offset and mask
        p3_offset = self.dcn_offset[0](p3)  # [B, 2 * 3 * 3, H, W]
        p3_mask = self.dcn_mask[0](p3).sigmoid()  # [B, 1 * 3 * 3, H, W]
        p4_offset = self.dcn_offset[1](p4)
        p4_mask = self.dcn_mask[1](p4).sigmoid()
        p5_offset = self.dcn_offset[2](p5)
        p5_mask = self.dcn_mask[2](p5).sigmoid()
        p6_offset = self.dcn_offset[3](p6)
        p6_mask = self.dcn_mask[3](p6).sigmoid()

        # Step2: get inner multi level feat (coarse align)
        # p3 sub layer
        p2_down = self.p2_down(p2, p3_offset, p3_mask)
        p4_up = self.p4_up(p4, p3_offset, p3_mask, p3.size()[2:])
        # p4 sub layer
        p3_down = self.p3_down(p3, p4_offset, p4_mask)
        p5_up = self.p5_up(p5, p4_offset, p4_mask, p4.size()[2:])
        # p5 sub layer
        p4_down = self.p4_down(p4, p5_offset, p5_mask)
        p6_up = self.p6_up(p6, p5_offset, p5_mask, p5.size()[2:])
        # p6 sub layer
        p5_down = self.p5_down(p5, p6_offset, p6_mask)
        p7_up = self.p7_up(p7, p6_offset, p6_mask, p6.size()[2:])

        if self.simple_fuse:
            p3_ms = p3 + p2_down + p4_up
            p4_ms = p4 + p3_down + p5_up
            p5_ms = p5 + p4_down + p6_up
            p6_ms = p6 + p5_down + p7_up
        else:
            p3_ms = [p3, p2_down, p4_up]
            p4_ms = [p4, p3_down, p5_up]
            p5_ms = [p5, p4_down, p6_up]
            p6_ms = [p6, p5_down, p7_up]

        return p3_ms, p4_ms, p5_ms, p6_ms, p7
