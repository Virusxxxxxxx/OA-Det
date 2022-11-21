import torch
from einops import rearrange
from mmcv.cnn import normal_init, build_norm_layer
from torch import nn, einsum
from mmcv.ops import DeformConv2d
from torch.utils.checkpoint import checkpoint
from functools import partial
from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d
import torch.nn.functional as F


class AddCoords(nn.Module):
    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(
                torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) +
                torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2)
            )
            ret = torch.cat([ret, rr], dim=1)

        return ret


class AlignConvModule(nn.Module):
    """ The module of AlignConv.
    Args:
        in_channels (list): multi level input channels. [768, 1024, 1024, 1024, 768]
        out_channels (int): Number of input channels.
        featmap_strides (list): The strides of featmap.
        align_conv_size (int): The size of align convolution.
    """

    def __init__(self, in_channels, out_channels, featmap_strides, align_conv_size):
        super(AlignConvModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides
        self.align_conv_size = align_conv_size
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the head."""
        self.alignconv = nn.ModuleList([
            AlignConv(
                in_ch,
                self.out_channels,
                kernel_size=self.align_conv_size,
                stride=s)
            for in_ch, s in zip(self.in_channels, self.featmap_strides)
        ])

    def forward(self, x, rbboxes):
        """
        Args:
            x (list[Tensor]): feature maps of multiple scales
            rbboxes (list[list[Tensor]]): best rbboxes of multiple scales of multiple images
        """
        mlvl_rbboxes = [torch.cat(rbbox) for rbbox in zip(*rbboxes)]
        out = []
        for x_scale, rbboxes_scale, ac_scale in zip(x, mlvl_rbboxes, self.alignconv):
            feat_refined_scale = ac_scale(x_scale, rbboxes_scale)
            out.append(feat_refined_scale)
        return out


class AlignConv(nn.Module):
    """
    Args:
        in_channels (int): Number of input channels.
        kernel_size (int, optional): The size of kernel.
        stride (int, optional): Stride of the convolution. Default: None
        deform_groups (int, optional): Number of deformable group partitions.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=None,
                 deform_groups=1):
        super(AlignConv, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.deform_conv = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            deform_groups=deform_groups)
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        """Initialize weights of the head."""
        normal_init(self.deform_conv, std=0.01)

    @torch.no_grad()
    def get_offset(self, anchors, featmap_size, stride):
        """Get the offset of AlignConv."""
        dtype, device = anchors.dtype, anchors.device
        feat_h, feat_w = featmap_size
        pad = (self.kernel_size - 1) // 2
        idx = torch.arange(-pad, pad + 1, dtype=dtype, device=device)
        yy, xx = torch.meshgrid(idx, idx)
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)

        # get sampling locations of default conv 获取原始卷积情况下，特征图上每个点的特征采样位置
        xc = torch.arange(0, feat_w, device=device, dtype=dtype)
        yc = torch.arange(0, feat_h, device=device, dtype=dtype)
        yc, xc = torch.meshgrid(yc, xc)
        xc = xc.reshape(-1)
        yc = yc.reshape(-1)
        x_conv = xc[:, None] + xx  # shape = [H*W, k*k]
        y_conv = yc[:, None] + yy  # shape = [H*W, k*k]

        # get sampling locations of anchors  计算基于 anchor 的采样位置 L
        x_ctr, y_ctr, w, h, a = torch.unbind(anchors, dim=1)  # 解包 anchor
        x_ctr, y_ctr, w, h = \
            x_ctr / stride, y_ctr / stride, \
            w / stride, h / stride
        cos, sin = torch.cos(a), torch.sin(a)
        dw, dh = w / self.kernel_size, h / self.kernel_size
        x, y = dw[:, None] * xx, dh[:, None] * yy
        xr = cos[:, None] * x - sin[:, None] * y
        yr = sin[:, None] * x + cos[:, None] * y
        x_anchor, y_anchor = xr + x_ctr[:, None], yr + y_ctr[:, None]
        # get offset filed
        offset_x = x_anchor - x_conv
        offset_y = y_anchor - y_conv
        # x, y in anchors is opposite in image coordinates,
        # so we stack them with y, x other than x, y
        offset = torch.stack([offset_y, offset_x], dim=-1)
        offset = offset.reshape(anchors.size(0),
                                -1).permute(1, 0).reshape(-1, feat_h, feat_w)
        return offset

    def forward(self, x, anchors):
        """Forward function of AlignConv."""
        anchors = anchors.reshape(x.shape[0], x.shape[2], x.shape[3], 5)
        num_imgs, H, W = anchors.shape[:3]
        offset_list = [
            self.get_offset(anchors[i].reshape(-1, 5), (H, W), self.stride)
            for i in range(num_imgs)
        ]
        offset_tensor = torch.stack(offset_list, dim=0)
        x = self.relu(self.deform_conv(x, offset_tensor.detach()))
        return x


def exists(val):
    return val is not None


def summarize_qkv_chunk(q, k, v, mask, attn_bias_chunk, causal, qk_start_indices):
    q_start_index, k_start_index, q_chunk_size, k_chunk_size, device = *qk_start_indices, q.shape[-2], k.shape[-2], q.device

    weight = einsum('b h i d, b h j d -> b h i j', q, k)

    if exists(attn_bias_chunk):
        weight = weight + attn_bias_chunk

    mask_value = -torch.finfo(weight.dtype).max

    if exists(mask):
        mask = rearrange(mask, 'b j -> b 1 1 j')
        weight = weight.masked_fill(~mask, mask_value)

    if causal and q_start_index < (k_start_index + k_chunk_size - 1):
        causal_mask = torch.ones((q_chunk_size, k_chunk_size), dtype=torch.bool, device=device).triu(
            q_start_index - k_start_index + 1)
        weight = weight.masked_fill(causal_mask, mask_value)

    weight_max = weight.amax(dim=-1, keepdim=True).detach()
    weight = weight - weight_max

    exp_weight = weight.exp()
    weighted_value = einsum('b h i j, b h j d -> b h i d', exp_weight, v)

    return exp_weight.sum(dim=-1), weighted_value, rearrange(weight_max, '... 1 -> ...')


checkpointed_summarize_qkv_chunk = partial(checkpoint, summarize_qkv_chunk)


def memory_efficient_attention(
        q, k, v,
        mask=None,
        causal=False,
        attn_bias=None,
        q_bucket_size=512,
        k_bucket_size=1024,
        eps=1e-8
):
    scale = q.shape[-1] ** -0.5
    q = q * scale

    # function
    needs_backwards = q.requires_grad or k.requires_grad or v.requires_grad
    summarize_qkv_fn = checkpointed_summarize_qkv_chunk if needs_backwards else summarize_qkv_chunk

    # chunk all the inputs
    q_chunks = q.split(q_bucket_size, dim=-2)
    k_chunks = k.split(k_bucket_size, dim=-2)
    v_chunks = v.split(k_bucket_size, dim=-2)
    mask_chunks = mask.split(k_bucket_size, dim=-1) if exists(mask) else ((None,) * len(k_chunks))

    if exists(attn_bias):
        i, j = attn_bias.shape[-2:]
        attn_bias_chunks = attn_bias.split(q_bucket_size, dim=-2)
        attn_bias_chunks = list(map(lambda t: t.split(k_bucket_size, dim=-1), attn_bias_chunks))

    # loop through all chunks and accumulate

    out = []
    for q_index, q_chunk in enumerate(q_chunks):
        exp_weights = []
        weighted_values = []
        weight_maxes = []

        for k_index, (k_chunk, v_chunk, mask_chunk) in enumerate(zip(k_chunks, v_chunks, mask_chunks)):
            q_start_index = q_index * q_bucket_size
            k_start_index = k_index * k_bucket_size

            if causal and k_start_index > (q_start_index + q_chunk.shape[-2] - 1):
                # if chunk is to be all masked out causally, skip
                continue

            attn_bias_chunk = attn_bias_chunks[q_index][k_index] if exists(attn_bias) else None

            exp_weight_chunk, weighted_value_chunk, weight_max_chunk = summarize_qkv_fn(
                q_chunk,
                k_chunk,
                v_chunk,
                mask_chunk,
                attn_bias_chunk,
                causal,
                (q_start_index, k_start_index)
            )

            exp_weights.append(exp_weight_chunk)
            weighted_values.append(weighted_value_chunk)
            weight_maxes.append(weight_max_chunk)

        weight_maxes = torch.stack(weight_maxes, dim=-1)

        weighted_values = torch.stack(weighted_values, dim=-1)
        exp_weights = torch.stack(exp_weights, dim=-1)

        global_max = weight_maxes.amax(dim=-1, keepdim=True)
        renorm_factor = (weight_maxes - global_max).exp().detach()

        exp_weights = exp_weights * renorm_factor
        weighted_values = weighted_values * rearrange(renorm_factor, '... c -> ... 1 c')

        all_values = weighted_values.sum(dim=-1)
        all_weights = exp_weights.sum(dim=-1)

        normalized_values = all_values / (rearrange(all_weights, '... -> ... 1') + eps)
        out.append(normalized_values)

    return torch.cat(out, dim=-2)


class DCNv2_SA(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 stride=1,
                 norm_cfg=dict(type='GN', num_groups=8, requires_grad=True)):
        super(DCNv2_SA, self).__init__()
        self.dcn = ModulatedDeformConv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm = build_norm_layer(norm_cfg, out_channel)[1]
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, offset, mask, size=None):
        if size is not None:
            x = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
        return self.act(self.norm(self.dcn(x.contiguous(), offset.contiguous(), mask)))