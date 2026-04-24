
import math
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

from fvcore.nn.weight_init import c2_msra_fill, c2_xavier_fill

from detectron2.utils.registry import Registry
from detectron2.layers import Conv2d

SPARSE_INST_DECODER_REGISTRY = Registry("SPARSE_INST_DECODER")
SPARSE_INST_DECODER_REGISTRY.__doc__ = "registry for SparseInst decoder"


def _make_stack_3x3_convs(num_convs, in_channels, out_channels):
    convs = []
    for _ in range(num_convs):
        convs.append(
            Conv2d(in_channels, out_channels, 3, padding=1))
        convs.append(nn.ReLU(True))
        in_channels = out_channels
    return nn.Sequential(*convs)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = Conv2d(in_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU(True)
        self.conv2 = Conv2d(out_channels, out_channels, 3, padding=1)
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        x += residual
        return self.relu(x)
#
# def _make_stack_3x3_convs(num_blocks, in_channels, out_channels):
#     blocks = [BasicBlock(in_channels, out_channels)]
#     for _ in range(1, num_blocks):
#         blocks.append(BasicBlock(out_channels, out_channels))
#     return nn.Sequential(*blocks)

class BiFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.num_levels = len(in_channels_list)
        self.fpn_convs = nn.ModuleList()
        for in_ch in in_channels_list:
            self.fpn_convs.append(Conv2d(in_ch, out_channels, 1))

    def forward(self, features):
        assert len(features) == self.num_levels

        # Step 1: 1x1 conv to align channels
        aligned_features = [conv(feat) for conv, feat in zip(self.fpn_convs, features)]

        # Step 2: Top-down path
        top_down = [aligned_features[-1]]
        for i in reversed(range(self.num_levels - 1)):
            upsampled = F.interpolate(
                top_down[-1],
                size=aligned_features[i].shape[2:],  # 确保上采样到目标尺寸
                mode='bilinear',
                align_corners=False
            )
            top_down.append(upsampled + aligned_features[i])

        # Step 3: Bottom-up path
        bottom_up = [top_down[0]]
        for i in range(1, self.num_levels):
            # 修改：使用 interpolate 代替 avg_pool，确保尺寸匹配
            downsampled = F.interpolate(
                bottom_up[-1],
                size=top_down[i].shape[2:],  # 确保下采样到目标尺寸
                mode='bilinear',
                align_corners=False
            )
            bottom_up.append(downsampled + top_down[i])

        # Step 4: Final output is the top-down path
        return top_down[::-1]

class InstanceBranch(nn.Module):

    def __init__(self, cfg, in_channels):
        super().__init__()
        # norm = cfg.MODEL.SPARSE_INST.DECODER.NORM
        dim = cfg.MODEL.SPARSE_INST.DECODER.INST.DIM
        num_convs = cfg.MODEL.SPARSE_INST.DECODER.INST.CONVS
        num_masks = cfg.MODEL.SPARSE_INST.DECODER.NUM_MASKS
        kernel_dim = cfg.MODEL.SPARSE_INST.DECODER.KERNEL_DIM
        self.num_classes = cfg.MODEL.SPARSE_INST.DECODER.NUM_CLASSES

        self.inst_convs = _make_stack_3x3_convs(num_convs, in_channels, dim)
        # iam prediction, a simple conv
        self.iam_conv = nn.Conv2d(dim, num_masks, 3, padding=1)

        # outputs
        self.cls_score = nn.Linear(dim, self.num_classes)
        self.mask_kernel = nn.Linear(dim, kernel_dim)
        self.objectness = nn.Linear(dim, 1)

        self.prior_prob = 0.01
        self._init_weights()

    def _init_weights(self):
        for m in self.inst_convs.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        for module in [self.iam_conv, self.cls_score]:
            init.constant_(module.bias, bias_value)
        init.normal_(self.iam_conv.weight, std=0.01)
        init.normal_(self.cls_score.weight, std=0.01)

        init.normal_(self.mask_kernel.weight, std=0.01)
        init.constant_(self.mask_kernel.bias, 0.0)

    def forward(self, features):
        # instance features (x4 convs)
        features = self.inst_convs(features)
        # predict instance activation maps
        iam = self.iam_conv(features)
        iam_prob = iam.sigmoid()

        B, N = iam_prob.shape[:2]
        C = features.size(1)
        # BxNxHxW -> BxNx(HW)
        iam_prob = iam_prob.view(B, N, -1)
        # aggregate features: BxCxHxW -> Bx(HW)xC
        inst_features = torch.bmm(iam_prob, features.view(B, C, -1).permute(0, 2, 1))
        normalizer = iam_prob.sum(-1).clamp(min=1e-6)
        inst_features = inst_features / normalizer[:, :, None]
        # predict classification & segmentation kernel & objectness
        pred_logits = self.cls_score(inst_features)
        pred_kernel = self.mask_kernel(inst_features)
        pred_scores = self.objectness(inst_features)
        return pred_logits, pred_kernel, pred_scores, iam

class MaskBranch(nn.Module):

    def __init__(self, cfg, in_channels):
        super().__init__()
        dim = cfg.MODEL.SPARSE_INST.DECODER.MASK.DIM
        num_convs = cfg.MODEL.SPARSE_INST.DECODER.MASK.CONVS
        kernel_dim = cfg.MODEL.SPARSE_INST.DECODER.KERNEL_DIM
        self.mask_convs = _make_stack_3x3_convs(num_convs, in_channels, dim)
        self.projection = nn.Conv2d(dim, kernel_dim, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.mask_convs.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)
        c2_msra_fill(self.projection)

    def forward(self, features):
        # mask features (x4 convs)
        features = self.mask_convs(features)
        return self.projection(features)


@SPARSE_INST_DECODER_REGISTRY.register()
class BaseIAMDecoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        # add 2 for coordinates
        self.base_channels = cfg.MODEL.SPARSE_INST.ENCODER.NUM_CHANNELS
        in_channels = self.base_channels + 2

        self.scale_factor = cfg.MODEL.SPARSE_INST.DECODER.SCALE_FACTOR
        self.output_iam = cfg.MODEL.SPARSE_INST.DECODER.OUTPUT_IAM

        self.inst_branch = InstanceBranch(cfg, in_channels)
        self.mask_branch = MaskBranch(cfg, in_channels)

        # BiFPN for multi-scale feature fusion
        # 创建多尺度特征通道列表
        self.bifpn_levels = cfg.MODEL.SPARSE_INST.DECODER.get("BIFPN_LEVELS", 3)
        self.bifpn = BiFPN([self.base_channels] * self.bifpn_levels, self.base_channels)

    @torch.no_grad()
    def compute_coordinates(self, x):
        h, w = x.size(2), x.size(3)
        y_loc = torch.linspace(-1, 1, h, device=x.device)
        x_loc = torch.linspace(-1, 1, w, device=x.device)
        y_loc, x_loc = torch.meshgrid(y_loc, x_loc)
        y_loc = y_loc.expand([x.shape[0], 1, -1, -1])
        x_loc = x_loc.expand([x.shape[0], 1, -1, -1])
        locations = torch.cat([x_loc, y_loc], 1)
        return locations.to(x)

    def forward(self, features):
        # 创建多尺度特征列表（通过不同步长的池化）
        multi_scale_features = []
        for i in range(self.bifpn_levels):
            if i == 0:
                # 原始特征
                multi_scale_features.append(features)
            else:
                # 下采样特征
                scale_factor = 2 ** i
                downsampled = F.avg_pool2d(
                    features,
                    kernel_size=scale_factor,
                    stride=scale_factor
                )
                multi_scale_features.append(downsampled)

        # 应用BiFPN进行多尺度特征融合
        fused_features_list = self.bifpn(multi_scale_features)
        # 使用最高分辨率的融合特征（原始分辨率）
        fused_features = fused_features_list[0]

        # 添加坐标特征
        coord_features = self.compute_coordinates(fused_features)
        features_with_coord = torch.cat([coord_features, fused_features], dim=1)

        # 通过分支处理特征
        pred_logits, pred_kernel, pred_scores, iam = self.inst_branch(features_with_coord)
        mask_features = self.mask_branch(features_with_coord)

        N = pred_kernel.shape[1]
        B, C, H, W = mask_features.shape
        pred_masks = torch.bmm(pred_kernel, mask_features.view(B, C, H * W)).view(B, N, H, W)

        pred_masks = F.interpolate(
            pred_masks, scale_factor=self.scale_factor,
            mode='bilinear', align_corners=False)

        output = {
            "pred_logits": pred_logits,
            "pred_masks": pred_masks,
            "pred_scores": pred_scores,
        }

        if self.output_iam:
            iam = F.interpolate(iam, scale_factor=self.scale_factor,
                                mode='bilinear', align_corners=False)
            output['pred_iam'] = iam

        return output

class GroupInstanceBranch(nn.Module):

    def __init__(self, cfg, in_channels):
        super().__init__()
        dim = cfg.MODEL.SPARSE_INST.DECODER.INST.DIM
        num_convs = cfg.MODEL.SPARSE_INST.DECODER.INST.CONVS
        num_masks = cfg.MODEL.SPARSE_INST.DECODER.NUM_MASKS
        kernel_dim = cfg.MODEL.SPARSE_INST.DECODER.KERNEL_DIM
        self.num_groups = cfg.MODEL.SPARSE_INST.DECODER.GROUPS
        self.num_classes = cfg.MODEL.SPARSE_INST.DECODER.NUM_CLASSES

        self.inst_convs = _make_stack_3x3_convs(num_convs, in_channels, dim)
        # iam prediction, a group conv
        expand_dim = dim * self.num_groups
        self.iam_conv = nn.Conv2d(dim, num_masks * self.num_groups, 3, padding=1, groups=self.num_groups)
        # outputs
        self.fc = nn.Linear(expand_dim, expand_dim)

        self.cls_score = nn.Linear(expand_dim, self.num_classes)
        self.mask_kernel = nn.Linear(expand_dim, kernel_dim)
        self.objectness = nn.Linear(expand_dim, 1)

        self.prior_prob = 0.01
        self._init_weights()

    def _init_weights(self):
        for m in self.inst_convs.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        for module in [self.iam_conv, self.cls_score]:
            init.constant_(module.bias, bias_value)
        init.normal_(self.iam_conv.weight, std=0.01)
        init.normal_(self.cls_score.weight, std=0.01)

        init.normal_(self.mask_kernel.weight, std=0.01)
        init.constant_(self.mask_kernel.bias, 0.0)
        c2_xavier_fill(self.fc)

    def forward(self, features):
        # instance features (x4 convs)
        features = self.inst_convs(features)
        # predict instance activation maps
        iam = self.iam_conv(features)
        iam_prob = iam.sigmoid()

        B, N = iam_prob.shape[:2]
        C = features.size(1)
        # BxNxHxW -> BxNx(HW)
        iam_prob = iam_prob.view(B, N, -1)
        # aggregate features: BxCxHxW -> Bx(HW)xC
        inst_features = torch.bmm(iam_prob, features.view(B, C, -1).permute(0, 2, 1))
        normalizer = iam_prob.sum(-1).clamp(min=1e-6)
        inst_features = inst_features / normalizer[:, :, None]

        inst_features = inst_features.reshape(
            B, 4, N // 4, -1).transpose(1, 2).reshape(B, N // 4, -1)

        inst_features = F.relu_(self.fc(inst_features))
        # predict classification & segmentation kernel & objectness
        pred_logits = self.cls_score(inst_features)
        pred_kernel = self.mask_kernel(inst_features)
        pred_scores = self.objectness(inst_features)
        return pred_logits, pred_kernel, pred_scores, iam


@SPARSE_INST_DECODER_REGISTRY.register()
class GroupIAMDecoder(BaseIAMDecoder):

    def __init__(self, cfg):
        super().__init__(cfg)
        in_channels = cfg.MODEL.SPARSE_INST.ENCODER.NUM_CHANNELS + 2
        self.inst_branch = GroupInstanceBranch(cfg, in_channels)


def build_sparse_inst_decoder(cfg):
    name = cfg.MODEL.SPARSE_INST.DECODER.NAME
    return SPARSE_INST_DECODER_REGISTRY.get(name)(cfg)
