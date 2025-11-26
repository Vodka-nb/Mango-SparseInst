
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from fvcore.nn.weight_init import c2_msra_fill, c2_xavier_fill

from detectron2.utils.registry import Registry
from detectron2.layers import Conv2d

SPARSE_INST_ENCODER_REGISTRY = Registry("SPARSE_INST_ENCODER")
SPARSE_INST_ENCODER_REGISTRY.__doc__ = "registry for SparseInst decoder"

# class SelectiveKernelConv(nn.Module):
#     """增强型选择性核卷积（低计算量优化版）"""
#     def __init__(self, in_channels, out_channels, stride=1,
#                  kernels=[3, 5], reduction=32, groups=1):
#         super().__init__()
#         self.groups = groups
#
#         # Split阶段：多分支卷积
#         self.convs = nn.ModuleList([])
#         for k in kernels:
#             padding = (k - 1) // 2
#             self.convs.append(
#                 nn.Sequential(
#                     Conv2d(in_channels, out_channels, kernel_size=k,
#                            stride=stride, padding=padding, groups=groups),
#                     nn.BatchNorm2d(out_channels),
#                     nn.ReLU(inplace=True)
#                 )
#             )
#
#         # Fuse阶段：特征融合
#         mid_channels = max(out_channels // reduction, 16)  # reduction从16→32
#         self.fc = nn.Sequential(
#             Conv2d(out_channels, mid_channels, 1),
#             nn.ReLU(inplace=True)
#         )
#
#         # Select阶段：注意力机制（仅输出每个分支的全局权重）
#         self.attention = nn.Conv2d(mid_channels, len(kernels), 1)  # 输出K个权重
#
#         self._init_weights()
#
#     def _init_weights(self):
#         for m in self.modules():
#             if isinstance(m, Conv2d):
#                 c2_xavier_fill(m)
#
#     def forward(self, x):
#         # Split阶段
#         feats = [conv(x) for conv in self.convs]
#         feats = torch.stack(feats, dim=0)  # [K, B, C, H, W]
#
#         # Fuse阶段
#         U = sum(feats)
#         Z = self.fc(F.adaptive_avg_pool2d(U, (1, 1)))  # [B, mid_channels, 1, 1]
#
#         # Select阶段：生成分支权重并广播到所有通道
#         weights = self.attention(Z)  # [B, K, 1, 1]
#         weights = weights.unsqueeze(2).expand(-1, -1, feats.size(2), -1, -1)  # [B, K, C, 1, 1]
#         weights = weights.permute(1, 0, 2, 3, 4)  # [K, B, C, 1, 1]
#
#         # 加权融合
#         V = (weights * feats).sum(dim=0)
#         return V

class SelectiveKernelConv(nn.Module):
    """ 增强型选择性核卷积 """

    def __init__(self, in_channels, out_channels, stride=1,
                 kernels=[3, 5], reduction=16, groups=1):
        super().__init__()
        self.groups = groups
        # 多分支卷积（Split阶段）
        self.convs = nn.ModuleList([])
        for k in kernels:
            padding = (k - 1) // 2
            self.convs.append(
                nn.Sequential(
                    Conv2d(in_channels, out_channels, kernel_size=k,
                           stride=stride, padding=padding, groups=groups),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        # 特征融合（Fuse阶段）
        mid_channels = max(out_channels // reduction, 32)
        self.fc = nn.Sequential(
            Conv2d(out_channels, mid_channels, 1),
            nn.ReLU(inplace=True)
        )
        # 注意力机制（Select阶段）
        self.attention = nn.Conv2d(mid_channels, len(kernels) * out_channels, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                c2_xavier_fill(m)

    def forward(self, x):
        # Split阶段
        feats = [conv(x) for conv in self.convs]
        feats = torch.stack(feats, dim=0)  # [K, B, C, H, W]

        # Fuse阶段
        U = sum(feats)
        Z = self.fc(F.adaptive_avg_pool2d(U, (1, 1)))  # [B, mid, 1,1]

        # Select阶段（修正维度对齐）
        weights = self.attention(Z)  # [B, K*C, 1,1]

        # 动态获取维度参数
        K, B, C = len(self.convs), x.size(0), self.convs[0][0].out_channels
        weights = weights.view(B, K, C, 1, 1)  # [B, K, C, 1,1]
        weights = weights.permute(1, 0, 2, 3, 4)  # [K, B, C, 1,1]

        # 加权融合
        V = (weights * feats).sum(dim=0)
        return V


class PyramidPoolingModule(nn.Module):

    def __init__(self, in_channels, channels=512, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList(
            [self._make_stage(in_channels, channels, size) for size in sizes]
        )
        self.bottleneck = Conv2d(
            in_channels + len(sizes) * channels, in_channels, 1)

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = Conv2d(features, out_features, 1)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=F.relu_(stage(feats)), size=(
            h, w), mode='bilinear', align_corners=False) for stage in self.stages] + [feats]
        out = F.relu_(self.bottleneck(torch.cat(priors, 1)))
        return out


@SPARSE_INST_ENCODER_REGISTRY.register()
class InstanceContextEncoder(nn.Module):
    """
    Instance Context Encoder
    1. construct feature pyramids from ResNet
    2. enlarge receptive fields (ppm)
    3. multi-scale fusion
    """

    def __init__(self, cfg, input_shape):
        super().__init__()
        self.num_channels = cfg.MODEL.SPARSE_INST.ENCODER.NUM_CHANNELS
        self.in_features = cfg.MODEL.SPARSE_INST.ENCODER.IN_FEATURES
        self.in_channels = [input_shape[f].channels for f in self.in_features]
        fpn_laterals = []
        fpn_outputs = []
        for in_channel in reversed(self.in_channels):
            lateral_conv = Conv2d(in_channel, self.num_channels, 1)
            output_conv = Conv2d(self.num_channels, self.num_channels, 3, padding=1)
            c2_xavier_fill(lateral_conv)
            c2_xavier_fill(output_conv)
            fpn_laterals.append(lateral_conv)

            # 输出卷积替换为增强型自适应卷积
            output_conv = SelectiveKernelConv(
                in_channels=self.num_channels,
                out_channels=self.num_channels,
                kernels=[3, 5],  # 多尺度卷积核
                reduction=8  # 通道压缩比例
            )
            fpn_outputs.append(output_conv)

        self.fpn_laterals = nn.ModuleList(fpn_laterals)
        self.fpn_outputs = nn.ModuleList(fpn_outputs)
        
        # ppm
        self.ppm = PyramidPoolingModule(self.num_channels, self.num_channels // 4)

        # 双向特征金字塔（PANet）
        # self.bottom_up_fuse = nn.ModuleList([
        #     Conv2d(self.num_channels, self.num_channels, 1)
        #     for _ in range(len(self.in_channels) - 1)
        # ])

        # final fusion
        self.fusion = nn.Conv2d(self.num_channels * 3, self.num_channels, 1)
        c2_msra_fill(self.fusion)

    def forward(self, features):
        features = [features[f] for f in self.in_features]
        features = features[::-1] # 将特征列表顺序反转，实现自顶向下的特征融合, [::-1] 表示以步长 -1 遍历列表（即逆序）

        # 特征金字塔处理流程
        prev_features = self.ppm(self.fpn_laterals[0](features[0]))
        outputs = [self.fpn_outputs[0](prev_features)]  # 使用增强卷积
        # [1:] 表示从索引 1 开始到末尾（features[0] 已单独处理） 从第二个元素开始遍历特征列表
        for feature, lat_conv, output_conv in zip(features[1:], self.fpn_laterals[1:], self.fpn_outputs[1:]):
            lat_features = lat_conv(feature)
            top_down_features = F.interpolate(prev_features, scale_factor=2.0, mode='nearest')
            prev_features = lat_features + top_down_features
            outputs.insert(0, output_conv(prev_features))  # 增强卷积处理

        # 自底向上融合（PANet）
        # for i in range(len(outputs) - 1):

        #     outputs[i + 1] = outputs[i + 1] + F.interpolate(
        #         self.bottom_up_fuse[i](outputs[i]),
        #         size=outputs[i + 1].shape[2:],  # 显式指定目标尺寸
        #         mode='nearest'
        #     )

        size = outputs[0].shape[2:] #提取第一个输出特征图的高度和宽度,outputs[0] 是四维张量（Batch × Channel × Height × Width）,[2:]跳过前两个维度再开始遍历，即获取H×W
        features = [
            outputs[0]] + [F.interpolate(x, size, mode='bilinear', align_corners=False) for x in outputs[1:]]
        #上采样即是将当前特征图的批量、通道、高宽构成的张量，每个维度与目标特征图对齐
        # 假设输入特征尺寸：
        # - C3: (B, 256, 56, 56)
        # - C4: (B, 512, 28, 28)
        # - C5: (B, 1024, 14, 14)
        #
        # 输出对齐后：
        # - O3: (B, 256, 56, 56)
        # - O4: (B, 256, 56, 56)  # 上采样至O3尺寸
        # - O5: (B, 256, 56, 56)  # 上采样至O3尺寸
        features = self.fusion(torch.cat(features, dim=1))
        return features


def build_sparse_inst_encoder(cfg, input_shape):
    name = cfg.MODEL.SPARSE_INST.ENCODER.NAME
    return SPARSE_INST_ENCODER_REGISTRY.get(name)(cfg, input_shape)



# class SelectiveKernelConv(nn.Module):
#     """ 增强型选择性核卷积 """
#
#     def __init__(self, in_channels, out_channels, stride=1,
#                  kernels=[3, 5], reduction=16, groups=1):
#         super().__init__()
#         self.groups = groups
#         # 多分支卷积（Split阶段）
#         self.convs = nn.ModuleList([])
#         for k in kernels:
#             padding = (k - 1) // 2
#             self.convs.append(
#                 nn.Sequential(
#                     Conv2d(in_channels, out_channels, kernel_size=k,
#                            stride=stride, padding=padding, groups=groups),
#                     nn.BatchNorm2d(out_channels),
#                     nn.ReLU(inplace=True)
#                 )
#             )
#         # 特征融合（Fuse阶段）
#         mid_channels = max(out_channels // reduction, 32)
#         self.fc = nn.Sequential(
#             Conv2d(out_channels, mid_channels, 1),
#             nn.ReLU(inplace=True)
#         )
#         # 注意力机制（Select阶段）
#         self.attention = nn.Conv2d(mid_channels, len(kernels) * out_channels, 1)
#
#         self._init_weights()
#
#     def _init_weights(self):
#         for m in self.modules():
#             if isinstance(m, Conv2d):
#                 c2_xavier_fill(m)
#
#     def forward(self, x):
#         # Split阶段
#         feats = [conv(x) for conv in self.convs]
#         feats = torch.stack(feats, dim=0)  # [K, B, C, H, W]
#
#         # Fuse阶段
#         U = sum(feats)
#         Z = self.fc(F.adaptive_avg_pool2d(U, (1, 1)))  # [B, mid, 1,1]
#
#         # Select阶段（修正维度对齐）
#         weights = self.attention(Z)  # [B, K*C, 1,1]
#
#         # 动态获取维度参数
#         K, B, C = len(self.convs), x.size(0), self.convs[0][0].out_channels
#         weights = weights.view(B, K, C, 1, 1)  # [B, K, C, 1,1]
#         weights = weights.permute(1, 0, 2, 3, 4)  # [K, B, C, 1,1]
#
#         # 加权融合
#         V = (weights * feats).sum(dim=0)
#         return V

