import math
import sys
import os
import argparse
import traceback

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch import nn
from torch.nn import functional as F

# Detectron2 imports
from detectron2.layers import Conv2d, DeformConv, ModulatedDeformConv
from detectron2.utils.logger import setup_logger
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg

# SparseInst imports
from sparseinst.config import add_sparse_inst_config


class PyramidPoolingModuleONNX(nn.Module):
    def __init__(self, in_channels, channels, input_size, pool_sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList(
            [self._make_stage(in_channels, channels, input_size, pool_size)
             for pool_size in pool_sizes]
        )
        self.bottleneck = Conv2d(
            in_channels + len(pool_sizes) * channels, in_channels, 1)

    def _make_stage(self, features, out_features, input_size, pool_size):
        stride_y = math.floor((input_size[0] / pool_size))
        stride_x = math.floor((input_size[1] / pool_size))
        kernel_y = input_size[0] - (pool_size - 1) * stride_y
        kernel_x = input_size[1] - (pool_size - 1) * stride_x
        prior = nn.AvgPool2d(kernel_size=(
            kernel_y, kernel_x), stride=(stride_y, stride_x))
        conv = Conv2d(features, out_features, 1)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(
            input=F.relu_(stage(feats)), size=(h, w), mode='bilinear', align_corners=False) for stage in self.stages] + [feats]
        out = F.relu_(self.bottleneck(torch.cat(priors, 1)))
        return out


# --- 关键修改：定义一个能“吃掉”多余参数的 Wrapper 类 ---
class DCNv2Wrapper(nn.Conv2d):
    """
    这是一个伪装成 DCN 的标准卷积层。
    它的作用是接收 (x, offset, mask) 等参数，但只使用 x 进行标准卷积计算。
    从而骗过源代码的调用逻辑，同时输出标准的 ONNX 卷积节点。
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x, *args, **kwargs):
        # *args 会捕获 offset, mask 等多余参数，然后我们将它们忽略
        return super().forward(x)


def replace_coordatt_pooling(model, input_h=640, input_w=640, logger=None):
    replaced = 0
    for name, module in model.named_modules():
        if 'attn' in name.lower() and hasattr(module, 'pool_h') and hasattr(module, 'pool_w'):
            if 'layer3' in name:
                h_feat, w_feat = input_h // 16, input_w // 16
            elif 'layer4' in name:
                h_feat, w_feat = input_h // 32, input_w // 32
            else:
                continue
            
            module.pool_h = nn.AvgPool2d(kernel_size=(1, w_feat), stride=1)
            module.pool_w = nn.AvgPool2d(kernel_size=(h_feat, 1), stride=1)
            
            if logger:
                logger.info(f"✅ 替换 CoordAtt 池化: {name} | 特征图 {h_feat}x{w_feat}")
            replaced += 1
    
    if logger:
        logger.info(f"✅ 共替换 {replaced} 个 CoordAtt 模块的自适应池化层")
    return replaced


def replace_detectron_dcn(model, logger=None):
    """
    使用 DCNv2Wrapper 替换 DeformConv。
    """
    replaced = 0
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, (DeformConv, ModulatedDeformConv)):
                # 1. 提取参数
                in_channels = child.in_channels
                out_channels = child.out_channels
                kernel_size = child.kernel_size
                stride = child.stride
                padding = child.padding
                dilation = child.dilation
                groups = child.groups
                bias = child.bias is not None
                
                # 2. 创建 Wrapper 卷积 (关键修改点!)
                new_conv = DCNv2Wrapper(
                    in_channels, out_channels, kernel_size,
                    stride=stride, padding=padding,
                    dilation=dilation, groups=groups, bias=bias
                )
                
                # 3. 复制权重
                new_conv.weight.data = child.weight.data
                if bias:
                    new_conv.bias.data = child.bias.data
                
                new_conv.to(child.weight.device)
                
                # 4. 替换
                setattr(module, child_name, new_conv)
                replaced += 1
                if logger:
                    logger.info(f"🔄 降级替换 DCN (Wrapper): {name}.{child_name} -> DCNv2Wrapper")

    if logger:
        logger.info(f"✅ 共替换 {replaced} 个 DCN 层为 DCNv2Wrapper")


def register_linspace_force(logger=None):
    try:
        import torch.onnx.symbolic_opset11 as sym11
    except ImportError:
        if logger: logger.warning("无法导入 symbolic_opset11，尝试直接注册可能失败")
        return

    def linspace_symbolic(g, start, end, steps, *args):
        zero = g.op("Constant", value_t=torch.tensor(0, dtype=torch.int64))
        one = g.op("Constant", value_t=torch.tensor(1, dtype=torch.int64))
        steps_int = g.op("Cast", steps, to_i=7)
        indices = g.op("Range", zero, steps_int, one)
        indices_f = g.op("Cast", indices, to_i=1)
        start_f = g.op("Cast", start, to_i=1)
        end_f = g.op("Cast", end, to_i=1)
        steps_f = g.op("Cast", steps, to_i=1)
        steps_minus_1 = g.op("Sub", steps_f, g.op("Constant", value_t=torch.tensor(1.0)))
        range_val = g.op("Sub", end_f, start_f)
        delta = g.op("Div", range_val, steps_minus_1)
        return g.op("Add", start_f, g.op("Mul", delta, indices_f))

    sym11.linspace = linspace_symbolic
    if logger:
        logger.info("已强制 Monkey Patch 注册 linspace 到 symbolic_opset11")


def main():
    parser = argparse.ArgumentParser(description="Export model to the onnx format")
    parser.add_argument("--config-file", default="/home/liweijia/SparseInst/configs/Mango-sparseinst.yaml", metavar="FILE", help="path to config file")
    parser.add_argument('--width', default=640, type=int)
    parser.add_argument('--height', default=640, type=int)
    parser.add_argument('--level', default=0, type=int)
    parser.add_argument("--output", default="/home/liweijia/SparseInst/result", metavar="FILE", help="path to the output onnx file")
    parser.add_argument("--opts", help="Modify config options using the command-line 'KEY VALUE' pairs", default=[], nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # --- 1. 输出路径智能修正 ---
    save_path = args.output
    if os.path.isdir(save_path) or not save_path.endswith('.onnx'):
        if os.path.isdir(save_path):
            save_path = os.path.join(save_path, "sparseinst.onnx")
        elif not save_path.endswith('.onnx'):
            save_path = save_path + ".onnx"
    
    output_dir = os.path.dirname(save_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger(output=output_dir) 
    logger.info(f"最终输出路径确定为: {save_path}")

    # --- 2. 模型构建 ---
    cfg = get_cfg()
    add_sparse_inst_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.BACKBONE.FREEZE_AT = 0
    cfg.MODEL.RESNETS.NORM = "BN"
    cfg.freeze()

    height = args.height
    width = args.width

    model = build_model(cfg)
    num_channels = cfg.MODEL.SPARSE_INST.ENCODER.NUM_CHANNELS
    
    onnx_ppm = PyramidPoolingModuleONNX(num_channels, num_channels // 4, (height // 32, width // 32))
    model.encoder.ppm = onnx_ppm
    model.to(cfg.MODEL.DEVICE)

    checkpointer = DetectionCheckpointer(model)
    _ = checkpointer.load(cfg.MODEL.WEIGHTS)
    logger.info(f"成功加载权重: {cfg.MODEL.WEIGHTS}")

    # --- 3. 应用补丁 (已更新为 Wrapper 方式) ---
    replace_coordatt_pooling(model, height, width, logger)  
    replace_detectron_dcn(model, logger) 
    register_linspace_force(logger) 

    input_names = ["input_image"]
    output_names = ["scores", "masks"]

    model.forward = model.forward_test
    dummy_input = torch.zeros((1, 3, height, width)).to(cfg.MODEL.DEVICE)

    print("\n" + "-"*50)
    logger.info("⏳ 开始导出 ONNX 模型 (Verbose=False)...")
    print("-" * 50)

    try:
        # --- 4. 执行导出 ---
        torch.onnx.export(
            model,
            dummy_input,
            save_path, 
            verbose=False, 
            input_names=input_names,
            output_names=output_names,
            keep_initializers_as_inputs=False,
            opset_version=11, 
        )
        
        # --- 5. 结果验证 ---
        if os.path.exists(save_path):
            file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
            print("\n" + "="*60)
            print(f"🎉 导出成功 (Export Success)!")
            print(f"📂 文件路径: {save_path}")
            print(f"📦 文件大小: {file_size_mb:.2f} MB")
            print("="*60 + "\n")
            logger.info(f"Done. The onnx model is saved into {save_path}")
        else:
            logger.error(f"❌ 导出函数执行完毕，但在 {save_path} 未找到文件。")

    except Exception as e:
        print("\n" + "!!!!" * 20)
        logger.error("❌ 导出过程中发生致命错误 (Fatal Error)!")
        logger.error(f"❌ 错误信息: {e}")
        print("!!!!" * 20 + "\n")
        traceback.print_exc()

if __name__ == "__main__":
    main()