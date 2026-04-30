import subprocess
import os

# 参数配置（根据实际路径修改）
demo_script = "../demo.py"
config_file = "/home/xxxx/SparseInst/result/Mango_1300_sparse_ultra/config.yaml"
input_path = "/home/xxxx/SparseInst/Mango_1300_COCO/test/images/*.jpg"  # 支持通配符匹配多张图片<a target="_blank" href="https://www.cnblogs.com/god-of-death/p/18559484" class="hitref" data-title="yolo --- 快速上手 - 流水灯 - 博客园" data-snippet='yolo 任务名称 model=本地模型权重路径 source=图片路径 yolo predict model=yolo11n.pt source=&#39;https://ultralytics.com/images/bus.jpg&#39; 运行方式 - Python 脚本 模型训...' data-url="https://www.cnblogs.com/god-of-death/p/18559484">8</a>
output_dir = "/home/xxxx/SparseInst/Plot/inference_test"
model_weights = "/home/xxxx/SparseInst/result/Mango_1300_sparse_ultra/model_final.pth"
min_size_test = 512  # 输入图像的最小尺寸<a target="_blank" href="https://www.cnblogs.com/god-of-death/p/18559484" class="hitref" data-title="yolo --- 快速上手 - 流水灯 - 博客园" data-snippet='yolo 任务名称 model=本地模型权重路径 source=图片路径 yolo predict model=yolo11n.pt source=&#39;https://ultralytics.com/images/bus.jpg&#39; 运行方式 - Python 脚本 模型训...' data-url="https://www.cnblogs.com/god-of-death/p/18559484">8</a>

# 创建输出目录（若不存在）
os.makedirs(output_dir, exist_ok=True)  # 参考<a target="_blank" href="https://blog.51cto.com/u_16175437/6823258" class="hitref" data-title="python输出图片到指定文件夹_mob649e81553a70的技术博客_5..." data-snippet='image.save(output_path) 1. 完整代码示例 下面是完整的代码示例,包括上述的每一步骤。 fromPILimportImageimportos# 加载要输出的图片image=Image.open(&quot;inpu...' data-url="https://blog.51cto.com/u_16175437/6823258">4</a>

# 构建命令参数列表
command = [
    "python", demo_script,
    "--config-file", config_file,
    "--input", input_path,
    "--output", output_dir,
    "--opts",
    "MODEL.WEIGHTS", model_weights,
    "INPUT.MIN_SIZE_TEST", str(min_size_test)
]

# 执行命令
try:
    subprocess.run(command, check=True)
    print(f"推理完成！结果已保存至: {output_dir}")
except subprocess.CalledProcessError as e:
    print(f"执行失败: {e}")
except FileNotFoundError:
    print("错误：未找到demo.py或配置文件路径错误")
