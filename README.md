# Mango-SparseInst

A lightweight two-stage framework for instance segmentation of mango fruits and main branches, and picking point localization in automated harvesting.

## 📦 Dataset & Pretrained Weights

All resources are openly available via Zenodo:

- **Dataset:** [DOI: 10.5281/zenodo.19727667](https://doi.org/10.5281/zenodo.19727667)
- **Inference Weights:** [DOI: 10.5281/zenodo.19731143](https://doi.org/10.5281/zenodo.19731143)

## ⚙️ Environment Setup

```bash
# Clone repository
git clone https://github.com/vodka-nb/Mango-SparseInst.git
cd Mango-SparseInst
```

**Tested Environment:**
- Python 3.9+
- PyTorch 1.12+
- CUDA 11.6+
- Operating System: Linux (Ubuntu 18.04 / 20.04)
- detectron2: https://detectron2-zhcn.readthedocs.io/zh-cn/latest/tutorials/install.html

## 🚀 Quick Start

### Training

Train the model using the default configuration:

```bash
python train_net.py --config-file config/config.yaml --num-gpus 1
```

**Key Parameters:**
- `--config-file`: Path to configuration YAML file
- `--num-gpus`: Number of GPUs to use (default: 1)
- `--resume`: Resume training from checkpoint

Training logs and model checkpoints are saved to the `output/` directory by default.

### Evaluation

Evaluate the trained model on the validation set:

```bash
python test_net.py --config-file config/config.yaml \
    --eval-only \
    MODEL.WEIGHTS output/model_best.pth
```

**Output Metrics:**
- AP (Average Precision) at IoU thresholds 0.50:0.95
- AP50 (IoU = 0.50)
- AP50:95 (Average Precision at IoU thresholds 0.50:0.95)
- AR (Average Recall)
- Per-class metrics for fruits and branches

### FPEA Module

```bash
python inference/Picking_points.py
```

- Runs after segmentation
- No additional configuration required
- Optimizes picking points based on an energy function combining:
  1. Distance to fruit centroid
  2. Branch curvature
  3. Occlusion level
  4. Accessibility constraints

### ONNX Export

Export the model to ONNX format for deployment:

```bash
python onnx/onnx.py --config-file config/config.yaml \
    --checkpoint output/model_best.pth \
    --output onnx/mango_sparseinst.onnx \
    --opset 13
```

**Output format:**

```json
[
  {
    "image": "img_101.jpg",
    "instances": [
      {
        "instance_id": 1,
        "class": "fruit",
        "bbox": [x1, y1, x2, y2],
        "confidence": 0.95,
        "picking_point": {
          "x": 245.3,
          "y": 180.7,
          "confidence": 0.92
        }
      }
    ]
  }
]
```
