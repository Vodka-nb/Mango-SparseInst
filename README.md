
# Mango-SparseInst

A lightweight two-stage framework for instance segmentation of mango fruits and main branches, and picking point localization in automated harvesting.

## 📦 Dataset & Pretrained Weights

All resources are openly available via Zenodo:

- **Dataset:** [DOI: 10.5281/zenodo.19727667](https://doi.org/10.5281/zenodo.19727667)
- **Inference Weights:** [DOI: 10.5281/zenodo.19731143](https://doi.org/10.5281/zenodo.19731143)

### Dataset Structure


## ⚙️ Environment Setup

```bash
# Clone repository
git clone https://github.com/vodka-nb/Mango-SparseInst.git
cd Mango-SparseInst

# Install dependencies
Tested Environment:
Python 3.9+
PyTorch 1.12+
CUDA 11.6+
Operating System: Linux (Ubuntu 18.04/20.04)

🚀 Quick Start
Training
Train the model using the default configuration:
python train_net.py --config-file config/config.yaml --num-gpus 1
Key Parameters:
--config-file: Path to configuration YAML file
--num-gpus: Number of GPUs to use (default: 1)
--resume: Resume training from checkpoint
Training logs and model checkpoints are saved to output/ directory by default.

- Evaluation
Evaluate the trained model on the validation set:
python test_net.py --config-file config/config.yaml \
    --eval-only \
    MODEL.WEIGHTS output/model_best.pth

- Output Metrics:
AP (Average Precision) at IoU thresholds 0.50:0.95
AP50 (IoU = 0.50)
AP75 (IoU = 0.75)
AR (Average Recall)
Per-class metrics for fruits and branches

- FPEA Module:
   python inference/Picking_points.py
Runs after segmentation
No additional configuration required
Optimizes picking points based on energy function combining:
1.Distance to fruit centroid
2.Branch curvature
3.Occlusion level
4.Accessibility constraints

- ONNX Export
Export the model to ONNX format for deployment:
  python onnx/onnx.py --config-file config/config.yaml \
    --checkpoint output/model_best.pth \
    --output onnx/mango_sparseinst.onnx \
    --opset 13



 
