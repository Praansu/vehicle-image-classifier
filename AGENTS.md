# Vehicle Image Classifier

## Stack
- Python + PyTorch + TorchVision
- Pillow for image processing
- Matplotlib for visualization

## Key Files
- `src/model.py` — model architecture
- `src/train.py` — training loop
- `src/data_loader.py` — dataset loading
- `src/predict.py` — inference

## Conventions
- PyTorch nn.Module for models
- Standard train/val/test split
- GPU if available, fallback to CPU
