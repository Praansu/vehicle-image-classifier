# Vehicle Image Classifier

A PyTorch-based CNN for classifying vehicle images into 4 categories: car, bike, bus, and van.

Built as a learning project to understand:
- Building CNNs from scratch with PyTorch
- Data augmentation for small datasets
- Transfer learning with pretrained ResNet
- Training, evaluation, and inference pipelines

## Project Structure

```
vehicle-image-classifier/
├── src/
│   ├── model.py          # CNN from scratch + ResNet transfer learning
│   ├── train.py          # training loop
│   ├── predict.py        # run inference on new images
│   └── data_loader.py    # dataset loading and augmentation
├── data/                 # place your dataset here
├── notebooks/
│   └── explore.ipynb     # dataset exploration and visualization
├── requirements.txt
└── README.md
```

## Dataset

Organize your images in `data/` as:

```
data/
├── train/
│   ├── car/   (100+ images)
│   ├── bike/  (100+ images)
│   ├── bus/   (100+ images)
│   └── van/   (100+ images)
├── val/
│   └── (same structure)
└── test/
    └── (same structure)
```

## Usage

### Train from scratch

```bash
python src/train.py --epochs 20 --batch_size 32 --lr 0.001
```

### Train with transfer learning

```bash
python src/train.py --use_resnet --epochs 10 --batch_size 32 --lr 0.0001
```

### Predict

```bash
python src/predict.py --image path/to/image.jpg
```

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- Pillow
- matplotlib
