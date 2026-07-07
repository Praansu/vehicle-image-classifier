# Vehicle Image Classifier

A beginner-friendly PyTorch project that teaches how to build, train, and use a neural network to classify vehicle images (car, bike, bus, van).

**Learning journey:** Start with `data_loader.py` → understand how images are loaded → then `model.py` to see how a CNN is built → then `train.py` to see the full training loop → finally `predict.py` to use the trained model.

## What You'll Learn

- How to load and prepare image data for PyTorch
- Building a Convolutional Neural Network (CNN) from scratch
- Using a pretrained ResNet model (transfer learning)
- Training a model and tracking accuracy
- Making predictions on new images

## Project Structure

```
vehicle-image-classifier/
├── src/
│   ├── data_loader.py    # loads images from folders, prepares them for training
│   ├── model.py          # defines the CNN architecture + ResNet option
│   ├── train.py          # the training loop — where the learning happens
│   └── predict.py        # use your trained model on new images
├── data/                 # put your dataset here
├── requirements.txt
└── README.md
```

## Setup

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Organize your images in `data/` like this:
   ```
   data/
   ├── train/
   │   ├── car/   (at least 30-50 images per class)
   │   ├── bike/
   │   ├── bus/
   │   └── van/
   ├── val/
   │   └── (same folder structure)
   └── test/
       └── (same folder structure)
   ```

## How to Use

### Train a CNN from scratch (good for learning)

```bash
python src/train.py --epochs 20 --batch_size 32 --lr 0.001
```

### Train with transfer learning (better accuracy, faster)

```bash
python src/train.py --use_resnet --epochs 10 --batch_size 32 --lr 0.0001
```

### Predict on a new image

```bash
python src/predict.py --image path/to/your/image.jpg
```

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- Pillow
- matplotlib
