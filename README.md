# Vehicle Image Classifier

I wanted to understand how CNNs actually work — not just the theory, but the messy reality of training one from scratch. So I built this: a PyTorch classifier that tells cars, bikes, buses, and vans apart.

**Dataset:** ~4K vehicle images. **GPU time:** more than I'd like to admit. **Satisfaction when it finally worked:** priceless.

## The short version

You give it a vehicle image, it tells you what it is. Under the hood it's a CNN (Convolutional Neural Network) with optional transfer learning via ResNet-18.

## How the code is organized

```
vehicle-image-classifier/
├── src/
│   ├── data_loader.py    # loads images, preps them for training
│   ├── model.py          # CNN architecture + ResNet option
│   ├── train.py          # the training loop (where the magic happens)
│   └── predict.py        # use your trained model on new images
├── data/                 # put your dataset here
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

Put your images in `data/train/{class}/` and `data/val/{class}/`. Each class folder should have at least 30-50 images.

## Usage

### Train from scratch (slower, but you learn more)

```bash
python src/train.py --epochs 20 --batch_size 32 --lr 0.001
```

### Train with transfer learning (faster, better accuracy)

```bash
python src/train.py --use_resnet --epochs 10 --batch_size 32 --lr 0.0001
```

### Predict

```bash
python src/predict.py --image path/to/your/image.jpg
```

## What I figured out along the way

- Batch size matters more than I thought. Too big = memory issues. Too small = won't converge.
- Data augmentation is basically cheating — but the good kind of cheating.
- Transfer learning with ResNet-18 gives way better results than training from scratch (unsurprising, but still cool to see).
- Most of the time spent on this was debugging, not building. That's just how ML works, I guess.

## Requirements

Python 3.8+, PyTorch, torchvision, Pillow, matplotlib
