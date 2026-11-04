# Vehicle Classifier

PyTorch + ResNet18, classifies bus / car / bike / truck. Got 90% on my own 400 images (100 per class).

Labelled them all myself with a little PyQt tool I made because labelling 400 images by hand in folders was painful. Has a FastAPI API + Docker because I wanted to learn deployment too.

## how to run
```
pip install -r requirements.txt
python train.py --epochs 10
python api.py  # FastAPI on :8000
```

Weights in `weights/best.pt`. Data layout in `data/` — 4 folders.

## stack
PyTorch, FastAPI, OpenCV, Docker

TODO: bikes vs trucks still get mixed sometimes, need more night images.
