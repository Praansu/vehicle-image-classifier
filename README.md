# Vehicle Classifier

[![CI](https://img.shields.io/github/actions/workflow/status/Praansu/vehicle-image-classifier/ci.yml?style=flat-square)](https://github.com/Praansu/vehicle-image-classifier/actions)
[![License](https://img.shields.io/badge/License-MIT-1F2328?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-1F2328?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-1F2328?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Docker](https://img.shields.io/badge/Docker-ready-1F2328?style=flat-square&logo=docker&logoColor=white)](https://docker.com)

PyTorch + ResNet18, classifies **bus / car / bike / truck**. Got **90%** on my own 400 images (100 per class).

Labelled them all myself with a little PyQt tool I made because labelling 400 images by hand in folders was painful. Has a FastAPI API + Docker because I wanted to learn deployment too.

## Quick Start

```bash
# Train
pip install -r requirements.txt
python train.py --epochs 10

# Serve API
pip install -r requirements-api.txt
python api.py  # FastAPI on :8000
```

Or with Docker:
```bash
docker build -t vehicle-classifier .
docker run -p 8000:8000 vehicle-classifier
```

Weights in `weights/best.pt`. Data layout in `data/` — 4 folders.

## Stack

| Layer | Tech |
|-------|------|
| Model | PyTorch, ResNet18, transfer learning |
| API | FastAPI, Uvicorn |
| Container | Docker |
| Data | 400 hand-labelled images (100/class) |

## Results

| Class | Accuracy |
|-------|----------|
| Bus | ~92% |
| Car | ~94% |
| Bike | ~88% |
| Truck | ~86% |
| **Overall** | **~90%** |

Bikes vs trucks still get mixed sometimes — need more night images.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/predict` | Upload image, get prediction |
| GET | `/classes` | List class names |

## What I Learned

- Transfer learning with ResNet18 is surprisingly effective with small datasets
- Labelling tool (PyQt) saved hours vs manual folder sorting
- FastAPI + Docker makes deployment trivial
- Night images are a blind spot — data diversity matters more than model complexity