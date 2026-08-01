# Vehicle Image Classifier

Classifies vehicle photos into 4 classes — **bus, car, motorcycle, truck** — using a CNN trained with transfer learning. Includes a training pipeline (PyTorch) and a deployed inference API (FastAPI).

This was my first real ML project. I started with a CNN from scratch to understand how convolutions work, then switched to transfer learning with ResNet18 once I had a dataset — that's the pattern most real-world projects follow.

## What's in here

```
vehicle-image-classifier/
├── src/
│   ├── data_loader.py    # loads images, applies train/val transforms
│   ├── model.py          # VehicleCNN (from scratch) + ResNet18 (transfer learning)
│   ├── train.py          # training loop with validation
│   └── predict.py        # CLI prediction on a single image
├── api/
│   ├── main.py           # FastAPI inference API
│   ├── frontend/         # simple upload + confidence bars UI
│   └── vehicle_model.pth # trained model (ResNet18, ~90% val accuracy)
├── Dockerfile            # containerize the API
└── requirements*.txt
```

## Dataset

The dataset comes from [Asseh/Vehicle_Classification](https://huggingface.co/datasets/Asseh/Vehicle_Classification) on Hugging Face — 400 images, 100 per class (bus, car, motorcycle, truck). I split it 80/10/10 into train/val/test and removed the original "van" class from the code since the dataset didn't have one.

```bash
# Optional: download + split the dataset yourself (I ran this to create data/)
python scripts/prepare_data.py
```

## Train

```bash
pip install -r requirements.txt

# Transfer learning (recommended — freeze backbone, train the head only)
python src/train.py --data_dir data --epochs 10 --use_resnet

# CNN from scratch (slower, good for learning)
python src/train.py --data_dir data --epochs 20
```

My run: 10 epochs, ResNet18 transfer learning → **90% validation accuracy** on 320 training images.

## Run the API

```bash
pip install -r requirements-api.txt
uvicorn api.main:app --reload
```

Open http://localhost:8000 — upload a vehicle photo, see the top prediction and confidence bars for all 4 classes.

### Docker

```bash
docker build -t vehicle-classifier .
docker run -p 8000:8000 vehicle-classifier
```

### API

| Endpoint | Method | What it does |
|----------|--------|-------------|
| `/predict` | POST | Upload an image → top prediction + all class confidences |
| `/health` | GET | Status + device info |

## What I learned

- **Transfer learning beats from-scratch for small datasets.** 320 images is nowhere near enough to train a CNN properly. Freezing a pretrained ResNet18 backbone and only training the final layer got me to 90% accuracy in minutes — versus struggling to break 60% with my own architecture.
- **Class balance matters.** Even with only 100 images per class, keeping the split balanced made validation stable.
- **Serving a model is its own skill.** Loading the weights once at startup, keeping the model in `eval()` mode, and using `torch.no_grad()` for inference — small details that matter in production.
- **Transfer learning on CPU is fast.** Because the backbone weights are frozen, only the final layer gets gradients — so even my laptop CPU can train it in a few minutes.

## What I'd improve

- **Data augmentation:** I used basic flips/rotations. More augmentation (or more data) would push accuracy higher.
- **Model size:** ResNet18 is old. A newer small model like MobileNetV3 would be faster and smaller for deployment.
- **GPU training:** On a GPU I'd unfreeze more layers and train longer for better accuracy.
- **Frontend:** It works but is basic — no drag-drop, no history.
