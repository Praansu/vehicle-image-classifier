# Vehicle Image Classifier — API

Inference API for the vehicle image classifier. ResNet18 with transfer learning.

## Run locally

```bash
pip install -r requirements-api.txt
uvicorn api.main:app --reload
```

Open http://localhost:8000, upload a vehicle image, see the prediction with confidence bars.

## API

| Endpoint | Method | What it does |
|----------|--------|-------------|
| `/predict` | POST | Upload an image, returns top prediction + all class confidences |
| `/health` | GET | Status + device info |

## Docker

```bash
docker build -t vehicle-classifier -f Dockerfile .
docker run -p 8000:8000 vehicle-classifier
```

## Model

`api/vehicle_model.pth` — ResNet18 fine-tuned on 320 vehicle images (bus, car, motorcycle, truck) with a frozen backbone. ~90% validation accuracy.
