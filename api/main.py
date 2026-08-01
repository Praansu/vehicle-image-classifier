"""Vehicle Image Classifier — FastAPI inference API."""

import io
import sys
from pathlib import Path

import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

# Allow importing model.py from the src/ folder
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model import create_resnet_model  # noqa: E402

CLASSES = ["bus", "car", "motorcycle", "truck"]

app = FastAPI(title="Vehicle Image Classifier API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
MODEL_PATH = Path(__file__).parent / "vehicle_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

_model = None


def get_model():
    global _model
    if _model is None:
        _model = create_resnet_model(num_classes=len(CLASSES))
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        _model.load_state_dict(state_dict)
        _model.to(DEVICE)
        _model.eval()
    return _model


@app.get("/health")
async def health():
    return {"status": "ok", "device": str(DEVICE), "classes": CLASSES}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise HTTPException(400, "Only image files are allowed (jpg, png, webp)")

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(400, "Image too large — max 10MB")

    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
        input_tensor = _transform(image).unsqueeze(0).to(DEVICE)
    except Exception:
        raise HTTPException(400, "Could not read image file")

    model = get_model()
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0]

    top_probs, top_indices = torch.topk(probabilities, k=len(CLASSES))

    predictions = [
        {
            "class": CLASSES[idx.item()],
            "confidence": round(prob.item(), 4),
        }
        for prob, idx in zip(top_probs, top_indices)
    ]

    return JSONResponse({
        "top_prediction": predictions[0],
        "all_predictions": predictions,
    })


# Serve the frontend — mounted last so it doesn't shadow API routes
frontend_dir = Path(__file__).parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
