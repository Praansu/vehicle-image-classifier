# Contributing to Vehicle Image Classifier

Thank you for considering contributing!

## How to Contribute

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes** with clear, focused commits
4. **Run tests and linting**: `ruff check src/ api/ && ruff format --check src/ api/`
5. **Open a Pull Request** with a clear description

## Code Style

- Python: ruff for linting and formatting
- Commit messages: Conventional Commits

## Development Setup

```bash
git clone https://github.com/Praansu/vehicle-image-classifier.git
cd vehicle-image-classifier
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-api.txt
# Train model
python src/train.py --data_dir data --epochs 10 --use_resnet
# Run API
uvicorn api.main:app --reload
```

## Project Structure

```
vehicle-image-classifier/
├── src/
│   ├── model.py      # VehicleCNN + ResNet18 transfer learning
│   ├── train.py      # Training loop with validation
│   ├── data_loader.py # Dataset loading + augmentation
│   └── predict.py    # CLI inference
├── api/
│   ├── main.py       # FastAPI inference API
│   └── frontend/     # Simple upload UI
├── scripts/
│   └── generate_confusion_matrix.py
└── .github/workflows/ # CI/CD pipelines
```

## Areas for Contribution

- Add Grad-CAM visualization
- Try newer architectures (MobileNetV3, EfficientNet)
- Add more data augmentation
- Write unit/integration tests
- Improve frontend UX
- Add Docker Compose for production