# Vehicle Image Classifier

Wanted to understand how CNNs actually work — the messy reality, not just the theory. Built a PyTorch classifier that tells cars, bikes, buses, and vans apart. ~4K training images.

### usage

```bash
pip install -r requirements.txt
# put images in data/train/{class}/ and data/val/{class}/
python src/train.py --epochs 20  # train from scratch
python src/train.py --use_resnet --epochs 10  # transfer learning (better)
python src/predict.py --image path/to/img.jpg
```

### structure

```
src/
  data_loader.py   — load & prep images
  model.py         — CNN + ResNet
  train.py         — training loop
  predict.py       — run inference
data/              — your dataset here
```

Batch size matters more than I thought. Data augmentation is basically cheating (the good kind). Transfer learning with ResNet-18 blows training from scratch out of the water. Most of the time was spent debugging, not building. That's just ML.
