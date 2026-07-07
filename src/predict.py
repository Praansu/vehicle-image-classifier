import argparse
import torch
from PIL import Image
import torchvision.transforms as transforms

from model import VehicleCNN, create_resnet_model


CLASSES = ['car', 'bike', 'bus', 'van']


def predict(image_path, model_path, use_resnet=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if use_resnet:
        model = create_resnet_model(num_classes=len(CLASSES))
    else:
        model = VehicleCNN(num_classes=len(CLASSES))

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = output.max(1)

    class_name = CLASSES[predicted.item()]
    confidence = torch.softmax(output, dim=1)[0][predicted].item()

    print(f'Prediction: {class_name} ({confidence:.2%} confidence)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--model', default='vehicle_model.pth')
    parser.add_argument('--use_resnet', action='store_true')
    args = parser.parse_args()

    predict(args.image, args.model, args.use_resnet)
