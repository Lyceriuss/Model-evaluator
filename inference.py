import torch
from torchvision import transforms
from PIL import Image
import json
import os

# Import your model class
from my_package.model import ClothingModel

def main():
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Number of classes (update based on your dataset)
    num_classes = 28  # Adjust this number accordingly

    # Initialize the model
    model = ClothingModel(num_classes=num_classes)
    model.load_state_dict(torch.load('resnet50_clothing_model_front_only.pth', map_location=device))
    model.to(device)
    model.eval()

    # Load label mappings
    with open('index_to_label.json', 'r') as f:
        index_to_label = json.load(f)

    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Path to your test image
    test_image_path = 'path/to/your/test_image.jpg'  # Update this path

    # Predict the label
    predicted_label = predict_image(test_image_path, model, transform, device, index_to_label)
    print(f"The predicted label is: {predicted_label}")

def predict_image(image_path, model, transform, device, index_to_label):
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_label = index_to_label[str(predicted.item())]

    return predicted_label

if __name__ == '__main__':
    main()
