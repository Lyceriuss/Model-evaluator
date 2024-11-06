# config.py
import os
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

root_dir = os.path.join(os.getcwd(), "data", "circular_fashion_v1_extracted")
