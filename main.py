# main.py

import os
import json
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np  
import argparse  # Added import

from my_package import ClothingDataset, ClothingModel

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train Clothing Classification Model')
    parser.add_argument('--model', type=str, default='resnet50', choices=['resnet50', 'efficientnet_b0'], help='Model architecture to use')
    args = parser.parse_args()
    model_name = args.model

    # Set random seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images
        transforms.ToTensor(),          # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Root directory for your data
    root_dir = os.path.join(os.getcwd(), "data", "circular_fashion_v1_extracted")
    
    # Initialize dataset and dataloader
    sample_size = None  # Use all data
    dataset = ClothingDataset(root_dir=root_dir, transform=transform, sample_size=sample_size)
    print(f"Number of samples in dataset: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=ClothingDataset.custom_collate)
    
    # Get label mappings
    num_classes = len(dataset.labels_set)
    label_to_index = dataset.label_to_index
    index_to_label = dataset.index_to_label
    print("Number of classes:", num_classes)
    print("Label to index mapping:", label_to_index)
    
    # Initialize model, loss function, optimizer
    model = ClothingModel(num_classes=num_classes, model_name=model_name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # Training parameters
    num_epochs = 5
    
    # Lists to store loss and accuracy
    train_loss_per_epoch = []
    train_accuracy_per_epoch = []
    
    with open(f'training_log_{model_name}.txt', 'w') as log_file:
        log_file.write('Image_Path\tPredicted_Label\tActual_Label\tCorrect\n')  # Header

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            log_file.write(f'Epoch {epoch+1}\n')
            for images, labels, image_paths in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Logging per batch
                for i in range(labels.size(0)):
                    img_path = image_paths[i]
                    file_name = os.path.basename(img_path)  # To reduce the size of the path
                    pred_label_idx = predicted[i].item()
                    actual_label_idx = labels[i].item()

                    pred_label = index_to_label.get(pred_label_idx, f"Unknown_{pred_label_idx}")
                    actual_label = index_to_label.get(actual_label_idx, f"Unknown_{actual_label_idx}")
                    is_correct = pred_label_idx == actual_label_idx
                    # Write to log file
                    log_file.write(f"{file_name}\t{pred_label}\t{actual_label}\t{is_correct}\n")

            epoch_loss = running_loss / len(dataloader)
            epoch_accuracy = 100 * correct / total

            # Append to the lists
            train_loss_per_epoch.append(epoch_loss)
            train_accuracy_per_epoch.append(epoch_accuracy)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
    
    # Plotting the results
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_loss_per_epoch, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss per Epoch ({model_name})')
    plt.legend()
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accuracy_per_epoch, label='Training Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Training Accuracy per Epoch ({model_name})')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'training_performance_{model_name}.png')
    plt.show()
    
    # Save the trained model
    model_save_path = f'{model_name}_clothing_model.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved successfully as {model_save_path}.")

if __name__ == '__main__':
    main()
