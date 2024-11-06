# evaluate.py

import os
import json
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np  
import torch.nn as nn

from my_package import ClothingDataset, ClothingModel
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def main():

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
    
    # Initialize dataset
    dataset = ClothingDataset(root_dir=root_dir, transform=transform)
    print(f"Number of samples in dataset: {len(dataset)}")
    
    # Split dataset into training and test sets
    dataset_length = len(dataset)
    test_size = int(0.1 * dataset_length)  # 10% for test
    train_size = dataset_length - test_size

    # Ensure reproducible split
    generator = torch.Generator().manual_seed(seed)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=generator)
    
    # Create DataLoader for test set
    batch_size = 32
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=ClothingDataset.custom_collate)
    
    # Get label mappings
    num_classes = len(dataset.labels_set)
    label_to_index = dataset.label_to_index
    index_to_label = dataset.index_to_label
    print("Number of classes:", num_classes)
    print("Label to index mapping:", label_to_index)
    
    # Initialize model
    model = ClothingModel(num_classes=num_classes).to(device)

    # Load saved model
    model_path = 'resnet50_clothing_model_front_only.pth'  # Update with your model file name

    # Suppress FutureWarning if necessary
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")

    # Evaluation
    criterion = nn.CrossEntropyLoss()
    test_running_loss = 0.0
    test_correct = 0
    test_total = 0

    # Lists to store predictions and labels for confusion matrix
    all_preds = []
    all_labels = []

    # Open testing log file
    with open('testing_log.txt', 'w') as test_log_file:
        test_log_file.write('Image_Path\tPredicted_Label\tActual_Label\tCorrect\n')  # Header

        with torch.no_grad():
            for images, labels, image_paths in tqdm(test_loader, desc="Evaluating"):
                if images.size(0) == 0:
                    continue  # Skip empty batches
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                test_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

                # Collect all predictions and labels
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # Logging per batch
                for i in range(labels.size(0)):
                    img_path = image_paths[i]
                    file_name = os.path.basename(img_path)  # To reduce the size of the path
                    pred_label_idx = predicted[i].item()
                    actual_label_idx = labels[i].item()

                    pred_label = index_to_label.get(pred_label_idx, f"Unknown_{pred_label_idx}")
                    actual_label = index_to_label.get(actual_label_idx, f"Unknown_{actual_label_idx}")
                    is_correct = pred_label_idx == actual_label_idx
                    # Write to testing log file
                    test_log_file.write(f"{file_name}\t{pred_label}\t{actual_label}\t{is_correct}\n")

    test_loss = test_running_loss / len(test_loader)
    test_accuracy = 100 * test_correct / test_total
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    # Optionally, plot confusion matrix or other evaluation metrics
    # For example:

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12,10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=index_to_label.values(), yticklabels=index_to_label.values())
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Print classification report
    print(classification_report(all_labels, all_preds, target_names=list(index_to_label.values())))

if __name__ == '__main__':
    main()
