# model_evaluation.py

import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import torch.nn as nn
import joblib  # For potential caching needs

from my_package import ClothingDataset, ClothingModel

def get_datasets(root_dir, transform, seed=42):
    """
    Load and split the dataset into training and test sets.
    """
    # Initialize dataset
    dataset = ClothingDataset(root_dir=root_dir, transform=transform)
    
    # Split dataset into training and test sets
    dataset_length = len(dataset)
    test_size = int(0.1 * dataset_length)  # 10% for test
    train_size = dataset_length - test_size

    # Ensure reproducible split
    generator = torch.Generator().manual_seed(seed)
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size], generator=generator)

    return train_dataset, test_dataset, dataset

def evaluate_model(model_name, model_path, _test_dataset, batch_size=32, device='cpu'):
    """
    Evaluate the specified model on the test dataset.
    The '_test_dataset' parameter is ignored by Streamlit's caching.
    """
    from sklearn.metrics import classification_report, confusion_matrix

    # Path to precomputed results
    results_path = f'evaluation_results_{model_name}.pkl'
    if os.path.isfile(results_path):
        return joblib.load(results_path)
    
    # Check if the model file exists
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")

    # Set device
    device = torch.device(device)

    # Get label mappings
    num_classes = len(_test_dataset.dataset.labels_set)
    index_to_label = _test_dataset.dataset.index_to_label

    # Initialize model
    model = ClothingModel(num_classes=num_classes, model_name=model_name).to(device)

    # Load saved model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # DataLoader for test set
    test_loader = DataLoader(
        _test_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
        collate_fn=ClothingDataset.custom_collate)

    # Evaluation
    criterion = nn.CrossEntropyLoss()
    test_running_loss = 0.0
    test_correct = 0
    test_total = 0

    # Lists to store predictions and labels
    all_preds = []
    all_labels = []
    all_image_paths = []

    with torch.no_grad():
        for images, labels, image_paths in test_loader:
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
            all_image_paths.extend(image_paths)

    test_loss = test_running_loss / len(test_loader)
    test_accuracy = 100 * test_correct / test_total

    # Get the sorted list of label indices
    label_indices = sorted(index_to_label.keys())
    # Ensure target_names corresponds to label_indices
    target_names = [index_to_label[i] for i in label_indices]

    # Generate classification report
    report = classification_report(
        all_labels,
        all_preds,
        labels=label_indices,
        target_names=target_names,
        zero_division=0,
        output_dict=True  # Return as dict for easier handling
    )

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=label_indices)

    # Prepare evaluation results
    evaluation_results = {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'classification_report': report,
        'confusion_matrix': cm,
        'label_indices': label_indices,
        'target_names': target_names,
        'all_preds': all_preds,
        'all_labels': all_labels,
        'all_image_paths': all_image_paths
    }

    # Save evaluation results
    joblib.dump(evaluation_results, results_path)

    return evaluation_results
