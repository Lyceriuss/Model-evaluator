# evaluate.py

import os
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from model_evaluation import get_datasets, evaluate_model

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Evaluate Clothing Classification Model')
    parser.add_argument('--model', type=str, default='resnet50', choices=['resnet50', 'efficientnet_b0'], help='Model architecture to evaluate')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (e.g., "cuda" or "cpu")')
    args = parser.parse_args()
    model_name = args.model
    device = args.device

    # Set random seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Data transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images
        transforms.ToTensor(),          # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize
                             std=[0.229, 0.224, 0.225])
    ])

    # Root directory for your data
    root_dir = os.path.join(os.getcwd(), "data", "circular_fashion_v1_extracted")

    # Get datasets
    _, test_dataset, _ = get_datasets(root_dir, transform, seed=seed)

    # Evaluate model
    model_path = f'{model_name}_clothing_model.pth'
    evaluation_results = evaluate_model(model_name, model_path, test_dataset, device=device)

    # Extract evaluation results
    test_loss = evaluation_results['test_loss']
    test_accuracy = evaluation_results['test_accuracy']
    report = evaluation_results['classification_report']
    cm = evaluation_results['confusion_matrix']
    label_indices = evaluation_results['label_indices']
    target_names = evaluation_results['target_names']

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
    print(f"Classification Report for {model_name}:\n", report)

    # Save classification report to a file
    with open(f'classification_report_{model_name}.txt', 'w') as f:
        f.write(str(report))

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    plt.figure(figsize=(12, 10))
    disp.plot(include_values=True, cmap='Blues', ax=plt.gca(), xticks_rotation='vertical')
    plt.title(f'Confusion Matrix ({model_name})')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name}.png')
    plt.show()

if __name__ == '__main__':
    main()
