# evaluate.py

import argparse
from data.data_loader import DataLoaderManager
from models.model_evaluator import ModelEvaluator
from components.graphs import plot_confusion_matrix
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='Evaluate Clothing Classification Model')
    parser.add_argument('--model', type=str, required=True, choices=['resnet50', 'efficientnet_b0'], help='Model architecture to evaluate')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (e.g., "cuda" or "cpu")')
    parser.add_argument('--data-dir', type=str, default='data/circular_fashion_v1_extracted', help='Path to the data directory')
    args = parser.parse_args()

    # Initialize DataLoaderManager
    data_manager = DataLoaderManager(root_dir=args.data_dir)

    # Path to the model
    model_path = f'{args.model}_clothing_model.pth'

    # Initialize and run ModelEvaluator
    evaluator = ModelEvaluator(
        model_name=args.model,
        model_path=model_path,
        test_loader=data_manager.test_loader,
        device=args.device
    )

    evaluation_results = evaluator.evaluate()

    # Display evaluation metrics
    print(f"Test Loss: {evaluation_results['test_loss']:.4f}, Test Accuracy: {evaluation_results['test_accuracy']:.2f}%")
    print(f"Classification Report for {args.model}:\n", evaluation_results['classification_report'])

    # Plot and save confusion matrix
    cm_plot = plot_confusion_matrix(
        cm=evaluation_results['confusion_matrix'],
        labels=evaluation_results['target_names'],
        title=f'Confusion Matrix ({args.model.upper()})'
    )
    cm_plot.savefig(f'confusion_matrix_{args.model}.png')
    plt.show()

if __name__ == '__main__':
    main()
