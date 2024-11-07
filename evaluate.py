# evaluate.py

import argparse
import os
from data.data_loader import DataLoaderManager
from models.model_evaluator import ModelEvaluator
from components.graphs import plot_confusion_matrix
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='Evaluate Clothing Classification Model')
    parser.add_argument('--model', type=str, required=True, choices=['resnet50', 'efficientnet_b0', 'densenet121'], help='Model architecture to evaluate')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (e.g., "cuda" or "cpu")')
    parser.add_argument('--data-dir', type=str, default='data/circular_fashion_v1_extracted', help='Path to the data directory')
    args = parser.parse_args()

    # Define trained_models directory
    project_root = os.getcwd()
    trained_models_dir = os.path.join(project_root, 'trained_models')

    # Path to the model within trained_models/
    model_path = os.path.join(trained_models_dir, f'{args.model}_clothing_model.pth')

    # Initialize DataLoaderManager
    data_manager = DataLoaderManager(root_dir=args.data_dir)

    # Initialize and run ModelEvaluator
    evaluator = ModelEvaluator(
        model_name=args.model,
        model_path=model_path,
        test_loader=data_manager.test_loader,
        device=args.device
    )
    evaluation_results = evaluator.evaluate()

    # Save evaluation results to trained_models/
    evaluation_results_path = os.path.join(trained_models_dir, f'evaluation_results_{args.model}.pkl')
    evaluator.save_results(evaluation_results_path)

    # Save test log similar to training_log
    test_log_path = os.path.join(trained_models_dir, f'test_log_{args.model}.txt')
    with open(test_log_path, 'w') as log_file:
        log_file.write('Image_Path\tPredicted_Label\tActual_Label\tCorrect\n')  # Header
        for img_path, pred, actual, correct in zip(evaluation_results['all_image_paths'],
                                                   evaluation_results['all_preds'],
                                                   evaluation_results['all_labels'],
                                                   [p == a for p, a in zip(evaluation_results['all_preds'], evaluation_results['all_labels'])]):
            file_name = os.path.basename(img_path)
            pred_label = data_manager.dataset.index_to_label.get(pred, f"Unknown_{pred}")
            actual_label = data_manager.dataset.index_to_label.get(actual, f"Unknown_{actual}")
            log_file.write(f"{file_name}\t{pred_label}\t{actual_label}\t{correct}\n")
    print(f"Test log saved to {test_log_path}")

    # Display evaluation metrics
    print(f"Test Loss: {evaluation_results['test_loss']:.4f}, Test Accuracy: {evaluation_results['test_accuracy']:.2f}%")
    print(f"Classification Report for {args.model}:\n", evaluation_results['classification_report'])

    # Plot and save confusion matrix in trained_models/
    cm_plot = plot_confusion_matrix(
        cm=evaluation_results['confusion_matrix'],
        labels=evaluation_results['target_names'],
        title=f'Confusion Matrix ({args.model.upper()})'
    )
    confusion_matrix_path = os.path.join(trained_models_dir, f'confusion_matrix_{args.model}.png')
    cm_plot.savefig(confusion_matrix_path)
    plt.show()

    print(f"Confusion matrix saved to {confusion_matrix_path}")
    print(f"Evaluation results saved to {evaluation_results_path}")

if __name__ == '__main__':
    main()
