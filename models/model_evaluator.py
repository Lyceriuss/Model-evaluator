# models/model_evaluator.py

import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import joblib  # For saving evaluation results

from sklearn.metrics import classification_report, confusion_matrix

from .model import ClothingModel  # Relative import

class ModelEvaluator:
    def __init__(self, model_name, model_path, test_loader, device='cpu'):
        """
        Initializes the ModelEvaluator with the specified model and data.

        Args:
            model_name (str): Name of the model ('resnet50', 'efficientnet_b0', 'densenet121').
            model_path (str): Path to the saved model file.
            test_loader (DataLoader): DataLoader for the test dataset.
            device (str): Device to perform computations on ('cuda' or 'cpu').
        """
        self.model_name = model_name
        self.model_path = model_path
        self.test_loader = test_loader
        self.device = torch.device(device)
        self.evaluation_results = {}
        self.model = self.load_model()

    def load_model(self):
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"Model file '{self.model_path}' not found.")

        # Initialize model
        num_classes = len(self.test_loader.dataset.dataset.labels_set)
        model = ClothingModel(num_classes=num_classes, model_name=self.model_name).to(self.device)

        # Load model state
        model.load_state_dict(torch.load(self.model_path, map_location=self.device)) 

        model.eval()
        return model

    def evaluate(self):
        """
        Evaluates the model on the test dataset.

        Returns:
            dict: Evaluation metrics including loss, accuracy, classification report, and confusion matrix.
        """
        criterion = nn.CrossEntropyLoss()
        running_loss = 0.0
        correct = 0
        total = 0

        all_preds = []
        all_labels = []
        all_image_paths = []

        with torch.no_grad():
            for images, labels, image_paths in self.test_loader:
                if images is None:
                    continue  # Skip empty batches
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Collect all predictions and labels
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_image_paths.extend(image_paths)

        test_loss = running_loss / len(self.test_loader)
        test_accuracy = 100 * correct / total

        # Retrieve label names
        label_indices = sorted(self.test_loader.dataset.dataset.label_to_index.values())
        target_names = [self.test_loader.dataset.dataset.index_to_label[i] for i in label_indices]

        report = classification_report(
            all_labels,
            all_preds,
            labels=label_indices,
            target_names=target_names,
            zero_division=0,
            output_dict=True
        )

        cm = confusion_matrix(all_labels, all_preds, labels=label_indices)

        self.evaluation_results = {
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

        return self.evaluation_results

    def save_results(self, save_path):
        """
        Saves the evaluation results to a pickle file.

        Args:
            save_path (str): Path to save the evaluation results.
        """
        joblib.dump(self.evaluation_results, save_path)
        print(f"Evaluation results saved to {save_path}")
