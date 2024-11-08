# main.py

import os
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

from data.data_loader import DataLoaderManager
from models.model import ClothingModel

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train Clothing Classification Model')
    parser.add_argument('--model', type=str, default='resnet50', choices=['resnet18','resnet50', 'efficientnet_b0', 'densenet121','vit-base-patch16-224', 'microsoft/beit-base-patch16-224-pt22k-ft22k'], help='Model architecture to use')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use for training')
    parser.add_argument('--data-dir', type=str, default='data/circular_fashion_v1_extracted', help='Path to the data directory')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=0.0001, help='Learning rate for optimizer')
    args = parser.parse_args()
    model_name = args.model
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    print(f"Using device: {device}")

    # Define trained_models directory
    trained_models_dir = os.path.join(os.getcwd(), 'trained_models')
    os.makedirs(trained_models_dir, exist_ok=True)

    # Initialize DataLoaderManager
    data_manager = DataLoaderManager(
        root_dir=os.path.join(os.getcwd(), args.data_dir),
        test_split=0.1,
        batch_size=batch_size,
        num_workers=4
    )

    # Initialize model
    num_classes = len(data_manager.dataset.labels_set)
    model = ClothingModel(num_classes=num_classes, model_name=model_name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training parameters
    num_epochs = epochs

    # Lists to store loss and accuracy
    train_loss_per_epoch = []
    train_accuracy_per_epoch = []

    # Define paths for log and performance plot
    training_log_path = os.path.join(trained_models_dir, f'training_log_{model_name}.txt')
    training_performance_path = os.path.join(trained_models_dir, f'training_performance_{model_name}.png')

    # Open log file
    with open(training_log_path, 'w') as log_file:
        log_file.write('Image_Path\tPredicted_Label\tActual_Label\tCorrect\n')  # Header

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            log_file.write(f'Epoch {epoch+1}\n')
            for images, labels, image_paths in tqdm(data_manager.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                if images is None:
                    continue  # Skip empty batches
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

                    pred_label = data_manager.dataset.index_to_label.get(pred_label_idx, f"Unknown_{pred_label_idx}")
                    actual_label = data_manager.dataset.index_to_label.get(actual_label_idx, f"Unknown_{actual_label_idx}")
                    is_correct = pred_label_idx == actual_label_idx
                    # Write to log file
                    log_file.write(f"{file_name}\t{pred_label}\t{actual_label}\t{is_correct}\n")

            epoch_loss = running_loss / len(data_manager.train_loader)
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
    plt.savefig(training_performance_path)
    plt.show()

    # Save the trained model
    model_save_path = os.path.join(trained_models_dir, f'{model_name}_clothing_model.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved successfully as {model_save_path}.")

if __name__ == '__main__':
    main()
