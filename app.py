# app.py
from torchvision import transforms
import streamlit as st
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from model_evaluation import get_datasets, evaluate_model

def main():
    st.title("Clothing Classification Model Evaluation")

    # Sidebar for model selection
    st.sidebar.title("Model Selection")
    model_options = ['resnet50', 'efficientnet_b0']
    model_name = st.sidebar.selectbox('Select Model', model_options)

    # Device selection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

    st.header(f"Evaluation Results for {model_name}")

    st.write(f"**Test Loss:** {test_loss:.4f}")
    st.write(f"**Test Accuracy:** {test_accuracy:.2f}%")

    # Display classification report
    st.subheader("Classification Report")
    st.text(str(report))

    # Plot and display confusion matrix
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(include_values=True, cmap='Blues', ax=ax, xticks_rotation='vertical')
    plt.title(f'Confusion Matrix ({model_name})')
    st.pyplot(fig)

if __name__ == '__main__':
    main()
