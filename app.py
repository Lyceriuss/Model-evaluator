# app.py

import streamlit as st
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.metrics import ConfusionMatrixDisplay
import pandas as pd  # For displaying classification report
from PIL import Image  # For image handling

from model_evaluation import get_datasets, evaluate_model

# Import necessary for caching
# Note: Streamlit's caching is now handled via `@st.cache_resource`
@st.cache_resource(show_spinner=False)
def load_evaluation_results(model_name, model_path, _test_dataset, device):
    """
    Load and evaluate the model, caching the results.
    """
    return evaluate_model(model_name, model_path, _test_dataset, device=device)

def main():
    st.title("🧥 Clothing Classification Model Evaluation Dashboard")

    # Sidebar for model selection
    st.sidebar.title("🔧 Configuration")
    model_options = ['resnet50', 'efficientnet_b0']
    model_name = st.sidebar.selectbox('Select Model', model_options)

    # Device selection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    st.sidebar.write(f"**Device:** {device.upper()}")

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

    # Path to the model
    model_path = f'{model_name}_clothing_model.pth'

    # Evaluate model with caching
    with st.spinner(f"Evaluating {model_name.upper()}..."):
        try:
            evaluation_results = load_evaluation_results(model_name, model_path, test_dataset, device=device)
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()

    # Extract evaluation results
    test_loss = evaluation_results['test_loss']
    test_accuracy = evaluation_results['test_accuracy']
    report = evaluation_results['classification_report']
    cm = evaluation_results['confusion_matrix']
    label_indices = evaluation_results['label_indices']
    target_names = evaluation_results['target_names']

    # Display evaluation metrics
    st.header(f"📊 Evaluation Results for **{model_name.upper()}**")
    st.write(f"**Test Loss:** {test_loss:.4f}")
    st.write(f"**Test Accuracy:** {test_accuracy:.2f}%")

    # Display classification report
    st.subheader("📈 Classification Report")
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.highlight_max(axis=0))

    # Plot and display confusion matrix
    st.subheader("🌀 Confusion Matrix")
    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(include_values=True, cmap='Blues', ax=ax, xticks_rotation='vertical', values_format='d')
    plt.title(f'Confusion Matrix ({model_name.upper()})')
    plt.tight_layout()
    st.pyplot(fig)

    # Display misclassified examples
    st.subheader("❌ Misclassified Examples")
    misclassified_indices = [
        i for i, (pred, actual) in enumerate(zip(evaluation_results['all_preds'], evaluation_results['all_labels']))
        if pred != actual
    ]
    if misclassified_indices:
        num_examples = min(5, len(misclassified_indices))
        example_indices = np.random.choice(misclassified_indices, num_examples, replace=False)
        for idx in example_indices:
            image_path = evaluation_results['all_image_paths'][idx]
            image_full_path = os.path.join(root_dir, image_path)
            if os.path.exists(image_full_path):
                try:
                    image = Image.open(image_full_path)
                    pred_label = target_names[evaluation_results['all_preds'][idx]]
                    actual_label = target_names[evaluation_results['all_labels'][idx]]
                    st.image(
                        image, 
                        caption=f"**Predicted:** {pred_label} | **Actual:** {actual_label}", 
                        use_column_width=True
                    )
                except Exception as e:
                    st.write(f"Error loading image {image_full_path}: {e}")
            else:
                st.write(f"Image not found: {image_full_path}")
    else:
        st.write("No misclassified examples found.")

if __name__ == '__main__':
    main()
