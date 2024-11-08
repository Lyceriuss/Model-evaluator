# streamlit_app/app.py

import sys
import os

# Define the project root directory
project_root = 'C:/Users/Delic/Desktop/Quincy-AI/Kodning/Resnet-50'

# Add the project root to sys.path if it's not already there
if project_root not in sys.path:
    sys.path.append(project_root)

import streamlit as st
import torch
from PIL import Image
import io

from data import DataLoaderManager
from models import ModelComparer
from components import plot_confusion_matrix, plot_comparison_chart, ComparisonHandler

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def main():
    # Configure the Streamlit page
    st.set_page_config(page_title="Clothing Classification Dashboard", layout="wide")
    st.title("🧥 Clothing Classification Model Evaluation Dashboard")

    # Define trained_models directory
    trained_models_dir = os.path.join(project_root, 'trained_models')

    # Discover available models based on evaluation result files in trained_models/
    model_files = [f for f in os.listdir(trained_models_dir) if f.startswith('evaluation_results_') and f.endswith('.pkl')]
    model_options = [f.replace('evaluation_results_', '').replace('.pkl', '') for f in model_files]

    # Sort the model_options for better user experience
    model_options = sorted(model_options)

    if not model_options:
        st.warning("No evaluation results found in `trained_models/`. Please evaluate models using `evaluate.py` before using the dashboard.")
        st.stop()

    # Sidebar configuration
    st.sidebar.title("🔧 Configuration")
    # Ensure 'All Models Comparison' is the first option, followed by individual models
    selected_option = st.sidebar.selectbox('Select Option', ['All Models Comparison'] + model_options)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    st.sidebar.write(f"**Device:** {device.upper()}")

    # Conditional rendering based on selected_option
    if selected_option == 'All Models Comparison':
        st.header("📊 All Models Comparison")

        # Initialize ComparisonHandler with trained_models_dir
        comparison_handler = ComparisonHandler(model_names=model_options, evaluation_dir=trained_models_dir)

        # Get comparison table
        comparison_table = comparison_handler.get_comparison_table()

        if comparison_table.empty:
            st.write("No evaluation results found for the selected models.")
        else:
            # Display comparison table
            st.subheader("Comparison Metrics")
            min_loss_model = comparison_table.loc[comparison_table['Test Loss'].idxmin(), 'Model']
            max_accuracy_model = comparison_table.loc[comparison_table['Test Accuracy (%)'].idxmax(), 'Model']

            styled_comparison_table = comparison_table.style \
            .highlight_min(subset=['Test Loss'], color='yellow') \
            .highlight_max(subset=['Test Accuracy (%)'], color='darkblue') \
            .apply(lambda x: ['background-color: darkblue' if x['Model'] == min_loss_model or x['Model'] == max_accuracy_model else '' for i in x], axis=1)

            st.dataframe(styled_comparison_table)

            # Generate and display Test Accuracy comparison chart
            comparison_chart = comparison_handler.comparer.generate_comparison_chart(metric='Test Accuracy (%)')
            st.subheader("📈 Test Accuracy Comparison")
            st.pyplot(comparison_chart)

            # Download Test Accuracy comparison chart
            buf = io.BytesIO()
            comparison_chart.savefig(buf, format='png')
            buf.seek(0)
            st.download_button(
                label="💾 Download Test Accuracy Comparison Chart (PNG)",
                data=buf,
                file_name='test_accuracy_comparison_chart.png',
                mime='image/png'
            )

            # Generate and display Test Loss comparison chart
            loss_comparison_chart = comparison_handler.comparer.generate_comparison_chart(metric='Test Loss')
            st.subheader("📉 Test Loss Comparison")
            st.pyplot(loss_comparison_chart)

            # Download Test Loss comparison chart
            buf_loss = io.BytesIO()
            loss_comparison_chart.savefig(buf_loss, format='png')
            buf_loss.seek(0)
            st.download_button(
                label="💾 Download Test Loss Comparison Chart (PNG)",
                data=buf_loss,
                file_name='test_loss_comparison_chart.png',
                mime='image/png'
            )

            # Allow user to select base model for effectiveness comparison
            base_model = st.selectbox(
                'Select Base Model for Effectiveness Comparison',
                options=model_options,
                index=0  # Default to the first model
            )
            effectiveness_df = comparison_handler.comparer.calculate_effectiveness(base_model)

            # Display effectiveness table
            st.subheader("📊 Effectiveness Improvement")
            st.dataframe(effectiveness_df[['Model', 'Accuracy Improvement (%)']].set_index('Model').round(2))

            # Generate and display effectiveness chart
            plt.figure(figsize=(10, 6))
            sns.barplot(
                x='Model',
                y='Accuracy Improvement (%)',
                data=effectiveness_df,
                palette='viridis'
            )
            plt.title('Model Accuracy Improvement (%)')
            plt.ylabel('Accuracy Improvement (%)')
            plt.xlabel('Model')
            plt.tight_layout()
            effectiveness_chart = plt.gcf()
            st.pyplot(effectiveness_chart)

            # Download effectiveness chart
            buf_eff = io.BytesIO()
            effectiveness_chart.savefig(buf_eff, format='png')
            buf_eff.seek(0)
            st.download_button(
                label="💾 Download Effectiveness Chart (PNG)",
                data=buf_eff,
                file_name='effectiveness_chart.png',
                mime='image/png'
            )

            # Display effectiveness percentages
            st.subheader("📈 Effectiveness Percentage")
            st.write("Shows how much more effective each model is compared to the base model.")

            # Display the effectiveness DataFrame
            st.dataframe(effectiveness_df[['Model', 'Accuracy Improvement (%)']].set_index('Model').round(2))

    else:
        # Individual Model Evaluation
        model_name = selected_option
        model_path = os.path.join(trained_models_dir, f'{model_name}_clothing_model.pth')

        # Check if model file exists
        if not os.path.isfile(model_path):
            st.error(f"Model file '{model_path}' not found. Please train the model before evaluation.")
            st.stop()

        # Load evaluation results
        evaluation_results_path = os.path.join(trained_models_dir, f'evaluation_results_{model_name}.pkl')
        if not os.path.isfile(evaluation_results_path):
            st.warning(f"Evaluation results for '{model_name}' not found. Please evaluate the model using `evaluate.py`.")
            st.stop()

        try:
            evaluation_results = joblib.load(evaluation_results_path)
        except Exception as e:
            st.error(f"Failed to load evaluation results for '{model_name}': {e}")
            st.stop()

        # Display evaluation metrics
        st.header(f"📊 Evaluation Results for **{model_name.upper()}**")
        st.write(f"**Test Loss:** {evaluation_results['test_loss']:.4f}")
        st.write(f"**Test Accuracy:** {evaluation_results['test_accuracy']:.2f}%")

        # Display classification report
        st.subheader("📈 Classification Report")
        try:
            report_df = pd.DataFrame(evaluation_results['classification_report']).transpose()
            st.dataframe(report_df.style.highlight_max(axis=0))
        except Exception as e:
            st.error(f"Failed to display classification report: {e}")

        # Download classification report
        try:
            report_json = report_df.to_json()
            st.download_button(
                label="💾 Download Classification Report (JSON)",
                data=report_json,
                file_name=f"classification_report_{model_name}.json",
                mime="application/json"
            )
        except Exception as e:
            st.error(f"Failed to create download button for classification report: {e}")

        # Display confusion matrix
        st.subheader("🌀 Confusion Matrix")
        try:
            cm_plot = plot_confusion_matrix(
                cm=evaluation_results['confusion_matrix'],
                labels=evaluation_results['target_names'],
                title=f'Confusion Matrix ({model_name.upper()})'
            )
            st.pyplot(cm_plot)
        except Exception as e:
            st.error(f"Failed to plot confusion matrix: {e}")

        # Download confusion matrix plot
        try:
            buf = io.BytesIO()
            cm_plot.savefig(buf, format='png')
            buf.seek(0)
            st.download_button(
                label="💾 Download Confusion Matrix (PNG)",
                data=buf,
                file_name=f'confusion_matrix_{model_name}.png',
                mime='image/png'
            )
        except Exception as e:
            st.error(f"Failed to create download button for confusion matrix: {e}")

if __name__ == '__main__':
    main()
