# components/graphs.py

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

def plot_confusion_matrix(cm, labels, title='Confusion Matrix'):
    """
    Plots the confusion matrix.

    Args:
        cm (array): Confusion matrix.
        labels (list): List of label names.
        title (str): Title of the plot.

    Returns:
        matplotlib.pyplot.Figure: The confusion matrix figure.
    """
    plt.figure(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(include_values=True, cmap='Blues', xticks_rotation='vertical', values_format='d')
    plt.title(title)
    plt.tight_layout()
    return plt

def plot_comparison_chart(comparison_df, metric='Test Accuracy (%)', title='Model Comparison'):
    """
    Plots a bar chart comparing a specific metric across models.

    Args:
        comparison_df (DataFrame): DataFrame containing comparison metrics.
        metric (str): The metric to plot.
        title (str): Title of the plot.

    Returns:
        matplotlib.pyplot.Figure: The comparison chart figure.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y=metric, data=comparison_df, palette='viridis')
    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel('Model')
    plt.ylim(0, 100)
    plt.tight_layout()
    return plt
