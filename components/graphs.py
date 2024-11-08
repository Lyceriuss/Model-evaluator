# components/graphs.py

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay


def plot_confusion_matrix(cm, labels, title='Confusion Matrix', figsize=(15, 15), fontsize=10):
    """
    Plots a confusion matrix with better formatting.

    Args:
        cm (array): Confusion matrix values.
        labels (list): List of label names.
        title (str): Title of the plot.
        figsize (tuple): Size of the figure.
        fontsize (int): Font size for labels.
    """
    fig, ax = plt.subplots(figsize=figsize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(include_values=True, cmap='Blues', ax=ax, xticks_rotation='vertical', values_format='d')

    ax.set_title(title, fontsize=fontsize + 4)
    ax.set_xlabel('Predicted label', fontsize=fontsize + 2)
    ax.set_ylabel('True label', fontsize=fontsize + 2)
    plt.xticks(fontsize=fontsize, rotation=45, ha='right')
    plt.yticks(fontsize=fontsize)
    plt.tight_layout()

    plt.show()
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
