# components/comparison.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from models.model_comparer import ModelComparer
from components.graphs import plot_comparison_chart

class ComparisonHandler:
    def __init__(self, model_names, evaluation_dir='.'):
        """
        Initializes the ComparisonHandler with the specified models.

        Args:
            model_names (list): List of model names to compare.
            evaluation_dir (str): Directory where evaluation results are stored.
        """
        self.comparer = ModelComparer(model_names=model_names, evaluation_dir=evaluation_dir)
        self.comparison_df = self.comparer.generate_comparison_table()

    def get_comparison_table(self):
        """
        Retrieves the comparison table DataFrame.

        Returns:
            DataFrame: Comparison metrics for all models.
        """
        return self.comparison_df

    def get_effectiveness_df(self):
        """
        Calculates and retrieves the effectiveness improvement DataFrame.

        Returns:
            DataFrame: Effectiveness improvement metrics.
        """
        # The base model is selected in the Streamlit app, so pass it as an argument
        # This method is adjusted in the Streamlit app to accept the base model
        pass

    def get_comparison_chart(self, metric='Test Accuracy (%)'):
        """
        Generates and retrieves the comparison chart.

        Args:
            metric (str): The metric to plot.

        Returns:
            matplotlib.pyplot.Figure: The comparison chart figure.
        """
        return self.comparer.generate_comparison_chart(metric=metric)
