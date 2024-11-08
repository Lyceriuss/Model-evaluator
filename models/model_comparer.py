# models/model_comparer.py

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class ModelComparer:
    def __init__(self, model_names, evaluation_dir='.'):
        """
        Initializes the ModelComparer with the specified models.

        Args:
            model_names (list): List of model names to compare.
            evaluation_dir (str): Directory where evaluation results are stored.
        """
        self.model_names = model_names
        self.evaluation_dir = evaluation_dir
        self.comparison_df = pd.DataFrame()

    def load_evaluation_results(self):
        """
        Loads evaluation results for all specified models.

        Returns:
            DataFrame: Comparison metrics for all models.
        """
        data = []
        for model in self.model_names:
            results_path = os.path.join(self.evaluation_dir, f'evaluation_results_{model}.pkl')
            if not os.path.isfile(results_path):
                print(f"Warning: Evaluation results for '{model}' not found at {results_path}. Skipping.")
                continue
            evaluation = joblib.load(results_path)
            data.append({
                'Model': model,
                'Test Loss': evaluation['test_loss'],
                'Test Accuracy (%)': evaluation['test_accuracy']
            })
        self.comparison_df = pd.DataFrame(data)
        return self.comparison_df

    def generate_comparison_table(self):
        """
        Generates a comparison table for all models.

        Returns:
            DataFrame: Comparison metrics for all models.
        """
        if self.comparison_df.empty:
            self.load_evaluation_results()
        return self.comparison_df

    def generate_comparison_chart(self, metric='Test Accuracy (%)'):
        """
        Generates a bar chart comparing a specific metric across models.

        Args:
            metric (str): The metric to plot.

        Returns:
            matplotlib.pyplot.Figure: The comparison chart figure.
        """
        if self.comparison_df.empty:
            self.load_evaluation_results()

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Model', y=metric, data=self.comparison_df, palette='viridis')
        plt.title(f'Model Comparison - {metric}')
        plt.ylabel(metric)
        plt.xlabel('Model')

        # Set y-axis limit based on metric
        if metric == 'Test Loss':
            plt.ylim(0, 1)  # Fixed range for Test Loss
        else:
            metric_values = self.comparison_df[metric]
            plt.ylim(0, max(metric_values) * 1.1)  # Dynamic range for other metrics

        plt.tight_layout()
        return plt

    def calculate_effectiveness(self, base_model_name):
        """
        Calculates the effectiveness improvement of each model compared to the specified base model.

        Args:
            base_model_name (str): The model to compare others against.

        Returns:
            DataFrame: Effectiveness improvement metrics.
        """
        if self.comparison_df.empty:
            self.load_evaluation_results()

        if base_model_name not in self.comparison_df['Model'].values:
            print(f"Base model '{base_model_name}' not found in comparison data.")
            return pd.DataFrame()

        base_accuracy = self.comparison_df.loc[self.comparison_df['Model'] == base_model_name, 'Test Accuracy (%)'].values[0]
        self.comparison_df['Accuracy Improvement (%)'] = self.comparison_df['Test Accuracy (%)'] - base_accuracy
        return self.comparison_df
