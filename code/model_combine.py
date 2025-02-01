import os
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from packages.utils.obj_state import load_state


class ModelCombine:
    def __init__(self, model_folder_path):
        self.model_folder_path = model_folder_path

    def combine(
        self,
        scores_csv_name="export_scores.csv",
        csv_output_path=None,
    ):
        self.df_scores = pd.DataFrame()
        self.df_features = pd.DataFrame()

        if csv_output_path is not None:
            scores_csv_path = os.path.join(csv_output_path, scores_csv_name)
        else:
            scores_csv_path = os.path.join(self.model_folder_path, scores_csv_name)

        if os.path.exists(scores_csv_path):
            os.remove(scores_csv_path)

        self.result_paths = [
            file for file in os.listdir(self.model_folder_path) if file.endswith(".pkl")
        ]

        self.columns = []

        self.csv_rows = []
        self.full_rows = []
        for result_path in self.result_paths:
            model_result = load_state(os.path.join(self.model_folder_path, result_path))

            # Remove the .csv extension from the column name
            row_name = result_path[
                :-12
            ]  # This removes the last 4 characters, which should be '_opt_res.pkl'

            row = {
                "model_name": row_name,
                "cv_scores_mean": model_result["cv_scores"].mean(),
                "cv_scores_std": model_result["cv_scores"].std(),
                "train_rmse": model_result["results_df"]
                .loc[model_result["results_df"]["Metric"] == "RMSE", "Training Set"]
                .values[0],
                "train_mape": model_result["results_df"]
                .loc[model_result["results_df"]["Metric"] == "MAPE", "Training Set"]
                .values[0],
                "train_r2": model_result["results_df"]
                .loc[model_result["results_df"]["Metric"] == "R²", "Training Set"]
                .values[0],
                "train_mae": model_result["results_df"]
                .loc[model_result["results_df"]["Metric"] == "MAE", "Training Set"]
                .values[0],
                "train_mse": model_result["results_df"]
                .loc[model_result["results_df"]["Metric"] == "MSE", "Training Set"]
                .values[0],
                "test_rmse": model_result["results_df"]
                .loc[model_result["results_df"]["Metric"] == "RMSE", "Test Set"]
                .values[0],
                "test_mape": model_result["results_df"]
                .loc[model_result["results_df"]["Metric"] == "MAPE", "Test Set"]
                .values[0],
                "test_r2": model_result["results_df"]
                .loc[model_result["results_df"]["Metric"] == "R²", "Test Set"]
                .values[0],
                "test_mae": model_result["results_df"]
                .loc[model_result["results_df"]["Metric"] == "MAE", "Test Set"]
                .values[0],
                "test_mse": model_result["results_df"]
                .loc[model_result["results_df"]["Metric"] == "MSE", "Test Set"]
                .values[0],
            }
            model_result["model_name"] = row_name
            self.csv_rows.append(row)
            self.full_rows.append(model_result)

        self.evaluation_df = pd.DataFrame(self.csv_rows)

        self.evaluation_df.index = self.evaluation_df.index + 1
        self.evaluation_df.to_csv(scores_csv_path, index=True, encoding="utf-8-sig")

    def plot(self):
        models = self.evaluation_df["model_name"]

        fig, axs = plt.subplots(2, 3, figsize=(24, 16))

        def plot_bar_with_limits(ax, x, y, yerr=None, threshold=100, **kwargs):
            y_limited = np.clip(y, -threshold, threshold)
            bars = ax.bar(x, y_limited, yerr=yerr, **kwargs)

            for i, (value, bar) in enumerate(zip(y, bars)):
                if abs(value) > threshold:
                    if abs(value) > 1000:
                        value_str = f"{value:.2e}" 
                    else:
                        value_str = f"{value:.2f}"

                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        np.sign(value) * threshold,
                        value_str,
                        ha="center",
                        va="bottom" if value > 0 else "top",
                    )

            return bars

        # Plot 1: CV scores comparison
        plot_bar_with_limits(
            axs[0, 0],
            models,
            -self.evaluation_df["cv_scores_mean"] * 100.0,
            yerr=self.evaluation_df["cv_scores_std"] * 100.0,
            capsize=4,
        )
        axs[0, 0].set_xlabel("Models")
        axs[0, 0].set_ylabel("CV Scores")
        axs[0, 0].set_title("CV Scores Comparison")
        axs[0, 0].grid(True)

        # Plot 2: RMSE comparison
        x = np.arange(len(models))
        width = 0.35
        plot_bar_with_limits(
            axs[0, 1],
            x - width / 2,
            self.evaluation_df["train_rmse"],
            width=width,
            label="Train",
        )
        plot_bar_with_limits(
            axs[0, 1],
            x + width / 2,
            self.evaluation_df["test_rmse"],
            width=width,
            label="Test",
        )
        axs[0, 1].set_xlabel("Models")
        axs[0, 1].set_ylabel("RMSE")
        axs[0, 1].set_title("RMSE Comparison")
        axs[0, 1].set_xticks(x)
        axs[0, 1].set_xticklabels(models)
        axs[0, 1].legend()
        axs[0, 1].grid(True)

        # Plot 3: MAPE comparison
        plot_bar_with_limits(
            axs[0, 2],
            x - width / 2,
            self.evaluation_df["train_mape"],
            width=width,
            label="Train",
        )
        plot_bar_with_limits(
            axs[0, 2],
            x + width / 2,
            self.evaluation_df["test_mape"],
            width=width,
            label="Test",
        )
        axs[0, 2].set_xlabel("Models")
        axs[0, 2].set_ylabel("MAPE")
        axs[0, 2].set_title("MAPE Comparison")
        axs[0, 2].set_xticks(x)
        axs[0, 2].set_xticklabels(models)
        axs[0, 2].legend()
        axs[0, 2].grid(True)

        # Plot 4: R2 comparison
        plot_bar_with_limits(
            axs[1, 0],
            x - width / 2,
            self.evaluation_df["train_r2"],
            width=width,
            label="Train",
        )
        plot_bar_with_limits(
            axs[1, 0],
            x + width / 2,
            self.evaluation_df["test_r2"],
            width=width,
            label="Test",
        )
        axs[1, 0].set_xlabel("Models")
        axs[1, 0].set_ylabel("R2")
        axs[1, 0].set_title("R2 Comparison")
        axs[1, 0].set_xticks(x)
        axs[1, 0].set_xticklabels(models)
        axs[1, 0].legend()
        axs[1, 0].grid(True)

        # Plot 5: MAE comparison
        plot_bar_with_limits(
            axs[1, 1],
            x - width / 2,
            self.evaluation_df["train_mae"],
            width=width,
            label="Train",
        )
        plot_bar_with_limits(
            axs[1, 1],
            x + width / 2,
            self.evaluation_df["test_mae"],
            width=width,
            label="Test",
        )
        axs[1, 1].set_xlabel("Models")
        axs[1, 1].set_ylabel("MAE")
        axs[1, 1].set_title("MAE Comparison")
        axs[1, 1].set_xticks(x)
        axs[1, 1].set_xticklabels(models)
        axs[1, 1].legend()
        axs[1, 1].grid(True)

        # Plot 6: MSE comparison (newly added)
        plot_bar_with_limits(
            axs[1, 2],
            x - width / 2,
            self.evaluation_df["train_mse"],
            width=width,
            label="Train",
        )
        plot_bar_with_limits(
            axs[1, 2],
            x + width / 2,
            self.evaluation_df["test_mse"],
            width=width,
            label="Test",
        )
        axs[1, 2].set_xlabel("Models")
        axs[1, 2].set_ylabel("MSE")
        axs[1, 2].set_title("MSE Comparison")
        axs[1, 2].set_xticks(x)
        axs[1, 2].set_xticklabels(models)
        axs[1, 2].legend()
        axs[1, 2].grid(True)

        # Remove extra subplots
        fig.delaxes(axs[1, 2])

        plt.tight_layout()
        plt.show()

    def plot_distribution(self):
        """
        Plot multiple y-y subplots, with each subplot corresponding to a model in evaluation_df.

        Parameters:
        evaluation_df -- DataFrame containing model evaluation results
        """
        num_plots = len(self.evaluation_df) * 2  # 每个模型有两个子图

        # Calculate rows and columns to make the subplot layout as square as possible
        ncols = math.ceil(math.sqrt(num_plots))
        nrows = math.ceil(num_plots / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))

        # Define blue and orange colors used in the plot
        train_color = "#1f77b4"  # 图中的蓝色
        test_color = "#ff7f0e"  # 图中的橙色

        # If there's only one subplot, axes is not a 2D array, convert it to 2D array
        if num_plots == 1:
            axes = np.array([axes])

        for i, row in enumerate(self.full_rows):
            model_name = row["model_name"]

            # Training set subplot
            train_row_num = i * 2 // ncols
            train_col_num = i * 2 % ncols
            train_ax = axes[train_row_num, train_col_num]

            y_train_true = row["train_true"]
            y_train_pred = row["train_pred"]

            # Calculate x and y axis range for current subplot
            min_val = (
                np.minimum(y_train_true.min().min(), y_train_pred.min().min()) - 10
            )
            max_val = (
                np.maximum(y_train_true.max().max(), y_train_pred.max().max()) + 10
            )

            train_ax.scatter(
                y_train_true, y_train_pred, color=train_color, label="Train"
            )  # Specify training set color as blue
            train_ax.set_xlabel("True Values")
            train_ax.set_ylabel("Predicted Values")
            train_ax.set_title(f"Train Y-Y Plot for {model_name}")
            train_ax.set_xlim(min_val, max_val)
            train_ax.set_ylim(min_val, max_val)
            train_ax.set_aspect("equal", adjustable="box")  # Set aspect ratio to square
            train_ax.legend()  # Add legend

            # Test set subplot
            test_row_num = (i * 2 + 1) // ncols
            test_col_num = (i * 2 + 1) % ncols
            test_ax = axes[test_row_num, test_col_num]

            y_test_true = row["test_true"]
            y_test_pred = row["test_pred"]

            test_ax.scatter(
                y_test_true, y_test_pred, color=test_color, label="Test"
            )  # Specify test set color as red
            test_ax.set_xlabel("True Values")
            test_ax.set_ylabel("Predicted Values")
            test_ax.set_title(f"Test Y-Y Plot for {model_name}")
            test_ax.set_xlim(min_val, max_val)
            test_ax.set_ylim(min_val, max_val)
            test_ax.set_aspect("equal", adjustable="box")  # Set aspect ratio to square
            test_ax.legend()  # Add legend

        # Hide extra subplots
        for j in range(i * 2 + 2, nrows * ncols):
            fig.delaxes(axes.flatten()[j])

        plt.tight_layout()
        plt.show()
