# Based on the mlxtend library by Sebastian Raschka
# The original mlxtend code is licensed under the BSD 3-clause License
# https://github.com/rasbt/mlxtend

# Original mlxtend licensing text:
# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
# License: BSD 3 clause


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from sklearn.model_selection import cross_val_score
from sklearn.metrics import check_scoring
from sklearn.base import clone
from tqdm import tqdm

from joblib import Parallel, delayed
from copy import deepcopy
from itertools import combinations
import scipy.stats
from packages.utils.feature_name_clean import clean_feature_names


class SequentialFeatureSelection:
    def __init__(self, dataframe_X, dataframe_y, model, forward=True, floating=False):
        """
        Initialize the SequentialFeatureSelection class with the provided dataframe and a regression model.

        Parameters:
        dataframe (DataFrame): A pandas DataFrame where the first column is the target variable and the remaining columns are features.
        model (object): A regression model instance that has a fit method.
        """
        self.forward = forward
        self.floating = floating

        self.dataframe = pd.concat(
            [dataframe_y.reset_index(drop=True), dataframe_X.reset_index(drop=True)],
            axis=1,
        )
        self.target_column = self.dataframe.columns[
            0
        ]  # Assuming the first column is the target
        self.model = model
        self.features_name = list(
            self.dataframe.columns[1:]
        )  # All features except the target
        self.n_features = len(self.features_name)

    def _merge_features_idx_with_features_name(self, features_idx=None):
        if features_idx is None:
            features_idx = list(range(self.n_features))

        current_features = []
        for idx in features_idx:
            current_features.append(self.features_name[idx])

        current_features = [
            feature
            for feature in self.dataframe.columns[1:]
            if feature in current_features
        ]
        return current_features

    def _calc_score(self, features_idx):
        cloned_model = clone(self.model)
        current_features = self._merge_features_idx_with_features_name(features_idx)
        if self.cv == -1:
            cloned_model.fit(
                clean_feature_names(self.dataframe[current_features]),
                clean_feature_names(self.dataframe[self.target_column]),
            )
            scores = np.array(
                [
                    self.scorer(
                        cloned_model,
                        clean_feature_names(self.dataframe[current_features]),
                        clean_feature_names(self.dataframe[self.target_column]),
                    )
                ]
            )
        else:
            scores = cross_val_score(
                cloned_model,
                clean_feature_names(self.dataframe[current_features]),
                clean_feature_names(self.dataframe[self.target_column]),
                cv=self.cv,
                scoring=self.scorer,
                n_jobs=1,
            )
        return features_idx, scores

    def feature_selection(
        self,
        cv=-1,
        scoring="neg_mean_absolute_percentage_error",
        feature_stop=-1,
        tolerance=0,
        floating_tolerance=0,
        postfix="",
        plot=True,
        n_jobs=-1,
    ):
        self.cv = cv
        self.scorer = check_scoring(self.model, scoring=scoring)

        self.features_history = []  # To keep track of feature names at each step
        self.score_history = []  # To keep track of score at each step
        self.cv_score_history = []  # To keep track of cv_score at each step
        self.result_dataframe = None
        self.selected_features = []  # Starts with no features
        if feature_stop == -1:
            self.feature_stop = self.n_features
        else:
            self.feature_stop = min(self.n_features, feature_stop)

        best_score = -np.inf
        feature_counts = []

        self.results_dict = {}

        # Set up the plot
        if plot:
            fig, ax = plt.subplots()
            (line,) = ax.plot([], [], marker="o", linestyle="-", color="blue")
            (highlight,) = ax.plot([], [], "ro")
            ax.set_xlim(0, self.n_features + 1)
            ax.set_xlabel("Number of Features")
            if cv == -1:
                ax.set_ylabel("Score")
            else:
                ax.set_ylabel("Cross-validated Score")
            if self.floating:
                ax.set_title(f"SFFS + {self.model} {postfix}")
            else:
                ax.set_title(f"SFS + {self.model} {postfix}")
        else:
            pbar = tqdm(
                total=int(self.feature_stop),
                desc="SFFS",
                postfix=dict(score=0),
            )

        if self.forward:
            current_features_idx = tuple()
            k_stop = self.feature_stop
        else:
            current_features_idx = tuple(range(self.n_features))
            k_stop = 1

        current_feature_count = len(current_features_idx)
        if current_feature_count > 0:
            current_features_idx, current_cv_scores = self._calc_score(
                current_features_idx
            )
            self.results_dict[current_feature_count] = {
                "feature_idx": current_features_idx,
                "avg_score": np.nanmean(current_cv_scores),
                "cv_scores": current_cv_scores,
            }

        feature_origin_set = set(range(self.n_features))

        while current_feature_count != k_stop:
            feature_pre_subset = set(current_features_idx)
            if self.forward:
                feature_search_set = feature_origin_set
                feature_must_include_set = feature_pre_subset
            else:
                feature_search_set = feature_pre_subset
                feature_must_include_set = set()

            current_features_idx, current_score, current_cv_scores = (
                self._feature_selector(
                    feature_search_set,
                    feature_must_include_set,
                    is_forward=self.forward,
                    n_jobs=n_jobs,
                )
            )
            current_feature_count = len(current_features_idx)

            self.features_history.append(
                self._merge_features_idx_with_features_name(current_features_idx)[:]
            )  # record a snapshot of current features
            self.score_history.append(current_score)
            self.cv_score_history.append(current_cv_scores.tolist())

            feature_counts.append(current_feature_count)

            if current_score > best_score + tolerance:
                best_score = current_score
                if plot:
                    highlight.set_data(feature_counts[-1], best_score)
                self.selected_features = self.features_history[-1].copy()

            # Update the plot
            if plot:
                line.set_data(feature_counts, self.score_history)
                ax.relim()  # Recalculate axis limits
                ax.autoscale_view()  # Automatically adjust axis scaling based on data constraints
                clear_output(wait=True)
                display(fig)
            else:
                pbar.n = feature_counts[-1]  # Set current absolute position
                pbar.set_postfix(score=f"{-current_score:.4f}")
                pbar.refresh()  # Refresh progress bar display

            # floating can lead to multiple same-sized subsets
            if current_feature_count not in self.results_dict or (
                current_score > self.results_dict[current_feature_count]["avg_score"]
            ):
                current_features_idx = tuple(sorted(current_features_idx))
                self.results_dict[current_feature_count] = {
                    "feature_idx": current_features_idx,
                    "avg_score": current_score,
                    "cv_scores": current_cv_scores,
                }

            if self.floating:
                forward_floating = not self.forward
                (best_feature_idx,) = set(current_features_idx) ^ feature_pre_subset
                for _ in range(self.n_features):
                    if self.forward and (len(current_features_idx)) <= 1:
                        break
                    if not self.forward and (
                        len(feature_origin_set) - len(current_features_idx) <= 1
                    ):
                        break

                    if forward_floating:
                        # corresponding to self.forward=False
                        feature_search_set = feature_origin_set - {best_feature_idx}
                        feature_must_include_set = set(current_features_idx)
                    else:
                        # corresponding to self.forward=True
                        feature_search_set = set(current_features_idx)
                        feature_must_include_set = {best_feature_idx}

                    (
                        current_features_idx_floating,
                        current_score_floating,
                        current_cv_score_floating,
                    ) = self._feature_selector(
                        feature_search_set,
                        feature_must_include_set,
                        is_forward=forward_floating,
                        n_jobs=n_jobs,
                    )

                    if current_score_floating <= current_score + floating_tolerance:
                        break
                    if (
                        current_score_floating
                        <= self.results_dict[len(current_features_idx_floating)][
                            "avg_score"
                        ]
                    ):
                        break
                    else:
                        current_features_idx, current_score, current_cv_scores = (
                            current_features_idx_floating,
                            current_score_floating,
                            current_cv_score_floating,
                        )
                        current_features_idx = tuple(sorted(current_features_idx))
                        current_feature_count = len(current_features_idx)
                        self.results_dict[current_feature_count] = {
                            "feature_idx": current_features_idx,
                            "avg_score": current_score,
                            "cv_scores": current_cv_scores,
                        }

                    self.features_history.append(
                        self._merge_features_idx_with_features_name(
                            current_features_idx
                        )[:]
                    )  # record a snapshot of current features
                    self.score_history.append(current_score)
                    self.cv_score_history.append(current_cv_scores.tolist())

                    feature_counts.append(current_feature_count)

                    if current_score > best_score and self.forward:
                        if current_score > best_score + tolerance or feature_counts[
                            -1
                        ] < len(self.selected_features):
                            best_score = current_score
                            if plot:
                                highlight.set_data(feature_counts[-1], best_score)
                            self.selected_features = self.features_history[-1].copy()

                    if plot:
                        # Update the plot
                        line.set_data(feature_counts, self.score_history)
                        ax.relim()  # Recalculate axis limits
                        ax.autoscale_view()  # Automatically adjust axis scaling based on data constraints
                        clear_output(wait=True)
                        display(fig)
                    else:
                        pbar.n = feature_counts[-1]  # Set current absolute position
                        pbar.set_postfix(score=f"{-current_score:.4f}")
                        pbar.refresh()  # Refresh progress bar display

        if plot:
            plt.close(fig)
        else:
            pbar.close()

        self.result_dataframe = self.get_feature_history_raw()
        return self.selected_features

    def _feature_selector(
        self, feature_search_set, feature_must_include_set, is_forward, n_jobs=-1
    ):
        """
        Select the best feature set based on the model's scoring function.

        This method evaluates different subsets of features, either adding one feature at a time (forward selection)
        or removing one feature at a time (backward elimination), to determine the best performing feature set.

        Parameters:
        feature_search_set (set): The set of features to consider for selection.
        feature_must_include_set (set): The set of features that must be included in every evaluated subset.
        is_forward (bool): A flag indicating the direction of feature selection:
                        True for forward selection, False for backward elimination.

        Returns:
        out (tuple): A tuple containing the best feature subset found (set), its average score (float),
                    and the cross-validation scores (list). If no feature subset is found, the tuple will
                    contain (None, None, None).
        """
        out = (None, None, None)

        # Calculate the remaining features to consider by removing the must-include features from the search set.
        feature_remaining_set = feature_search_set - feature_must_include_set
        # Convert the set of remaining features into a list for further processing.
        feature_remaining_list = list(feature_remaining_set)
        # Count the number of features left after subtracting the must-include features.
        feature_remaining_len = len(feature_remaining_list)

        # If there are no remaining features, return the initialized output directly.
        if feature_remaining_len <= 0:
            return out

        # If the selection mode is forward, generate all possible single feature combinations.
        # If the selection mode is backward, generate combinations of all features except one.
        if is_forward:
            all_feature_subsets_unfitted = combinations(feature_remaining_list, r=1)
        else:
            all_feature_subsets_unfitted = combinations(
                feature_remaining_list, r=feature_remaining_len - 1
            )

        # Set up parallel processing with all available CPU cores.
        parallel = Parallel(n_jobs=n_jobs)
        # Execute the score calculation for each feature subset in parallel.
        work = parallel(
            delayed(self._calc_score)(tuple(set(p) | feature_must_include_set))
            for p in all_feature_subsets_unfitted
        )

        # Initialize lists to store average cross-validation scores, detailed CV scores, and feature subsets.
        all_avg_scores = []
        all_cv_scores = []
        all_feature_subsets_fitted = []
        # Iterate over the work results to populate the lists with scores and subsets.
        for new_subset, cv_scores in work:
            all_avg_scores.append(
                np.nanmean(cv_scores)
            )  # Append the mean of cross-validation scores, handling NaN values.
            all_cv_scores.append(
                cv_scores
            )  # Append the detailed cross-validation scores.
            all_feature_subsets_fitted.append(new_subset)  # Append the feature subsets.

        # If any average scores were calculated, find the best feature subset.
        if len(all_avg_scores) > 0:
            best = np.argmax(
                all_avg_scores
            )  # Find the index of the highest average score.
            # Update the output with the best feature subset and its scores.
            out = (
                all_feature_subsets_fitted[best],
                all_avg_scores[best],
                all_cv_scores[best],
            )

        # Return the best feature subset along with its average score and cross-validation scores.
        return out

    def get_metric_dict(self, confidence_interval=0.95):
        """
        Calculate and add confidence interval bounds, standard deviation, and standard error
        to the metrics for each feature set in the result sets.

        The confidence interval bound is calculated using the t-distribution, which is more
        appropriate for small sample sizes than the normal distribution.

        Parameters:
        confidence_interval (float, optional): The confidence level for the interval. Default is 0.95.

        Returns:
        fdict (dict): A dictionary with the original result sets, including the new keys
                      'ci_bound', 'std_dev', and 'std_err' representing the confidence interval
                      bound, standard deviation, and standard error, respectively.
        """
        # DeepCopy ensures we do not modify the original result sets
        fdict = deepcopy(self.results_dict)

        # Iterate over the keys in the result sets dictionary
        for k in fdict:
            # Calculate the standard deviation of cross-validation scores
            std_dev = np.std(self.results_dict[k]["cv_scores"])

            # Calculate the standard error of the mean of the cross-validation scores
            std_err = scipy.stats.sem(self.results_dict[k]["cv_scores"])

            # Calculate the confidence interval bound using the t-distribution's percent point function
            bound = std_err * scipy.stats.t._ppf(
                (1 + confidence_interval) / 2.0,
                len(self.results_dict[k]["cv_scores"]) - 1,
            )

            # Add the calculated values to the dictionary
            fdict[k]["ci_bound"] = bound
            fdict[k]["std_dev"] = std_dev
            fdict[k]["std_err"] = std_err

        return fdict

    def get_feature_history_raw(self):
        """
        Formats and returns the feature history as a DataFrame.

        Returns:
        feature_history_df (DataFrame): A DataFrame containing the history of features after each selection step.
        """
        if self.result_dataframe is not None:
            return self.result_dataframe

        # Initialize a list to store the history records
        history_records = []

        # Iterate over the feature history and store the iteration number and concatenated feature names
        for i, features in enumerate(self.features_history):
            # Join all feature names in a single string separated by a chosen delimiter, e.g., ', '
            concatenated_features = ", ".join(features)
            concatenated_cv_scores = self.cv_score_history[i]
            # Append the iteration and the concatenated features as a new record
            history_records.append(
                {
                    "Iteration": i,
                    "Score": self.score_history[i],
                    "Feature_count": len(features),
                    "Features": concatenated_features,
                    "CV_Score": concatenated_cv_scores,
                }
            )

        # Create a DataFrame from the history records
        self.result_dataframe = feature_history_df = pd.DataFrame(history_records)

        return feature_history_df

    def get_feature_history(self):
        """
        Formats and returns the feature history as a DataFrame, including newly added features after each elimination step.

        Returns:
        feature_history_df (DataFrame): A DataFrame containing the history of features after each elimination step,
                                        including newly added features and their corresponding scores.
        """
        # Initialize a list to store the history records
        history_records = []

        previous_features_set = set()

        # Iterate over the feature counts from 1 to self.feature_stop
        for i in range(1, self.feature_stop + 1):
            # Get the current features for the current count
            current_features = self.get_features(i)
            current_features_set = set(current_features)

            # Get the newly added features by difference
            new_features = list(current_features_set - previous_features_set)
            # Sort features
            new_features = [
                feature
                for feature in self.dataframe.columns[1:]
                if feature in new_features
            ]

            # Update the previous features to be the current features
            previous_features_set = current_features_set

            # Append the feature count, score, cv score, and newly added features as a new record
            history_records.append(
                {
                    "Feature_count": i,
                    "Score": self.get_score(i),
                    "CV_Score": self.get_cv_scores(i),
                    "Feature_i": ", ".join(new_features) if new_features else "None",
                    "Features_i": ", ".join(current_features)
                    if current_features
                    else "None",
                }
            )

        # Create a DataFrame from the history records
        feature_history_df = pd.DataFrame(history_records)

        return feature_history_df

    def get_features(self, feature_count):
        """
        Extract a set number of features from the model.

        The features are based on the model's current state and the
        specified number of features to extract.

        Parameters:
        feature_count (int): The number of features to extract from the model.

        Returns:
        features (list): A list containing the extracted features. The length of the list
                is equal to feature_count.
        """

        # Check if feature_count is a positive integer
        if not isinstance(feature_count, int) or feature_count <= 0:
            raise ValueError("feature_count must be a positive integer")

        # Filter rows with the given feature_count
        filtered_df = self.result_dataframe[
            self.result_dataframe["Feature_count"] == feature_count
        ]

        # Find the row with the maximum score
        max_score_row = filtered_df.loc[filtered_df["Score"].idxmax()]

        # Extract the features from the row with the maximum score
        features_iteration = max_score_row["Iteration"]

        return self.features_history[features_iteration]

    def get_score(self, feature_count):
        """
        Calculate the model's score based on a certain number of features.

        The score is a hypothetical representation of the model's performance
        and is influenced by the number and nature of the features extracted.

        Parameters:
        feature_count (int): The number of features to consider when calculating
                            the score.

        Returns:
        score (float): The calculated score as a floating-point number.
        """

        # Check if feature_count is a positive integer
        if not isinstance(feature_count, int) or feature_count <= 0:
            raise ValueError("feature_count must be a positive integer")

        # Filter rows with the given feature_count
        filtered_df = self.result_dataframe[
            self.result_dataframe["Feature_count"] == feature_count
        ]

        # Find the row with the maximum score
        max_score_row = filtered_df.loc[filtered_df["Score"].idxmax()]

        # Extract the score from the row with the maximum score
        score = max_score_row["Score"]

        return score

    def get_cv_scores(self, feature_count):
        """
        Calculate the model's cv_scores based on a certain number of features.

        The cv_scores is a hypothetical representation of the model's performance
        and is influenced by the number and nature of the features extracted.

        Parameters:
        feature_count (int): The number of features to consider when calculating
                            the cv_scores.

        Returns:
        cv_scores (list): A list containing the cv_scores. The length of the list
                          is equal to feature_count.
        """

        # Check if feature_count is a positive integer
        if not isinstance(feature_count, int) or feature_count <= 0:
            raise ValueError("feature_count must be a positive integer")

        # Filter rows with the given feature_count
        filtered_df = self.result_dataframe[
            self.result_dataframe["Feature_count"] == feature_count
        ]

        # Find the row with the maximum score
        max_score_row = filtered_df.loc[filtered_df["Score"].idxmax()]

        # Extract the score from the row with the maximum score
        cv_scores = max_score_row["CV_Score"]

        return cv_scores
