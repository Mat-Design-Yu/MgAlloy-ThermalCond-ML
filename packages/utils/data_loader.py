import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import copy
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize
from scipy.cluster.hierarchy import dendrogram, linkage

import multiprocessing


class DataLoader:
    def __init__(
        self,
        target_csv_path=None,
        generation_features_csv_path=None,
        additional_features_csv_path=None,
        composition_formula_csv_path=None,
        silent=False,
    ):
        """
        DataLoader class initialization.

        Args:
            target_csv (str): The path to the target CSV file.
            generation_features_csv (str): The path to the generation features CSV file.
            additional_features_csv (str): The path to the additional features CSV file.
            composition_formula_csv (str): The path to the composition formula CSV file.
        """
        self.target_csv = target_csv_path
        self.generation_features_csv = generation_features_csv_path
        self.additional_features_csv = additional_features_csv_path
        self.composition_formula_csv = composition_formula_csv_path
        self.data_X = pd.DataFrame({})
        self.data_y = pd.DataFrame({})
        self.data_composition_formula = pd.DataFrame({})
        self.tolerance = 1e-9
        self.silent = silent

    def __deepcopy__(self, memo):
        # 创建一个新的DataLoader对象
        copied_obj = DataLoader()

        # 拷贝简单属性
        copied_obj.target_csv = self.target_csv
        copied_obj.generation_features_csv = self.generation_features_csv
        copied_obj.additional_features_csv = self.additional_features_csv
        copied_obj.composition_formula_csv = self.composition_formula_csv
        copied_obj.tolerance = self.tolerance

        # 使用DataFrame的copy()方法创建新的DataFrame对象
        copied_obj.data_X = self.data_X.copy(deep=True)
        copied_obj.data_y = self.data_y.copy(deep=True)
        copied_obj.data_composition_formula = self.data_composition_formula.copy(
            deep=True
        )

        return copied_obj

    def load_csv(self, encoding="utf-8-sig", shuffle=False, random_state=None):
        """
        Load data from CSV files and optionally shuffle the data.

        Args:
            encoding (str): The encoding of the CSV files. Default is 'utf-8-sig'.
            shuffle (bool): Whether to shuffle the loaded data. Default is False.
            random_state (int): The random state for shuffling. Default is None.

        Returns:
            tuple: A tuple containing the following elements:
                - data_X (DataFrame): Pandas DataFrame containing the feature data.
                - data_y (DataFrame): Pandas DataFrame containing the target data.
                - data_composition_formula (DataFrame): Pandas DataFrame containing the composition formula data.
        """
        data_generation_features = self._load_csv(
            self.generation_features_csv, encoding=encoding
        )
        data_additional_features = self._load_csv(
            self.additional_features_csv, encoding=encoding
        )

        self.data_X = pd.concat(
            [
                data_additional_features.reset_index(drop=True),
                data_generation_features.reset_index(drop=True),
            ],
            axis=1,
        )
        self.data_y = self._load_csv(self.target_csv, encoding=encoding)
        self.data_composition_formula = self._load_csv(
            self.composition_formula_csv, encoding=encoding
        )

        # Ensure all DataFrames have the same number of rows
        non_empty_dataframes = [
            df
            for df in [
                data_generation_features,
                data_additional_features,
                self.data_X,
                self.data_y,
                self.data_composition_formula,
            ]
            if not df.empty
        ]

        if not all(
            len(df) == len(non_empty_dataframes[0]) for df in non_empty_dataframes
        ):
            raise ValueError("Not all DataFrames have the same number of rows.")

        if shuffle:
            self.re_shuffle(random_state)

        return self.data_X, self.data_y, self.data_composition_formula

    def _load_csv(self, file_path, encoding="utf-8-sig"):
        """
        Load data from a CSV file using Pandas with the specified encoding.

        Args:
            file_path (str): The path to the CSV file.
            encoding (str): The encoding of the CSV file. Default is 'utf-8-sig'.

        Returns:
            DataFrame: Pandas DataFrame containing the loaded data, or an empty DataFrame if file_path is None.

        Raises:
            FileNotFoundError: If the specified file is not found.
            UnicodeDecodeError: If an error occurs while decoding the data.
            Exception: If any other error occurs while loading the data.
        """
        try:
            if file_path is not None:
                data = pd.read_csv(file_path, encoding=encoding)
                if not self.silent:
                    print(
                        f"Data loaded successfully from {file_path} with shape {data.shape}"
                    )
                return data
            else:
                return pd.DataFrame({})

        except FileNotFoundError:
            print(f"The file {file_path} was not found.")
        except UnicodeDecodeError as e:
            print(f"An error occurred while decoding the data: {e}")
            print(
                f"Try specifying a different encoding such as 'latin1', 'ISO-8859-1', or 'cp1252'."
            )
        except Exception as e:
            print(f"An error occurred while loading the data: {e}")
            return None

    def remove_constant_features(self):
        """
        Remove constant value features from the DataFrame.

        Returns:
            tuple: A tuple containing the following elements:
                - data_X (DataFrame): A DataFrame with constant features removed.
                - removed_details (DataFrame): A DataFrame with details of removed features.
        """
        # Identify constant features (columns with a unique value count of 1)
        constant_features = [
            col for col in self.data_X.columns if self.data_X[col].nunique() == 1
        ]
        # Calculate the number of constant features to be removed
        num_features_removed = len(constant_features)
        # Create a DataFrame to store details of the removed features
        removed_details = pd.DataFrame(
            {
                "Index": [
                    self.data_X.columns.get_loc(col) for col in constant_features
                ],
                "Name": constant_features,
                "Value": [self.data_X[col].iloc[0] for col in constant_features],
            }
        )
        # Drop the constant features from the DataFrame
        self.data_X = self.data_X.drop(columns=constant_features)

        # Print the number of removed constant features
        print(f"Removed {num_features_removed} constant features.")

        return self.data_X, removed_details

    def remove_duplicate_features(self):
        """
        Remove duplicate features from the DataFrame and provide a report.

        Returns:
            tuple: A tuple containing the following elements:
                - data_X (DataFrame): A DataFrame with duplicate features removed.
                - removed_details (DataFrame): A DataFrame with details of removed and retained features.
        """
        # Transpose the DataFrame to ease detection of duplicate rows (which are duplicate columns in the original DataFrame)
        df_transposed = self.data_X.T

        # Use DataFrame.duplicated() to mark duplicates, keep='first' means the first occurrence is considered unique
        duplicates_marked = df_transposed.duplicated(keep=False)

        # Group the duplicate features by their identical values as a string
        features_grouped = (
            df_transposed[duplicates_marked].groupby(list(df_transposed)).groups
        )

        # Iterate over the groups and determine which to keep and which to remove
        to_remove = []
        retained_removed_details = []
        for group, indices in features_grouped.items():
            # Convert indices to list of feature names
            features = list(indices)
            retained = features[0]
            removed = features[1:]
            to_remove.extend(removed)

            for feature in features:
                retained_removed_details.append(
                    {
                        "Feature": feature,
                        "Retained": feature == retained,
                        "Group": ", ".join(features),
                    }
                )

        # Create a DataFrame to store details of the removed features
        removed_details = pd.DataFrame(retained_removed_details)

        # Drop the duplicate features from the DataFrame
        self.data_X = self.data_X.drop(columns=to_remove)

        # Print the number of removed duplicate features
        print(f"Removed {len(to_remove)} duplicate features.")

        return self.data_X, removed_details

    def _process_feature_pair(self, i):
        """
        Process a pair of features to check for collinearity.

        Args:
            i (int): The index of the feature to compare with other features.

        Returns:
            list of tuples: Each tuple contains the index of the retained feature, the index of the removed feature,
                and the regression equation as a string.
        """
        collinear_pairs = []

        x_i = self.data_X.iloc[:, i].values.reshape(-1, 1)

        for j in range(i + 1, self.data_X.shape[1]):
            feature_j_name = self.data_X.columns[j]
            x_j = self.data_X.iloc[:, j].values
            model = LinearRegression()
            model.fit(x_i, x_j)
            score = model.score(x_i, x_j)

            if score > 1 - self.tolerance:
                equation = (
                    f"{feature_j_name} ({model.coef_[0]:.2f}*X+{model.intercept_:.2f})"
                )
                collinear_pairs.append((i, j, equation))

        return collinear_pairs  # Return the list of collinear pairs for this feature

    def remove_collinear_features(self, tolerance=1e-9):
        """
        Remove collinear features from the DataFrame and provide a report.

        Args:
            tolerance (float): The tolerance threshold for determining collinearity. Default is 1e-9.

        Returns:
            tuple: A tuple containing the following elements:
                - data_X (DataFrame): A DataFrame with collinear features removed.
                - removed_details (DataFrame): A DataFrame with details of removed and retained features.
        """
        self.tolerance = tolerance

        pool = multiprocessing.Pool()

        # Collect pairs of collinear features
        collinear_pairs_list = pool.map(
            self._process_feature_pair, range(self.data_X.shape[1])
        )

        pool.close()
        pool.join()

        # Flatten the list of collinear pairs
        collinear_pairs = [pair for sublist in collinear_pairs_list for pair in sublist]

        # Determine which features to remove
        features_to_remove = set()
        retained_removed_details = []

        # Sort pairs by the index of the first feature so that we prefer retaining features with a lower index
        collinear_pairs.sort(key=lambda x: x[0])

        for i, j, equation in collinear_pairs:
            # Only add feature j to removal list if it hasn't been added before
            if j not in features_to_remove:
                features_to_remove.add(j)
                retained_removed_details.append(
                    {
                        "Retained feature": self.data_X.columns[i],
                        "Removed feature": equation,
                    }
                )

        # Remove the features
        self.data_X = self.data_X.drop(
            columns=[self.data_X.columns[j] for j in features_to_remove]
        )
        removed_details = pd.DataFrame(retained_removed_details)

        print(f"Removed {len(features_to_remove)} collinear features.")

        return self.data_X, removed_details

    def remove_non_numeric_features(self):
        """
        Remove non-numeric features from the DataFrame.

        Returns:
            tuple: A tuple containing the following elements:
                - data_X (DataFrame): A DataFrame with non-numeric features removed.
                - removed_details (DataFrame): A DataFrame with details of removed features.
        """
        # Identify non-numeric features
        non_numeric_features = [
            col
            for col in self.data_X.columns
            if not pd.api.types.is_numeric_dtype(self.data_X[col])
        ]

        # Calculate the number of non-numeric features to be removed
        num_features_removed = len(non_numeric_features)

        # Create a DataFrame to store details of the removed features
        removed_details = pd.DataFrame(
            {
                "Index": [
                    self.data_X.columns.get_loc(col) for col in non_numeric_features
                ],
                "Name": non_numeric_features,
                "Value": [self.data_X[col].iloc[0] for col in non_numeric_features],
            }
        )

        # Drop the non-numeric features from the DataFrame
        self.data_X = self.data_X.select_dtypes(include=[np.number])

        # Print the number of removed non-numeric features
        print(f"Removed {num_features_removed} non-numeric features.")

        return self.data_X, removed_details

    def remove_nan_features(self, axis=1):
        """
        Remove features or samples with NaN values from the DataFrame.

        Args:
            axis (int, optional): The axis along which to remove NaN values.
                0 for removing rows (samples), 1 for removing columns (features).
                Defaults to 1.

        Returns:
            tuple: A tuple containing the following elements:
                - data_X (DataFrame): A DataFrame with features or samples containing NaN values removed.
                - removed_details1 (DataFrame): A DataFrame with details of removed features.
                - removed_details2 (DataFrame): A DataFrame with details of removed samples.
        """
        # Identify features with NaN values
        nan_features = [
            col for col in self.data_X.columns if self.data_X[col].isnull().any()
        ]

        # Calculate the number of features with NaN values to be removed
        num_features_removed = len(nan_features)

        # Create a DataFrame to store details of the removed features
        removed_details1 = pd.DataFrame(
            {
                "Index": [self.data_X.columns.get_loc(col) for col in nan_features],
                "Name": nan_features,
                "NaN Count": [self.data_X[col].isnull().sum() for col in nan_features],
            }
        )
        removed_details2 = pd.DataFrame({})

        # Identify samples with NaN values
        nan_samples = self.data_X.index[self.data_X.isnull().any(axis=1)]

        if axis == 1:
            # Drop the features with NaN values from the DataFrame
            self.data_X = self.data_X.dropna(axis=1)

            print(f"Removed {num_features_removed} features containing NaN values.")
        elif axis == 0:
            if not self.data_composition_formula.empty:
                removed_details2 = pd.DataFrame(
                    {
                        "Index": nan_samples,
                        "Formula": self.data_composition_formula.iloc[
                            nan_samples
                        ].values.flatten(),
                    }
                )
            else:
                removed_details2 = pd.DataFrame(
                    {"Index": nan_samples, "Formula": "unimport"}
                )

            # Calculate the number of samples with NaN values to be removed
            num_samples_removed = len(nan_samples)

            # Drop the samples with NaN values from data_X, data_y, and data_composition_formula
            self.data_X = self.data_X.dropna(axis=0)
            if not self.data_y.empty:
                self.data_y = self.data_y.loc[self.data_X.index]
            if not self.data_composition_formula.empty:
                self.data_composition_formula = self.data_composition_formula.loc[
                    self.data_X.index
                ]

            print(f"Removed {num_samples_removed} samples containing NaN values.")
        else:
            raise ValueError(
                "Invalid axis value. Axis must be 0 (rows) or 1 (columns)."
            )

        return self.data_X, removed_details1, removed_details2

    def min_max_scaler(self, feature_range=(0, 1), scaler=None):
        """
        Apply Min-Max scaling to the numeric data in the DataFrame.

        Args:
            feature_range (tuple): Desired range of transformed data. Default is (0, 1).
            scaler (MinMaxScaler, optional): An existing MinMaxScaler instance.
                                             If None, a new MinMaxScaler will be created and used.

        Returns:
            tuple: A tuple containing the scaled DataFrame and the MinMaxScaler used.
        """
        # If no scaler is provided, initialize a new MinMaxScaler with the desired feature range
        if scaler is None:
            scaler = MinMaxScaler(feature_range=feature_range)

        # Select numeric columns (exclude object/boolean data types)
        numeric_columns = self.data_X.select_dtypes(
            include=["int64", "float32", "float64"]
        ).columns

        # Fit the scaler on the numeric columns of the DataFrame
        self.data_X[numeric_columns] = scaler.fit_transform(
            self.data_X[numeric_columns]
        )

        # Print a message indicating successful normalization
        print(f"Data has been scaled to the range {feature_range}.")

        # Return the scaled DataFrame and the scaler
        return self.data_X, scaler

    def normalize(self, norm="l2", axis=1):
        """
        Normalize the numeric data in the DataFrame using L1, L2, or max normalization.

        Args:
            norm (str): The type of normalization to perform. Options are 'l1', 'l2', or 'max'. Default is 'l2'.
            axis (int): The axis along which to normalize. 0 for columns, 1 for rows. Default is 1.

        Returns:
            DataFrame: A DataFrame with normalized data.
        """
        # Select numeric columns (exclude object/boolean data types)
        numeric_columns = self.data_X.select_dtypes(
            include=["int64", "float32", "float64"]
        ).columns

        # Normalize the numeric columns of the DataFrame
        self.data_X[numeric_columns] = normalize(
            self.data_X[numeric_columns], norm=norm, axis=axis
        )

        # Print a message indicating successful normalization
        print(f"Data has been normalized using {norm} normalization along axis {axis}.")

        return self.data_X

    def standard_scaler(self, scaler=None):
        """
        Standardize the numeric data in the DataFrame using StandardScaler.

        Args:
            scaler (StandardScaler, optional): An existing StandardScaler instance.
                                               If None, a new StandardScaler will be created and used.

        Returns:
            tuple: A tuple containing the standardized DataFrame and the StandardScaler used.
        """
        # If no scaler is provided, initialize a new StandardScaler
        if scaler is None:
            scaler = StandardScaler()

        # Select numeric columns (exclude object/boolean data types)
        numeric_columns = self.data_X.select_dtypes(
            include=["int64", "float32", "float64"]
        ).columns

        # Fit the scaler on the numeric columns of the DataFrame and transform the data
        self.data_X[numeric_columns] = scaler.fit_transform(
            self.data_X[numeric_columns]
        )

        # Print a message indicating successful standardization
        print("Data has been standardized using StandardScaler.")

        # Return the standardized DataFrame and the scaler
        return self.data_X, scaler

    def re_shuffle(self, random_state=42):
        """
        Shuffle the DataFrame and update the data composition formula and target variables.

        Args:
            random_state (int): Random state for reproducibility.

        Returns:
            tuple: A tuple containing the following elements:
                - data_X (DataFrame): A DataFrame with the shuffled feature data.
                - data_y (Series): A Series with the shuffled target data.
                - data_composition_formula (DataFrame): A DataFrame with the shuffled composition formula data.
        """
        if not self.data_X.empty:
            self.data_X = self.data_X.sample(
                frac=1, random_state=random_state
            ).reset_index(drop=True)
        if not self.data_y.empty:
            self.data_y = self.data_y.sample(
                frac=1, random_state=random_state
            ).reset_index(drop=True)
        if not self.data_composition_formula.empty:
            self.data_composition_formula = self.data_composition_formula.sample(
                frac=1, random_state=random_state
            ).reset_index(drop=True)

        return (
            self.data_X.reset_index(drop=True),
            self.data_y.reset_index(drop=True),
            self.data_composition_formula.reset_index(drop=True),
        )

    def plot_corr_heatmap(self, cmap="RdBu_r"):
        """
        Plot a correlation matrix heatmap of the feature data.

        Args:
            cmap (str, optional): The color scheme for the heatmap. Defaults to 'RdBu_r'.

        Returns:
            None
        """
        # Calculate the correlation matrix
        correlation_matrix = self.data_X.corr()

        # Perform hierarchical clustering
        Z = linkage(correlation_matrix, method="ward")

        # Rearrange the correlation matrix based on clustering results
        idx = dendrogram(Z, no_plot=True)["leaves"]
        correlation_matrix = correlation_matrix.iloc[idx, idx]

        # Find the highest and lowest correlation values (excluding the diagonal)
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        tri_df = correlation_matrix.mask(mask)

        # Get the min and max correlation values and their corresponding locations
        min_corr = tri_df.min().min()
        max_corr = tri_df.max().max()
        min_location = tri_df.stack().idxmin()
        max_location = tri_df.stack().idxmax()

        print(f"Minimum correlation: {min_corr} between {min_location}")
        print(f"Maximum correlation: {max_corr} between {max_location}")

        # Plot the correlation matrix heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=False, cmap=cmap, square=True)
        plt.title("Correlation Matrix Heatmap")
        plt.show()
