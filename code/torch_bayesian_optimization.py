import numpy as np
import sklearn
from sklearn.model_selection import cross_val_score
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold
import torch
from tqdm import tqdm
import psutil
from ax.service.ax_client import AxClient
from ax.utils.common.logger import set_stderr_log_level, logging

set_stderr_log_level(logging.ERROR)


class TorchBayesianOptimization:
    def __init__(
        self, model_class, scoring, search_spaces, n_iter, cv, random_state, device
    ):
        self.model_class = model_class
        self.search_spaces = search_spaces
        self.objectives = scoring
        self.scoring_str = list(scoring.keys())[0]
        self.n_iter = n_iter
        self.cv = cv
        self.random_state = random_state
        self.tracking_metric_names = self.scoring_str
        self.ax_client = AxClient()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        if isinstance(self.scoring_str, str):
            self.scoring = get_scorer(self.scoring_str)
        self.kf = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)

    def load(self, data_loader, train_test_split=True, test_size=0.2):
        self.train_test_split = train_test_split
        if train_test_split:
            self.X_train, self.X_test, self.y_train, self.y_test = (
                sklearn.model_selection.train_test_split(
                    data_loader.data_X.values,
                    data_loader.data_y.values,
                    test_size=test_size,
                    random_state=self.random_state,
                )
            )
        else:
            self.X_train = data_loader.data_X.values
            self.y_train = data_loader.data_y.values
            self.X_test = None
            self.y_test = None

    def train_and_evaluate(self, parameterization):
        model = self.model_class(
            **parameterization, random_state=self.random_state, device=self.device
        )
        cv_scores = cross_val_score(
            model,
            self.X_train,
            self.y_train,
            cv=self.kf,
            scoring=self.scoring_str,
            n_jobs=self._get_physical_cores(),
        )
        return np.mean(cv_scores), np.std(cv_scores)

    def optimize(self, info=False):
        self.ax_client.create_experiment(
            parameters=self.search_spaces,
            objectives=self.objectives,
            tracking_metric_names=self.tracking_metric_names,
        )
        self.iteration_results = []

        if info == False:
            pbar = tqdm(
                total=self.n_iter,
                desc="Optimization Progress",
                postfix=dict(mean_cv_score=0, std_cv_score=0),
            )

        for _ in range(self.n_iter):
            parameters, trial_index = self.ax_client.get_next_trial()
            mean_cv_score, std_cv_score = self.train_and_evaluate(parameters)
            self.ax_client.complete_trial(
                trial_index=trial_index,
                raw_data={self.scoring_str: (mean_cv_score, std_cv_score)},
            )
            self.iteration_results.append(
                {
                    "params": parameters,
                    "mean_cv_score": mean_cv_score,
                    "std_cv_score": std_cv_score,
                }
            )

            if info == True:
                print(f"Iteration {len(self.iteration_results)}")
                print("Best params:")
                for param, value in parameters.items():
                    print(f"  {param}: {value}")
                print(f"Mean CV score: {mean_cv_score:.4f}")
                print(f"Std CV score: {std_cv_score:.4f}")
                print("-" * 50)
            else:
                pbar.set_postfix(
                    mean_cv_score=f"{mean_cv_score:.4f}",
                    std_cv_score=f"{std_cv_score:.4f}",
                )
                pbar.update(1)

        if info == False:
            pbar.close()

        self.best_params, metrics = self.ax_client.get_best_parameters(
            use_model_predictions=False
        )
        self.best_r2 = metrics

    def evaluate(self):
        model1 = self.model_class(
            **self.best_params, random_state=self.random_state, device=self.device
        )
        model2 = self.model_class(
            **self.best_params, random_state=self.random_state, device=self.device
        )

        cv_scores = cross_val_score(
            model1, self.X_train, self.y_train, cv=self.kf, scoring=self.scoring_str
        )
        print(f"Mean CV score: {np.mean(cv_scores)}, Std CV score: {np.std(cv_scores)}")

        model2.fit(self.X_train, self.y_train)
        train_score = self.scoring(model2, self.X_train, self.y_train)
        print(f"Training set performance: {train_score}")
        test_score = self.scoring(model2, self.X_test, self.y_test)
        print(f"Test set performance: {test_score}")

    def print_best_params(self):
        print(f"Best parameters found for {self.model_class}: {self.best_params}")

    def _get_physical_cores(self):
        return psutil.cpu_count(logical=False)
