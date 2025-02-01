from skopt import BayesSearchCV
from sklearn.model_selection import cross_validate
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import clone
from tqdm import tqdm
import numpy as np
import pandas as pd
import psutil
import warnings
from packages.utils.feature_name_clean import clean_feature_names


class BaseBayesianOptimization:
    def __init__(self, model_class, **kwargs):
        self.model_class = model_class
        self.model_params = kwargs.get("model_params", {})
        self.scoring = kwargs.get("scoring", "r2")
        self.search_spaces = kwargs.get("search_spaces")
        self.n_iter = kwargs.get("n_iter", 80)
        self.cv = kwargs.get("cv", 10)
        self.random_state = kwargs.get("random_state", 42)
        self.model = self.model_class(**self.model_params)
        if isinstance(self.scoring, str):
            self.scoring = get_scorer(self.scoring)
        self.kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        self.merged_params = {}

    def load(self, train_data_loader, test_data_loader=None):
        if test_data_loader is not None:
            self.X_train = clean_feature_names(train_data_loader.data_X)
            self.y_train = clean_feature_names(train_data_loader.data_y)
            self.X_test = clean_feature_names(test_data_loader.data_X)
            self.y_test = clean_feature_names(test_data_loader.data_y)
        else:
            self.X_train = clean_feature_names(train_data_loader.data_X)
            self.y_train = clean_feature_names(train_data_loader.data_y)
            self.X_test = None
            self.y_test = None

    def optimize(self, info=False):
        n_points = 1
        if info == False:
            pbar = tqdm(
                total=int(self.n_iter / n_points),
                desc="Bayesian Optimization",
                postfix=dict(mean_cv_score=0),
            )

        self.iteration_results = []

        def callback(optimizer_result):
            params_dict = {
                k: v
                for k, v in zip(self.search_spaces.keys(), optimizer_result.x_iters[-1])
            }
            self.iteration_results.append(
                {"params": params_dict, "score": optimizer_result.func_vals[-1]}
            )

            if info is True:
                print(f"Iteration {len(self.iteration_results)}")
                print("Best params:")
                for param, value in params_dict.items():
                    print(f"  {param}: {value}")
                print(f"Mean CV score: {-self.iteration_results[-1]['score']:.4f}")
                print("-" * 50)
            else:
                pbar.set_postfix(
                    mean_cv_score=f"{-self.iteration_results[-1]['score']:.4f}"
                )
                pbar.update(1)

        self.bayes_search = BayesSearchCV(
            self.model,
            self.search_spaces,
            n_iter=self.n_iter,
            random_state=self.random_state,
            scoring=self.scoring,
            cv=self.kf,
            n_jobs=self._get_physical_cores(),
            n_points=n_points,
            pre_dispatch="2*n_jobs",
        )
        self.bayes_search.fit(self.X_train, self.y_train, callback=callback)
        self.best_params = self.bayes_search.best_params_

        self.merged_params = self.model_params.copy()
        self.merged_params.update(self.best_params)

        if info is False:
            pbar.close()

    def mse(self, y_true, y_pred):
        y_true, y_pred = np.array(y_true).ravel(), np.array(y_pred).ravel()
        return np.mean((y_true - y_pred) ** 2)

    def mae(self, y_true, y_pred):
        y_true, y_pred = np.array(y_true).ravel(), np.array(y_pred).ravel()
        return np.mean(np.abs(y_true - y_pred))

    def rmse(self, y_true, y_pred):
        y_true, y_pred = np.array(y_true).ravel(), np.array(y_pred).ravel()
        return np.sqrt(mean_squared_error(y_true, y_pred))

    def mape(self, y_true, y_pred):
        y_true, y_pred = np.array(y_true).ravel(), np.array(y_pred).ravel()
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def r2(self, y_true, y_pred):
        y_true, y_pred = np.array(y_true).ravel(), np.array(y_pred).ravel()
        return r2_score(y_true, y_pred)

    def print_best_params(self):
        print(
            f"Best parameters found for {self.model_class}: {self.bayes_search.best_params_}"
        )
        return self.bayes_search.best_params_

    def evaluate(self):
        if self.merged_params == {}:
            model1 = self.model_class(**self.model_params)
            model2 = self.model_class(**self.model_params)
            model3 = self.model_class(**self.model_params)
        else:
            model1 = self.model_class(**self.merged_params)
            model2 = self.model_class(**self.merged_params)
            model3 = self.model_class(**self.merged_params)

        def custom_scorer(estimator, X, y):
            y_pred = estimator.predict(X)
            rmse = self.rmse(y, y_pred)
            mape = self.mape(y, y_pred)
            r2 = self.r2(y, y_pred)
            mae = self.mae(y, y_pred)
            mse = self.mse(y, y_pred)
            return {"rmse": rmse, "mape": mape, "r2": r2, "mae": mae, "mse": mse}

        cv_scores = cross_validate(
            model1,
            self.X_train,
            self.y_train,
            cv=self.kf,
            scoring=custom_scorer,
            n_jobs=1,
        )

        print("Mean CV scores:")
        for metric in ["rmse", "mape", "r2", "mae", "mse"]:
            mean_score = np.mean(cv_scores[f"test_{metric}"])
            std_score = np.std(cv_scores[f"test_{metric}"])
            print(f"{metric.upper()}: {mean_score:.4f} (±{std_score:.4f})")

        model2.fit(self.X_train, self.y_train)
        train_pred = model2.predict(self.X_train)
        test_pred = model2.predict(self.X_test)

        metrics = ["RMSE", "MAPE", "R²", "MAE", "MSE"]
        calc_funcs = [self.rmse, self.mape, self.r2, self.mae, self.mse]
        train_metrics = [func(self.y_train, train_pred) for func in calc_funcs]
        test_metrics = [func(self.y_test, test_pred) for func in calc_funcs]

        results = {
            "Metric": metrics,
            "Training Set": train_metrics,
            "Test Set": test_metrics,
        }
        results_df = pd.DataFrame(results)
        print(results_df)

        self.evaluate_result = {
            "unfit_model": model3,
            "fitted_model": model2,
            "cv_scores": cv_scores,
            "results_df": results_df,
            "train_pred": train_pred,
            "test_pred": test_pred,
            "train_true": self.y_train,
            "test_true": self.y_test,
        }
        self.evaluate_result.update(
            {f"train_{m.lower()}": v for m, v in zip(metrics, train_metrics)}
        )
        self.evaluate_result.update(
            {f"test_{m.lower()}": v for m, v in zip(metrics, test_metrics)}
        )

        if self.merged_params != {}:
            self.evaluate_result["best_params"] = self.bayes_search.best_params_

        return self.evaluate_result

    def _get_physical_cores(self):
        return psutil.cpu_count(logical=False)
