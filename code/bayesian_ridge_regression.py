from sklearn.linear_model import BayesianRidge
from skopt.space import Real, Categorical, Integer

from base_bayesian_optimization import BaseBayesianOptimization


class BRROpt(BaseBayesianOptimization):
    def __init__(self, scoring="r2", n_iter=40, cv=10, random_state=42):
        search_spaces = {
            "max_iter": Integer(100, 100000),
            "tol": Real(1e-5, 1e-3, prior="log-uniform"),
            "alpha_1": Real(1e-8, 1e-3, prior="log-uniform"),
            "alpha_2": Real(1e-8, 1e-3, prior="log-uniform"),
            "lambda_1": Real(1e-8, 1e-3, prior="log-uniform"),
            "lambda_2": Real(1e-8, 1e-3, prior="log-uniform"),
        }
        super().__init__(
            BayesianRidge,
            scoring=scoring,
            search_spaces=search_spaces,
            n_iter=n_iter,
            cv=cv,
            random_state=random_state,
        )
