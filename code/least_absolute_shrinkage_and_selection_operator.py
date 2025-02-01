from sklearn.linear_model import Lasso
from skopt.space import Real, Categorical, Integer

from base_bayesian_optimization import BaseBayesianOptimization


class LassoOpt(BaseBayesianOptimization):
    def __init__(self, scoring="r2", n_iter=40, cv=10, random_state=42):
        search_spaces = {
            "alpha": Real(1e-1, 100.0, prior="log-uniform"),
            "max_iter": Integer(100, 100000),
            "tol": Real(1e-5, 1e-3, prior="log-uniform"),
            "selection": Categorical(["cyclic", "random"]),
        }
        super().__init__(
            Lasso,
            scoring=scoring,
            search_spaces=search_spaces,
            n_iter=n_iter,
            cv=cv,
            random_state=random_state,
        )
