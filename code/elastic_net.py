from sklearn.linear_model import ElasticNet
from skopt.space import Real, Categorical, Integer

from base_bayesian_optimization import BaseBayesianOptimization


class ENOpt(BaseBayesianOptimization):
    def __init__(self, scoring="r2", n_iter=40, cv=10, random_state=42):
        search_spaces = {
            "alpha": Real(1e-6, 100.0, prior="log-uniform"),
            "max_iter": Integer(100, 100000),
            "tol": Real(1e-5, 1e-3, prior="log-uniform"),
            "l1_ratio": Real(0.1, 1, prior="uniform"),
        }
        super().__init__(
            ElasticNet,
            scoring=scoring,
            search_spaces=search_spaces,
            n_iter=n_iter,
            cv=cv,
            random_state=random_state,
        )
