from sklearn.linear_model import Ridge
from skopt.space import Real, Categorical, Integer

from base_bayesian_optimization import BaseBayesianOptimization


class RROpt(BaseBayesianOptimization):
    def __init__(self, scoring="r2", n_iter=40, cv=10, random_state=42):
        search_spaces = {
            "alpha": Real(1e-1, 100.0, prior="log-uniform"),
            "max_iter": Integer(100, 100000),
            "tol": Real(1e-5, 1e-3, prior="log-uniform"),
            "solver": Categorical(
                ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]
            ),
        }
        super().__init__(
            Ridge,
            scoring=scoring,
            search_spaces=search_spaces,
            n_iter=n_iter,
            cv=cv,
            random_state=random_state,
        )
