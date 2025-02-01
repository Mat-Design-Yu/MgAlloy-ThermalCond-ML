from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import (
    RBF,
    WhiteKernel,
    ExpSineSquared,
    RationalQuadratic,
    Matern,
)
from skopt.space import Real, Categorical, Integer

from base_bayesian_optimization import BaseBayesianOptimization


class GPROpt(BaseBayesianOptimization):
    def __init__(self, kernel=RBF(), scoring="r2", n_iter=40, cv=10, random_state=42):
        search_spaces = {
            "alpha": Real(1e-12, 1e-2, prior="log-uniform"),
            "n_restarts_optimizer": Integer(0, 10),
        }
        super().__init__(
            GPR,
            model_params={"kernel": kernel},
            scoring=scoring,
            search_spaces=search_spaces,
            n_iter=n_iter,
            cv=cv,
            random_state=random_state,
        )
