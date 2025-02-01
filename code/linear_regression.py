from sklearn.linear_model import LinearRegression as LR
from skopt.space import Real, Categorical, Integer

from base_bayesian_optimization import BaseBayesianOptimization


class LROpt(BaseBayesianOptimization):
    def __init__(self, scoring="r2", n_iter=40, cv=10, random_state=42):
        search_spaces = {
            "fit_intercept": Categorical([True, False]),
            "positive": Categorical([True, False]),
        }
        super().__init__(
            LR,
            scoring=scoring,
            search_spaces=search_spaces,
            n_iter=n_iter,
            cv=cv,
            random_state=random_state,
        )
