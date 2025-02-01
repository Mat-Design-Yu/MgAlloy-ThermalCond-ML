from sklearn.ensemble import RandomForestRegressor as RF
from skopt.space import Real, Categorical, Integer

from base_bayesian_optimization import BaseBayesianOptimization


class RFOpt(BaseBayesianOptimization):
    def __init__(self, scoring="r2", n_iter=40, cv=10, random_state=42):
        search_spaces = {
            "n_estimators": Integer(10, 100),
            "max_features": Real(0.01, 1.0, prior="uniform"),
            "max_depth": Integer(1, 15),
            "min_samples_split": Integer(2, 100),
            "min_samples_leaf": Integer(1, 50),
            "bootstrap": Categorical([True, False]),
        }
        super().__init__(
            RF,
            scoring=scoring,
            search_spaces=search_spaces,
            n_iter=n_iter,
            cv=cv,
            random_state=random_state,
        )
