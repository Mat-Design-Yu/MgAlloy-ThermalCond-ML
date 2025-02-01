from sklearn.ensemble import GradientBoostingRegressor as GBM
from skopt.space import Real, Categorical, Integer

from base_bayesian_optimization import BaseBayesianOptimization


class GBMOpt(BaseBayesianOptimization):
    def __init__(self, scoring="r2", n_iter=40, cv=10, random_state=42):
        search_spaces = {
            "n_estimators": Integer(100, 1000),
            "max_depth": Integer(1, 15),
            "learning_rate": Real(0.01, 1.0, prior="log-uniform"),
            "max_features": Real(0.01, 1.0, prior="uniform"),
            "min_samples_split": Integer(2, 20),
            "min_samples_leaf": Integer(1, 10),
            "subsample": Real(0.5, 1.0, prior="uniform"),
            "alpha": Real(0.1, 0.99, prior="uniform"),
        }
        super().__init__(
            GBM,
            scoring=scoring,
            search_spaces=search_spaces,
            n_iter=n_iter,
            cv=cv,
            random_state=random_state,
        )
