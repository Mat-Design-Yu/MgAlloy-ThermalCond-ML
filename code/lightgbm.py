from lightgbm import LGBMRegressor as LGBM
from skopt.space import Real, Categorical, Integer

from base_bayesian_optimization import BaseBayesianOptimization


class LGBMOpt(BaseBayesianOptimization):
    def __init__(self, scoring="r2", n_iter=40, cv=10, random_state=42):
        search_spaces = {
            "n_estimators": Integer(10, 100),
            "max_depth": Integer(1, 15),
            "learning_rate": Real(0.01, 1.0, prior="log-uniform"),
            "num_leaves": Integer(2, 256),
            "min_child_samples": Integer(1, 50),
            "subsample": Real(0.1, 1.0, prior="uniform"),
            "colsample_bytree": Real(0.1, 1.0, prior="uniform"),
            "reg_alpha": Real(1e-1, 100, prior="uniform"),
            "reg_lambda": Real(1e-1, 100, prior="uniform"),
            "max_bin": Integer(2, 1000),
        }
        super().__init__(
            LGBM,
            scoring=scoring,
            search_spaces=search_spaces,
            n_iter=n_iter,
            cv=cv,
            random_state=random_state,
        )
