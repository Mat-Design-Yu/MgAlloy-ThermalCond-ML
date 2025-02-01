from catboost import CatBoostRegressor as CB
from skopt.space import Real, Categorical, Integer

from base_bayesian_optimization import BaseBayesianOptimization


class CBOpt(BaseBayesianOptimization):
    def __init__(self, scoring="r2", n_iter=40, cv=10, random_state=42):
        search_spaces = {
            "iterations": Integer(10, 100),
            "learning_rate": Real(0.01, 1.0, prior="log-uniform"),
            "depth": Integer(1, 15),
            "l2_leaf_reg": Real(1e-1, 10, prior="log-uniform"),
            "model_size_reg": Real(1e-1, 10, prior="log-uniform"),
            "random_strength": Real(1e-2, 10, prior="log-uniform"),
            "bagging_temperature": Real(0.0, 1.0, prior="uniform"),
        }
        super().__init__(
            CB,
            model_params={"silent": True, "allow_writing_files": False},
            scoring=scoring,
            search_spaces=search_spaces,
            n_iter=n_iter,
            cv=cv,
            random_state=random_state,
        )
