from xgboost import XGBRegressor as XGB
from skopt.space import Real, Categorical, Integer

from base_bayesian_optimization import BaseBayesianOptimization


class XGBOpt(BaseBayesianOptimization):
    def __init__(self, scoring="r2", n_iter=40, cv=10, random_state=42):
        search_spaces = {
            "n_estimators": Integer(10, 100),
            "max_depth": Integer(1, 50),
            "learning_rate": Real(0.01, 1.0, prior="log-uniform"),
            "subsample": Real(0.5, 1.0, prior="uniform"),
            "gamma": Real(0, 10, prior="uniform"),
            "min_child_weight": Integer(1, 10),
            "colsample_bytree": Real(0.1, 1.0, prior="uniform"),
            "reg_alpha": Real(1e-1, 100, prior="uniform"),
            "reg_lambda": Real(1e-1, 100, prior="uniform"),
        }
        super().__init__(
            XGB,
            scoring=scoring,
            search_spaces=search_spaces,
            n_iter=n_iter,
            cv=cv,
            random_state=random_state,
        )
