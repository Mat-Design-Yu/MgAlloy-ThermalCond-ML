from sklearn.svm import SVR
from skopt.space import Real, Categorical, Integer

from base_bayesian_optimization import BaseBayesianOptimization


class SVROpt(BaseBayesianOptimization):
    def __init__(self, kernel=None, scoring="r2", n_iter=40, cv=10, random_state=42):
        if kernel == "linear":
            search_spaces = {
                "C": Real(1e-6, 1e6, prior="log-uniform"),
                "epsilon": Real(1e-6, 1e1, prior="log-uniform"),
                "max_iter": Integer(100, 10000),
            }
        elif kernel == "poly":
            search_spaces = {
                "C": Real(1e-6, 1e6, prior="log-uniform"),
                "epsilon": Real(1e-6, 1e1, prior="log-uniform"),
                "gamma": Real(1e-6, 1e1, prior="log-uniform"),
                "degree": Integer(1, 5),
                "coef0": Real(-10, 10, prior="uniform"),
                "max_iter": Integer(100, 10000),
            }
        elif kernel == "rbf":
            search_spaces = {
                "C": Real(1e-6, 1e6, prior="log-uniform"),
                "epsilon": Real(1e-6, 1e1, prior="log-uniform"),
                "gamma": Real(1e-6, 1e1, prior="log-uniform"),
                "max_iter": Integer(100, 10000),
            }
        elif kernel == "sigmoid":
            search_spaces = {
                "C": Real(1e-6, 1e6, prior="log-uniform"),
                "epsilon": Real(1e-6, 1e1, prior="log-uniform"),
                "gamma": Real(1e-6, 1e1, prior="log-uniform"),
                "coef0": Real(-10, 10, prior="uniform"),
                "max_iter": Integer(100, 10000),
            }
        elif kernel == None:
            search_spaces = {
                "C": Real(1e-6, 1e6, prior="log-uniform"),
                "epsilon": Real(1e-6, 1e1, prior="log-uniform"),
                "gamma": Real(1e-6, 1e1, prior="log-uniform"),
                "degree": Integer(1, 5),
                "coef0": Real(-10, 10, prior="uniform"),
                "kernel": Categorical(["linear", "poly", "rbf", "sigmoid"]),
                "max_iter": Integer(100, 10000),
            }
        super().__init__(
            SVR,
            model_params={"kernel": kernel},
            scoring=scoring,
            search_spaces=search_spaces,
            n_iter=n_iter,
            cv=cv,
            random_state=random_state,
        )
