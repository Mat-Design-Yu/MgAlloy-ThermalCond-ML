from sklearn.tree import DecisionTreeRegressor as DT
from skopt.space import Real, Categorical, Integer

from base_bayesian_optimization import BaseBayesianOptimization


class DTOpt(BaseBayesianOptimization):
    def __init__(self, scoring="r2", n_iter=40, cv=10, random_state=42):
        search_spaces = {
            "max_depth": Integer(1, 15),
            "min_samples_split": Integer(2, 100),
            "min_samples_leaf": Integer(1, 50),
            "max_features": Categorical([None, "sqrt", "log2"]),
            "splitter": Categorical(["best", "random"]),
            "min_impurity_decrease": Real(0.0, 10.0, prior="uniform"),
        }
        super().__init__(
            DT,
            scoring=scoring,
            search_spaces=search_spaces,
            n_iter=n_iter,
            cv=cv,
            random_state=random_state,
        )
