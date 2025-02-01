from sklearn.neighbors import KNeighborsRegressor as KNN
from skopt.space import Real, Categorical, Integer

from base_bayesian_optimization import BaseBayesianOptimization


class KNNOpt(BaseBayesianOptimization):
    def __init__(self, scoring="r2", n_iter=40, cv=10, random_state=42):
        search_spaces = {
            "n_neighbors": Integer(1, 60),
            "weights": Categorical(["uniform", "distance"]),
            "algorithm": Categorical(["auto", "ball_tree", "kd_tree", "brute"]),
            "leaf_size": Integer(1, 60),
            "p": Integer(1, 5),
            "metric": Categorical(["euclidean", "manhattan", "minkowski"]),
        }
        super().__init__(
            KNN,
            scoring=scoring,
            search_spaces=search_spaces,
            n_iter=n_iter,
            cv=cv,
            random_state=random_state,
        )
