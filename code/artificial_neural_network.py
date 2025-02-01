import torch
from sklearn.base import BaseEstimator, RegressorMixin
from ax.core.objective import Objective
from ax.core.metric import Metric

from torch_bayesian_optimization import TorchBayesianOptimization


class ArtificialNeuralNetwork(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        hidden_layer_sizes,
        learning_rate,
        weight_decay,
        num_epochs,
        batch_size,
        random_state,
        device,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.model = None
        self.device = device
        self.random_state = random_state
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_state)

    def fit(self, X, y):
        self.model = torch.nn.Sequential(
            torch.nn.Linear(X.shape[1], self.hidden_layer_sizes),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_layer_sizes, 1),
        ).to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        X_tensor = torch.from_numpy(X).float().to(self.device)
        y_tensor = torch.from_numpy(y).float().unsqueeze(1).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        batch_size = min(self.batch_size, len(dataset))
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        for epoch in range(self.num_epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                y_pred = self.model(batch_X)
                loss = torch.nn.functional.mse_loss(y_pred, batch_y)
                loss.backward()
                optimizer.step()

        return self

    def predict(self, X):
        X_tensor = torch.from_numpy(X).float().to(self.device)
        with torch.no_grad():
            y_pred = self.model(X_tensor).cpu().numpy()
        return y_pred


class ANNOpt(TorchBayesianOptimization):
    def __init__(self, scoring="r2", n_iter=40, cv=10, random_state=42, device=None):
        search_spaces = [
            {
                "name": "hidden_layer_sizes",
                "type": "range",
                "bounds": [20, 300],
                "value_type": "int",
            },
            {
                "name": "learning_rate",
                "type": "range",
                "bounds": [0.01, 10.0],
                "value_type": "float",
                "log_scale": True,
            },
            {
                "name": "weight_decay",
                "type": "range",
                "bounds": [0.0001, 0.01],
                "value_type": "float",
                "log_scale": True,
            },
            {
                "name": "num_epochs",
                "type": "range",
                "bounds": [10, 3000],
                "value_type": "int",
            },
            {
                "name": "batch_size",
                "type": "range",
                "bounds": [32, 256],
                "value_type": "int",
            },
        ]
        scoring = {
            scoring: Objective(
                metric=Metric(name=scoring, lower_is_better=False), minimize=False
            ),
        }
        super().__init__(
            ArtificialNeuralNetwork,
            scoring=scoring,
            search_spaces=search_spaces,
            n_iter=n_iter,
            cv=cv,
            random_state=random_state,
            device=device,
        )
