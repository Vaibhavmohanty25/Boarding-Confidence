from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor
from torch import nn
from xgboost import XGBRegressor

from data import HISTORY, STATIC, TARGET

MODEL_NAMES = ("Historical baseline", "CatBoost", "XGBoost", "LSTM", "BiLSTM")
SEED = 42


def baseline_keys(frame):
    return pd.MultiIndex.from_arrays(
        [
            frame["route_id"],
            frame["direction"],
            frame["departure_minute"] // 60,
            frame["weekday"].ge(5),
        ],
        names=["route_id", "direction", "departure_hour", "is_weekend"],
    )


@dataclass
class HistoricalBaseline:
    medians: pd.Series
    fallback: float

    def predict(self, frame):
        return self.medians.reindex(baseline_keys(frame)).fillna(self.fallback).to_numpy()


def model_inputs(frame):
    static = frame[STATIC].to_numpy(dtype=np.float32)
    ratios = frame[HISTORY].to_numpy(dtype=np.float32)

    present = np.isfinite(ratios)
    values = np.where(present, ratios, 0.0)
    history = np.stack([values, present], axis=-1).astype(np.float32)

    return static, history


def tree_inputs(frame):
    static, history = model_inputs(frame)
    return np.column_stack([static, history.reshape(len(static), 2 * len(HISTORY))])


@dataclass
class TreeModel:
    estimator: object

    def predict(self, frame):
        return self.estimator.predict(tree_inputs(frame))


class LoadRNN(nn.Module):
    def __init__(self, bidirectional):
        super().__init__()
        directions = 2 if bidirectional else 1
        self.recurrent = nn.LSTM(
            input_size=2,
            hidden_size=24,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.head = nn.Sequential(
            nn.Linear(24 * directions + len(STATIC), 24),
            nn.ReLU(),
            nn.Linear(24, 1),
        )

    def forward(self, static, history):
        _, (hidden, _) = self.recurrent(history)
        summary = hidden.transpose(0, 1).flatten(start_dim=1)
        return self.head(torch.cat([summary, static], dim=1)).squeeze(-1)


@dataclass
class RecurrentModel:
    network: LoadRNN
    static_mean: np.ndarray
    static_scale: np.ndarray
    target_mean: float
    target_scale: float

    def predict(self, frame):
        static, history = model_inputs(frame)
        standardized = (static - self.static_mean) / self.static_scale
        self.network.eval()

        with torch.inference_mode():
            estimate = self.network(
                torch.from_numpy(standardized), torch.from_numpy(history)
            ).numpy()

        return estimate * self.target_scale + self.target_mean


def fit_recurrent(frame, bidirectional):
    static, history = model_inputs(frame)
    target = frame[TARGET].to_numpy(dtype=np.float32)

    mean = static.mean(axis=0)
    scale = static.std(axis=0)
    scale[scale == 0] = 1
    target_mean = float(target.mean())
    target_scale = float(target.std()) or 1.0

    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy((static - mean) / scale),
        torch.from_numpy(history),
        torch.from_numpy((target - target_mean) / target_scale),
    )

    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    network = LoadRNN(bidirectional)
    batches = torch.utils.data.DataLoader(
        dataset, batch_size=512, shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    optimizer = torch.optim.Adam(network.parameters(), lr=0.003)
    loss_function = nn.MSELoss()

    for _ in range(24):
        network.train()
        for departure, earlier_loads, actual in batches:
            optimizer.zero_grad()
            prediction = network(departure, earlier_loads)
            loss = loss_function(prediction, actual)
            loss.backward()
            optimizer.step()

    return RecurrentModel(network, mean, scale, target_mean, target_scale)


def fit_model(name, frame):
    if name == "Historical baseline":
        loads = pd.Series(frame[TARGET].to_numpy(), index=baseline_keys(frame))
        return HistoricalBaseline(
            loads.groupby(level=loads.index.names).median(), float(loads.median())
        )

    if name in ("LSTM", "BiLSTM"):
        return fit_recurrent(frame, bidirectional=(name == "BiLSTM"))

    if name == "CatBoost":
        estimator = CatBoostRegressor(
            iterations=300, depth=5, learning_rate=0.05,
            loss_function="RMSE", random_seed=SEED, thread_count=2,
            verbose=False, allow_writing_files=False,
        )
    elif name == "XGBoost":
        estimator = XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            objective="reg:squarederror", tree_method="hist",
            random_state=SEED, n_jobs=2,
        )
    else:
        raise ValueError(f"Unknown model: {name}")

    target = frame[TARGET].to_numpy(dtype=np.float32)
    return TreeModel(estimator.fit(tree_inputs(frame), target))


def predict(model, frame):
    return np.clip(model.predict(frame), 0, frame["total_capacity"].to_numpy())