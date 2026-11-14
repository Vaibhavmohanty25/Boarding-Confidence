import math

import numpy as np
import pandas as pd

from models import predict


def metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    error = np.asarray(predicted, dtype=float) - actual
    absolute_error = np.abs(error)
    total_actual = np.abs(actual).sum()

    return {
        "departures": len(actual),
        "mae": float(absolute_error.mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "wape_pct": float(100 * absolute_error.sum() / total_actual)
        if total_actual > 0 else None,
        "mean_error": float(error.mean()),
    }


def interval_radius(actual, predicted, coverage=0.90):
    errors = np.abs(np.asarray(actual) - np.asarray(predicted))
    rank = min(math.ceil((len(errors) + 1) * coverage), len(errors))
    return float(np.sort(errors)[rank - 1])


def forecast(bundle, frame):
    point = predict(bundle["model"], frame)
    seats = frame["seats"].to_numpy()
    capacity = frame["total_capacity"].to_numpy()

    lower = np.maximum(point - bundle["radius"], 0)
    upper = np.minimum(point + bundle["radius"], capacity)

    flag = np.select(
        [
            point >= capacity - 0.5,
            upper >= capacity,
            point > seats,
            upper > seats,
        ],
        [
            "Forecast at capacity",
            "Interval reaches capacity",
            "Forecast above seats",
            "Interval exceeds seats",
        ],
        default="Forecast within seats",
    )

    return pd.DataFrame(
        {"prediction": point, "lower": lower, "upper": upper, "capacity_flag": flag},
        index=frame.index,
    )