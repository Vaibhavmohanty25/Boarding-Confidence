import json

import joblib
import pandas as pd
from xgboost import DMatrix, XGBRegressor

from data import HISTORY, RUN_DIR, STATIC, TARGET, build_features, load_trips, split_trips
from models import tree_inputs


def main():
    report = json.loads((RUN_DIR / "results.json").read_text(encoding="utf-8"))
    bundle = joblib.load(RUN_DIR / "winner.joblib")
    replay = pd.read_csv(RUN_DIR / "test_predictions.csv")
    replay["error"] = replay["prediction"] - replay[TARGET]
    replay["absolute_error"] = replay["error"].abs()

    typical_index = (replay["absolute_error"] - replay["absolute_error"].median()).abs().idxmin()
    largest_index = replay["absolute_error"].idxmax()
    examples = replay.loc[[typical_index, largest_index]].copy()
    examples.index = ["Typical error", "Largest error"]

    print(examples[
        [
            "trip_id", "prediction", TARGET, "error", "lower", "upper",
            "released_history", "capacity_flag",
        ]
    ].round(2).to_string())

    print("\nInterval coverage by period")
    print(replay.groupby("period")["covered"].mean().mul(100).round(1).to_string())

    estimator = getattr(bundle["model"], "estimator", None)
    if isinstance(estimator, XGBRegressor):
        test = build_features(split_trips(load_trips())["test"]).set_index("trip_id")
        names = STATIC + [
            label for lag in HISTORY for label in (f"{lag}_ratio", f"{lag}_available")
        ]

        for label, example in examples.iterrows():
            row = test.loc[[example["trip_id"]]]
            values = estimator.get_booster().predict(
                DMatrix(tree_inputs(row)), pred_contribs=True
            )[0]
            contributions = pd.Series(values[:-1], index=names)
            top = contributions.abs().nlargest(4).index
            print(f"\n{label} feature contributions in passengers")
            print(contributions.loc[top].round(2).to_string())

    print(f"\nSelected model: {report['winner']}")


if __name__ == "__main__":
    main()