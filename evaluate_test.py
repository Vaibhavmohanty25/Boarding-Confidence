import json
from hashlib import file_digest

import joblib

from data import DATA_PATH, HISTORY, RUN_DIR, TARGET, build_features, load_trips, split_trips
from inference import forecast, metrics
from models import predict


def score_groups(frame, column):
    return {
        str(key): metrics(group[TARGET], group["prediction"])
        for key, group in frame.groupby(column)
    }


def main():
    training = json.loads((RUN_DIR / "training.json").read_text(encoding="utf-8"))
    validation = json.loads((RUN_DIR / "validation.json").read_text(encoding="utf-8"))
    winner = joblib.load(RUN_DIR / "winner.joblib")

    with DATA_PATH.open("rb") as file:
        fingerprint = file_digest(file, "sha256").hexdigest()
    if fingerprint != training["dataset_sha256"] or fingerprint != winner["dataset_sha256"]:
        raise ValueError("The test data and saved models must come from the same run.")
    if winner["winner"] != validation["winner"]:
        raise ValueError("The saved model does not match the validation selection.")

    test = build_features(split_trips(load_trips())["test"])
    test["period"] = test["peak"].map({True: "Peak", False: "Off-peak"})
    test["history_available"] = test[HISTORY].notna().any(axis=1)
    scores = {}

    for name, filename in training["models"].items():
        model = joblib.load(RUN_DIR / filename)
        scored = test.assign(prediction=predict(model, test))
        scores[name] = {
            "overall": metrics(scored[TARGET], scored["prediction"]),
            "by_route": score_groups(scored, "route_id"),
            "by_period": score_groups(scored, "period"),
            "by_direction_and_period": score_groups(scored, ["direction", "period"]),
            "by_history_availability": score_groups(scored, "history_available"),
        }

    columns = [
        "trip_id", "route_id", "direction", "period",
        "scheduled_departure", "forecast_at", "seats", "total_capacity", TARGET,
    ]
    replay = test[columns].join(forecast(winner, test))
    replay["released_history"] = test[HISTORY].notna().sum(axis=1)
    replay["covered"] = replay[TARGET].between(replay["lower"], replay["upper"])
    replay.to_csv(RUN_DIR / "test_predictions.csv", index=False)

    results = {
        "winner": winner["winner"],
        "validation": validation["validation"],
        "test": scores,
        "test_interval_coverage": float(replay["covered"].mean()),
    }
    (RUN_DIR / "results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    for name, row in scores.items():
        score = row["overall"]
        print(
            f"{name:20} MAE {score['mae']:.3f}  "
            f"RMSE {score['rmse']:.3f}  WAPE {score['wape_pct']:.2f}%"
        )
    print(f"Selected model interval coverage: {replay['covered'].mean():.1%}")


if __name__ == "__main__":
    main()