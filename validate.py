import json
from hashlib import file_digest

import joblib

from data import DATA_PATH, RUN_DIR, TARGET, build_features, load_trips, split_trips
from inference import interval_radius, metrics
from models import predict


def main():
    training = json.loads((RUN_DIR / "training.json").read_text(encoding="utf-8"))
    with DATA_PATH.open("rb") as file:
        if file_digest(file, "sha256").hexdigest() != training["dataset_sha256"]:
            raise ValueError("The dataset changed. Run training again.")

    blocks = split_trips(load_trips())
    validation = build_features(blocks["validation"])
    scores = {}

    for name, filename in training["models"].items():
        model = joblib.load(RUN_DIR / filename)
        scores[name] = metrics(validation[TARGET], predict(model, validation))

    winner = min(scores, key=lambda name: (scores[name]["mae"], scores[name]["rmse"], name))
    model = joblib.load(RUN_DIR / training["models"][winner])

    calibration = build_features(blocks["calibration"])
    coverage = 0.90
    radius = interval_radius(calibration[TARGET], predict(model, calibration), coverage)

    joblib.dump(
        {
            "model": model,
            "winner": winner,
            "radius": radius,
            "nominal_coverage": coverage,
            "dataset_sha256": training["dataset_sha256"],
        },
        RUN_DIR / "winner.joblib",
        compress=3,
    )

    report = {
        "winner": winner,
        "selection_rule": "Lowest validation MAE, then RMSE, then model name",
        "validation": scores,
        "calibration_departures": len(calibration),
        "interval_radius": radius,
    }
    (RUN_DIR / "validation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    for name, row in scores.items():
        print(f"{name:20} MAE {row['mae']:.3f}")
    print(f"Selected {winner}. Interval radius: {radius:.2f} passengers.")


if __name__ == "__main__":
    main()