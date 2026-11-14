from hashlib import file_digest
from importlib.metadata import version

import joblib

from data import DATA_PATH, PERIODS, RUN_DIR, build_features, load_trips, split_trips
from models import MODEL_NAMES, SEED


def main():
    from models import fit_model

    train = build_features(split_trips(load_trips())["train"])

    with DATA_PATH.open("rb") as file:
        dataset_sha256 = file_digest(file, "sha256").hexdigest()

    RUN_DIR.mkdir(exist_ok=True)
    training = {
        "periods": PERIODS,
        "training_departures": len(train),
        "dataset_sha256": dataset_sha256,
        "seed": SEED,
        "versions": {
            package: version(package)
            for package in ("pandas", "pyarrow", "numpy", "catboost", "xgboost", "torch")
        },
        "models": {},
    }

    for number, name in enumerate(MODEL_NAMES, start=1):
        print(f"Training {name} on {len(train):,} departures", flush=True)
        filename = f"model_{number}.joblib"
        model = fit_model(name, train)
        joblib.dump(model, RUN_DIR / filename, compress=3)
        training["models"][name] = filename

    import json

    (RUN_DIR / "training.json").write_text(
        json.dumps(training, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()