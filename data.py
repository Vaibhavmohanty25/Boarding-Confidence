from pathlib import Path

import pandas as pd

DATA_PATH = Path("Boarding-Confidence-Data/trips.parquet")
RUN_DIR = Path("run")
TARGET = "peak_onboard"

PERIODS = {
    "train": ("2023-01-01", "2023-08-28"),
    "validation": ("2023-08-29", "2023-10-15"),
    "calibration": ("2023-10-16", "2023-11-13"),
    "test": ("2023-11-14", "2023-12-31"),
}

STATIC = [
    "departure_minute", "weekday", "season", "direction",
    "peak", "seats", "total_capacity",
]
HISTORY = ["lag_4", "lag_3", "lag_2", "lag_1"]
TIMESTAMPS = [
    "scheduled_departure", "forecast_at",
    "schedule_and_capacity_known_at", "load_available_at",
]
REQUIRED = [
    "trip_id", "route_id", "service_date", "direction",
    "weekday", "season", "peak", "seats", "total_capacity",
    TARGET, *TIMESTAMPS,
]


def load_trips():
    trips = pd.read_parquet(DATA_PATH, columns=REQUIRED)

    if trips.empty or trips[REQUIRED].isna().any().any():
        raise ValueError("Trips must have all required fields and outcomes.")
    if trips["trip_id"].duplicated().any():
        raise ValueError("Trip IDs must be unique.")

    for column in TIMESTAMPS:
        trips[column] = pd.to_datetime(trips[column])
    trips["service_date"] = pd.to_datetime(trips["service_date"]).dt.normalize()

    valid_counts = (
        trips["seats"].gt(0)
        & trips["seats"].le(trips["total_capacity"])
        & trips[TARGET].ge(0)
        & trips[TARGET].le(trips["total_capacity"])
    )
    if not valid_counts.all():
        raise ValueError("Found invalid seats, capacity or peak load.")

    lead_time = trips["scheduled_departure"] - trips["forecast_at"]
    if not lead_time.eq(pd.Timedelta(minutes=30)).all():
        raise ValueError("Forecasts must be issued 30 minutes before departure.")
    if not trips["schedule_and_capacity_known_at"].le(trips["forecast_at"]).all():
        raise ValueError("Schedule and capacity must be known at forecast time.")
    if not trips["load_available_at"].gt(trips["scheduled_departure"]).all():
        raise ValueError("Observed loads must be released after departure.")

    return trips


def split_trips(trips):
    blocks = {
        name: trips.loc[trips["service_date"].between(start, end)].copy()
        for name, (start, end) in PERIODS.items()
    }
    if any(block.empty for block in blocks.values()):
        raise ValueError("The dataset must cover all four date blocks.")

    ordered = list(blocks.values())
    for earlier, later in zip(ordered, ordered[1:]):
        if earlier["load_available_at"].max() > later["forecast_at"].min():
            raise ValueError("A label from an earlier block becomes available too late.")
    return blocks


def build_features(trips):
    rows = trips.copy()
    departure = rows["scheduled_departure"]
    rows["departure_minute"] = departure.dt.hour * 60 + departure.dt.minute

    group_columns = ["route_id", "service_date"]
    reports = rows.sort_values(["load_available_at", "trip_id"]).copy()
    reports["load_ratio"] = reports[TARGET] / reports["total_capacity"]

    by_route_and_day = reports.groupby(group_columns, sort=False)["load_ratio"]
    for lag, column in enumerate(reversed(HISTORY)):
        reports[column] = by_route_and_day.shift(lag)

    reports = reports.rename(columns={"load_available_at": "latest_history_release"})

    return pd.merge_asof(
        rows.sort_values(["forecast_at", "trip_id"]),
        reports[group_columns + ["latest_history_release", *HISTORY]],
        left_on="forecast_at",
        right_on="latest_history_release",
        by=group_columns,
        direction="backward",
    )