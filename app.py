import json
from hashlib import file_digest

import joblib
import streamlit as st

from data import DATA_PATH, HISTORY, RUN_DIR, TARGET
from data import build_features, load_trips, split_trips
from inference import forecast
from models import predict


@st.cache_data
def load_replay(data_modified):
    return build_features(split_trips(load_trips())["test"])


@st.cache_resource
def load_bundle(model_modified):
    return joblib.load(RUN_DIR / "winner.joblib")


@st.cache_resource
def load_model(model_path, modified_at):
    return joblib.load(model_path)


st.title("Boarding Confidence")
st.caption("One synthetic route. Use forecasts and capacity flags for human review.")

model_path = RUN_DIR / "winner.joblib"
if not model_path.is_file() or not DATA_PATH.is_file():
    st.error("Add the trip dataset, then run train.py and validate.py.")
    st.stop()

bundle = load_bundle(model_path.stat().st_mtime_ns)

with DATA_PATH.open("rb") as file:
    if file_digest(file, "sha256").hexdigest() != bundle["dataset_sha256"]:
        st.error("This dataset differs from the one used to train the saved model.")
        st.stop()

training_path = RUN_DIR / "training.json"
results_path = RUN_DIR / "results.json"
if not training_path.is_file() or not results_path.is_file():
    st.error("Add the saved training manifest and test results from the run folder.")
    st.stop()

training = json.loads(training_path.read_text(encoding="utf-8"))
results = json.loads(results_path.read_text(encoding="utf-8"))
winner = bundle["winner"]
model_files = training["models"]
if training["dataset_sha256"] != bundle["dataset_sha256"] or results["winner"] != winner:
    st.error("The saved model, training manifest, and test results must come from the same run.")
    st.stop()

model_names = list(model_files)
model_name = st.selectbox("Model", model_names, index=model_names.index(winner))
st.caption(f"Model selected on validation MAE: {winner}")

scores = results["test"][model_name]["overall"]
st.caption(f"Held-out test results across {scores['departures']:,} departures")
mae_col, rmse_col, wape_col = st.columns(3)
mae_col.metric("MAE", f"{scores['mae']:.2f} passengers")
rmse_col.metric("RMSE", f"{scores['rmse']:.2f} passengers")
wape = scores["wape_pct"]
wape_col.metric("WAPE", f"{wape:.2f}%" if wape is not None else "N/A")

replay = load_replay(DATA_PATH.stat().st_mtime_ns)
day = st.selectbox("Held-out service date", replay["service_date"].dt.date.unique())
daily = replay.loc[replay["service_date"].dt.date.eq(day)].set_index("trip_id", drop=False)


def departure_label(trip_id):
    trip = daily.loc[trip_id]
    return f"{trip['scheduled_departure']:%H:%M} | Direction {trip['direction']} | {trip_id}"


trip_id = st.selectbox("Scheduled departure", daily.index, format_func=departure_label)
selected = daily.loc[[trip_id]]
trip = selected.iloc[0]
st.write(f"Forecast issued at **{trip['forecast_at']:%H:%M %Z}**, 30 minutes before departure.")
st.write(f"Earlier reports available: **{trip[HISTORY].notna().sum()} of 4**")
st.write(f"Planned seats: **{trip['seats']}** | Total capacity: **{trip['total_capacity']}**")

if model_name == winner:
    result = forecast(bundle, selected).iloc[0]
    point = result["prediction"]
else:
    candidate_path = RUN_DIR / model_files[model_name]
    if not candidate_path.is_file():
        st.error(f"The saved model file for {model_name} is missing from the run folder.")
        st.stop()
    model = load_model(candidate_path, candidate_path.stat().st_mtime_ns)
    point = predict(model, selected)[0]

st.metric("Predicted peak onboard", f"{point:.1f} passengers")

if model_name == winner:
    coverage = bundle["nominal_coverage"]
    st.write(f"Nominal {coverage:.0%} interval: **{result['lower']:.1f} to {result['upper']:.1f}**")
    st.info(result["capacity_flag"])
else:
    st.caption(f"The prediction interval and capacity flag are available for {winner}.")

if st.checkbox("Show the simulated outcome"):
    st.write(f"Observed peak: **{trip[TARGET]} passengers**")
    st.write(f"Prediction minus actual: **{point - trip[TARGET]:+.1f} passengers**")

