from hashlib import file_digest

import joblib
import streamlit as st

from data import DATA_PATH, HISTORY, RUN_DIR, TARGET
from data import build_features, load_trips, split_trips
from inference import forecast


@st.cache_data
def load_replay(data_modified):
    return build_features(split_trips(load_trips())["test"])


@st.cache_resource
def load_bundle(model_modified):
    return joblib.load(RUN_DIR / "winner.joblib")


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

replay = load_replay(DATA_PATH.stat().st_mtime_ns)
day = st.selectbox("Held-out service date", replay["service_date"].dt.date.unique())
daily = replay.loc[replay["service_date"].dt.date.eq(day)].set_index("trip_id", drop=False)


def departure_label(trip_id):
    trip = daily.loc[trip_id]
    return f"{trip['scheduled_departure']:%H:%M} | Direction {trip['direction']} | {trip_id}"


trip_id = st.selectbox("Scheduled departure", daily.index, format_func=departure_label)
selected = daily.loc[[trip_id]]
trip = selected.iloc[0]
result = forecast(bundle, selected).iloc[0]

st.write(f"Model: **{bundle['winner']}**")
st.write(f"Forecast issued at **{trip['forecast_at']:%H:%M %Z}**, 30 minutes before departure.")
st.write(f"Earlier reports available: **{trip[HISTORY].notna().sum()} of 4**")
st.metric("Predicted peak onboard", f"{result['prediction']:.1f} passengers")
st.write(f"Nominal 90% interval: **{result['lower']:.1f} to {result['upper']:.1f}**")
st.write(f"Planned seats: **{trip['seats']}** | Total capacity: **{trip['total_capacity']}**")
st.info(result["capacity_flag"])

if st.checkbox("Show the simulated outcome"):
    st.write(f"Observed peak: **{trip[TARGET]} passengers**")
    st.write(f"Prediction minus actual: **{result['prediction'] - trip[TARGET]:+.1f} passengers**")