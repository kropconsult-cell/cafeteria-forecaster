"""
app.py — Cafeteria meal-demand forecaster (Streamlit)
=====================================================

Week 10 · Model Evaluation, Deployment, and Monitoring
AI for Science and Engineering · Instructor: Dr. Ian Krop

A simple interface for the cafeteria manager's 4:00 p.m. decision:
enter what is known about tomorrow, get a forecast of meals demanded,
an 80 % range, and warnings when the forecast needs human review.

What this app does NOT do: it never trains or retrains the model.
It only loads the pipeline saved by the Colab notebook (NB-19).

Files needed in the same folder as this script:
    cafeteria_model.joblib   the fitted pipeline (preprocessing + Random Forest)
    model_card.json          features, valid ranges, 80 % range, metrics, limitations
    requirements.txt         exact library versions used to save the model

Run it (in a terminal, in this folder):
    pip install -r requirements.txt
    streamlit run app.py

SYNTHETIC TEACHING EXAMPLE: the model was trained on synthetic data.
"""

import csv
import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import sklearn
import streamlit as st

BASE = Path(__file__).resolve().parent
MODEL_FILE = BASE / "cafeteria_model.joblib"
CARD_FILE = BASE / "model_card.json"
LOG_FILE = BASE / "forecast_log.csv"

st.set_page_config(page_title="Cafeteria demand forecaster", page_icon="🍽️", layout="wide")


# ---------------------------------------------------------------------------
# 1. Load the model card and the model ONCE (cached), with friendly errors
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    """Return (pipeline, card). Raises a readable error if anything is missing or incompatible."""
    if not CARD_FILE.exists() or not MODEL_FILE.exists():
        missing = [p.name for p in (MODEL_FILE, CARD_FILE) if not p.exists()]
        raise FileNotFoundError(
            f"Missing file(s): {', '.join(missing)}. Download cafeteria_deployment.zip from the "
            "Colab notebook (section NB-19), unzip it, and run `streamlit run app.py` inside that folder.")
    with open(CARD_FILE, encoding="utf-8") as f:
        card = json.load(f)
    pipeline = joblib.load(MODEL_FILE)
    # The saved pipeline must expect exactly the features listed in the model card, in the same order.
    expected = list(getattr(pipeline, "feature_names_in_", []))
    if expected != card["features"]:
        raise ValueError("The model and the model card disagree about the input features. "
                         "Re-download both files from the same notebook run.")
    return pipeline, card


try:
    pipeline, card = load_artifacts()
except Exception as err:  # show the problem in the app instead of a stack trace
    st.error(f"**The forecaster could not start.** {err}")
    st.info("If the message mentions a version or 'unpickling' problem, install the exact versions with "
            "`pip install -r requirements.txt` (a model file only reliably loads with the scikit-learn "
            "version that saved it).")
    st.stop()

FEATURES = card["features"]
RANGES = card["training_ranges"]
STATES = card["calendar_states"]
DAYS = card["day_of_week_categories"]
LOW, HIGH = card["prediction_interval_80"]["low_offset"], card["prediction_interval_80"]["high_offset"]

# Example inputs for the classroom demonstration (sidebar). Values are typical of the synthetic data.
EXAMPLES = {
    "Normal term weekday": dict(day="Tuesday", state="teaching", week=6, holiday=False, event=False,
                                students=3800, rain=2.0, temp=24.0, lag1=2200, lag7=2150, roll7=1950),
    "Out of range: very large intake": dict(day="Tuesday", state="teaching", week=6, holiday=False, event=False,
                                            students=5000, rain=2.0, temp=24.0, lag1=2200, lag7=2150, roll7=1950),
    "Special event day": dict(day="Thursday", state="teaching", week=9, holiday=False, event=True,
                              students=3800, rain=0.0, temp=25.0, lag1=2250, lag7=2300, roll7=2000),
    "Break-time Saturday": dict(day="Saturday", state="break", week=0, holiday=False, event=False,
                                students=620, rain=8.0, temp=21.0, lag1=330, lag7=210, roll7=300),
}

# ---------------------------------------------------------------------------
# 2. Page header and sidebar
# ---------------------------------------------------------------------------
st.title("🍽️ Cafeteria meal-demand forecaster")
st.caption("Decision support for the 4:00 p.m. decision: how many meals will students want **tomorrow**? "
           "⚠ Teaching example — the model was trained on **synthetic** data.")

with st.sidebar:
    st.header("Example inputs")
    example_name = st.selectbox("Load an example", list(EXAMPLES), index=0)
    ex = EXAMPLES[example_name]
    st.markdown("---")
    st.markdown("**The 4 p.m. rule:** enter only what is known at 4:00 p.m. today. "
                "Breakfast and lunch are over, so today's demand is already counted.")
    log_forecasts = st.checkbox("Save each forecast to forecast_log.csv (for monitoring)", value=True)

k = example_name  # widget keys include the example name, so switching examples resets the form

# ---------------------------------------------------------------------------
# 3. Inputs (one per model feature; is_weekend and the calendar flags are derived)
# ---------------------------------------------------------------------------
st.subheader("Tomorrow's information")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("**Calendar**")
    day = st.selectbox("Day of the week", DAYS, index=DAYS.index(ex["day"]), key=f"day_{k}")
    is_weekend = int(day in ("Saturday", "Sunday"))
    st.caption(f"Weekend: **{'yes' if is_weekend else 'no'}** (set automatically from the day)")
    state = st.radio("Academic calendar", list(STATES), index=list(STATES).index(ex["state"]),
                     horizontal=True, key=f"state_{k}",
                     help="Choosing one state sets term, exam and break flags consistently.")
    lo_w, hi_w = STATES[state]["semester_week"]
    week = st.number_input(f"Semester week (allowed for '{state}': {lo_w}–{hi_w})", min_value=0, max_value=17,
                           value=int(ex["week"]) if lo_w <= ex["week"] <= hi_w else lo_w, step=1,
                           key=f"week_{k}_{state}")   # resets to a valid week when the calendar state changes
    holiday = st.checkbox("Public holiday", value=ex["holiday"], key=f"hol_{k}")
    event = st.checkbox("Special university event scheduled", value=ex["event"], key=f"event_{k}")

with c2:
    st.markdown("**People and weather**")
    students = st.number_input("Estimated students on campus", min_value=0, max_value=20000,
                               value=int(ex["students"]), step=10, key=f"stud_{k}")
    rain = st.number_input("Forecast rainfall (mm)", min_value=0.0, max_value=300.0,
                           value=float(ex["rain"]), step=0.5, key=f"rain_{k}")
    temp = st.number_input("Forecast temperature (°C)", min_value=-10.0, max_value=50.0,
                           value=float(ex["temp"]), step=0.5, key=f"temp_{k}")

with c3:
    st.markdown("**Recent demand (already recorded)**")
    lag1 = st.number_input("Meals demanded today", min_value=0, max_value=20000,
                           value=int(ex["lag1"]), step=10, key=f"lag1_{k}")
    lag7 = st.number_input("Meals demanded 7 days before tomorrow", min_value=0, max_value=20000,
                           value=int(ex["lag7"]), step=10, key=f"lag7_{k}")
    roll7 = st.number_input("Average demand over the last 7 days", min_value=0.0, max_value=20000.0,
                            value=float(ex["roll7"]), step=10.0, key=f"roll7_{k}")

# ---------------------------------------------------------------------------
# 4. Validation: hard errors block the forecast; warnings ask for human review
# ---------------------------------------------------------------------------
flags = STATES[state]
row = {
    "day_of_week": day, "is_weekend": is_weekend, "students_on_campus": int(students),
    "exam_period": flags["exam_period"], "public_holiday": int(holiday), "special_event": int(event),
    "academic_break": flags["academic_break"], "term_in_session": flags["term_in_session"],
    "semester_week": int(week), "forecast_rainfall_mm": float(rain), "forecast_temperature_c": float(temp),
    "meals_demanded_lag_1": float(lag1), "meals_demanded_lag_7": float(lag7),
    "meals_demanded_rolling_7d": float(roll7),
}

errors, warnings = [], []
if not lo_w <= week <= hi_w:
    errors.append(f"Semester week {week} is not possible during '{state}' (allowed: {lo_w}–{hi_w}).")
if set(row) != set(FEATURES):
    errors.append("Internal error: the form does not provide exactly the features the model expects.")
for name, value in row.items():
    if name != "day_of_week" and value is None:
        errors.append(f"Missing value for {name}.")

for name, value in row.items():
    if name in RANGES and not RANGES[name]["min"] <= value <= RANGES[name]["max"]:
        warnings.append(f"**{name} = {value:g}** is outside the range seen in training "
                        f"({RANGES[name]['min']:g}–{RANGES[name]['max']:g}). The model has no experience here; "
                        "Random Forests cannot extrapolate.")
if event:
    warnings.append("**Special event:** there are few similar days in the training data, so this forecast is less reliable.")
if row["meals_demanded_lag_1"] > 0 and not 0.2 <= row["meals_demanded_rolling_7d"] / max(row["meals_demanded_lag_1"], 1) <= 5:
    warnings.append("The 7-day average looks inconsistent with today's demand. Please double-check the recent-demand inputs.")
if state == "break" and event:
    warnings.append("Special events in a break did not occur in the training data.")

# ---------------------------------------------------------------------------
# 5. Forecast
# ---------------------------------------------------------------------------
st.markdown("---")
if errors:
    for e in errors:
        st.error(e)
    st.stop()

X = pd.DataFrame([row], columns=FEATURES)          # exactly the training feature order
forecast = float(pipeline.predict(X)[0])
low, high = max(0.0, forecast + LOW), forecast + HIGH

left, right = st.columns([1, 2])
with left:
    st.metric("Estimated meals demanded tomorrow", f"{forecast:,.0f}")
    st.write(f"**80 % range:** {low:,.0f} – {high:,.0f} meals")
    st.caption(f"On similar past days, about 80 % of actual demand fell in this range "
               f"(on later test days it caught {card['prediction_interval_80']['test_coverage']:.0%}).")
with right:
    if warnings:
        st.warning("**Human review recommended before acting on this forecast:**\n\n- " + "\n- ".join(warnings))
    else:
        st.success("Inputs are within the conditions the model was trained and tested on.")
    st.info("This is an **estimate**. Combine it with your own knowledge of tomorrow "
            "(menu, events, anything the model cannot know). You make the final decision.")

if log_forecasts:
    new_file = not LOG_FILE.exists()
    try:
        with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(["logged_at"] + FEATURES + ["forecast", "range_low", "range_high", "review_flags"])
            writer.writerow([datetime.now().isoformat(timespec="seconds")] + [row[c] for c in FEATURES]
                            + [round(forecast), round(low), round(high), len(warnings)])
        st.caption("Forecast saved to forecast_log.csv. Add actual demand later to monitor performance (Section 5).")
    except OSError:
        st.caption("Could not write forecast_log.csv (read-only folder). The forecast is still valid.")

# ---------------------------------------------------------------------------
# 6. Model card summary (documentation and transparency)
# ---------------------------------------------------------------------------
with st.expander("About this model (model card)"):
    m = card["metrics"]
    st.markdown(f"""
**{card['model_name']}**, version {card['version']} (created {card['created']})

- **Intended use:** {card['intended_use']}
- **Not intended for:** {'; '.join(card['not_intended_for'])}
- **Trained on:** {card['data']['training_period'][0]} to {card['data']['training_period'][1]} ({card['data']['source']})
- **Typical error:** validation MAE {m['validation']['MAE']:.0f} meals; test MAE {m['test']['MAE']:.0f} meals
- **Monitoring:** {card['monitoring']['rule']} (threshold {card['monitoring']['alert_threshold_mae']:.0f} meals)
- **Known limitations:** {'; '.join(card['limitations'])}
""")
    saved_with = card["software"]["scikit-learn"]
    if saved_with != sklearn.__version__:
        st.warning(f"Version mismatch: the model was saved with scikit-learn {saved_with}, "
                   f"but this computer has {sklearn.__version__}. Install the versions in requirements.txt.")
    else:
        st.caption(f"scikit-learn {sklearn.__version__} matches the version used to save the model.")
