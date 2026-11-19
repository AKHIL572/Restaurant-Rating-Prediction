import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

try:
    from streamlit_app.predict_utils import (
        get_top_n_cuisines,
        build_input_row,
        predict_rating,
        get_reliability_warnings,
        MIN_RELIABLE_SAMPLES,
    )
except ModuleNotFoundError:
    from predict_utils import (
        get_top_n_cuisines,
        build_input_row,
        predict_rating,
        get_reliability_warnings,
        MIN_RELIABLE_SAMPLES,
    )

# ==========================================
# Page Config
# ==========================================
st.set_page_config(
    page_title="Restaurant Rating Estimator (Pre-Launch)",
    layout="wide"
)

st.title("🍽 Restaurant Rating Estimator")
st.markdown(
    "**Pre-launch rating estimation tool.** Estimates a restaurant's likely customer rating "
    "*before it opens*, using only attributes you can plan in advance (location, price tier, "
    "cuisine, planned services). It does **not** use review volume or live popularity signals, "
    "since a restaurant that hasn't opened yet cannot have those. "
    "See `docs/01_problem_statement.md` for the full use-case definition."
)
st.divider()

# ==========================================
# Load Model & Assets
# ==========================================
ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = (
    ROOT_DIR / "model" if (ROOT_DIR / "model").exists() else ROOT_DIR / "Model"
)
DATA_DIR = (
    ROOT_DIR / "Dataset"
    if (ROOT_DIR / "Dataset").exists()
    else ROOT_DIR / "dataset"
)

MODEL_PATH = MODEL_DIR / "restaurant_rating_model.pkl"
FEATURE_PATH = MODEL_DIR / "feature_columns.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.pkl"
DATA_PATH = DATA_DIR / "cleaned_dataset.csv"


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURE_PATH)
    metadata = joblib.load(METADATA_PATH)
    return model, features, metadata


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    # rated restaurants only, matches training data
    return df[df["Aggregate rating"] > 0].copy()


try:
    model, feature_columns, metadata = load_model()
    df = load_data()
except FileNotFoundError as e:
    st.error(
        "Required model or data file is missing. Make sure Notebook 3 has been run and "
        f"produced the expected artifacts.\n\nDetails: {e}"
    )
    st.stop()

# ==========================================
# Reference tables (for dropdowns + reliability warnings)
# ==========================================
city_counts = df["City"].value_counts()
country_counts = df["Country Name"].value_counts()
city_country_map = df.groupby("City")["Country Name"].first().to_dict()

TOP_20_CUISINES = get_top_n_cuisines(feature_columns)

countries = sorted(country_counts.index.tolist())

# ==========================================
# Layout — inputs restricted to pre-launch-available attributes only
# (Votes / vote count is intentionally NOT an input field — see BRD BR-3:
#  a restaurant that hasn't opened cannot have a vote count yet.)
# ==========================================
col1, col2 = st.columns(2)

with col1:
    country_name = st.selectbox("Country", countries, index=countries.index(
        "India") if "India" in countries else 0)

    cities_in_country = sorted(
        [c for c, cn in city_country_map.items() if cn == country_name]
    )
    city = st.selectbox("City", cities_in_country if cities_in_country else sorted(
        city_counts.index.tolist()))

    price_range = st.selectbox(
        "Price Range",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "1 - Low", 2: "2 - Medium", 3: "3 - High", 4: "4 - Premium"}[x]
    )

    avg_cost_usd = st.number_input(
        "Average Cost for Two (USD)", min_value=1.0, value=10.0, step=1.0,
        help="Enter cost in US Dollars. The underlying model was trained on currency-normalized "
             "USD values (see docs/06_assumptions_and_limitations.md) — entering a local-currency "
             "figure directly will give a distorted estimate."
    )

with col2:
    has_table_booking = st.selectbox(
        "Planning to offer Table Booking?", ["No", "Yes"])
    has_online_delivery = st.selectbox(
        "Planning to offer Online Delivery?", ["No", "Yes"])
    is_delivering_now = st.selectbox(
        "Delivery active from Day 1?", ["No", "Yes"])

st.divider()

selected_cuisines = st.multiselect(
    "Select Cuisines (only the 20 most common cuisines in the training data are modeled)",
    options=TOP_20_CUISINES,
    default=TOP_20_CUISINES[:2],
    help="A cuisine outside this list has no learned effect on the current model and won't "
         "influence the prediction — see docs/07_model_card.md, Known Limitations."
)

st.divider()

# ==========================================
# Reliability warnings — shown before prediction, not hidden
# ==========================================
warnings = get_reliability_warnings(
    city, country_name, city_counts, country_counts, metadata)

if warnings:
    for w in warnings:
        st.warning(f"⚠️ {w}")

# ==========================================
# Prediction Button
# ==========================================
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict = st.button("🔮 Estimate Rating", use_container_width=True)

# ==========================================
# Prediction Logic
# ==========================================
if predict:

    if len(selected_cuisines) == 0:
        st.error("Please select at least one cuisine.")
        st.stop()

    table_booking_flag = 1 if has_table_booking == "Yes" else 0
    online_delivery_flag = 1 if has_online_delivery == "Yes" else 0
    delivering_now_flag = 1 if is_delivering_now == "Yes" else 0

    try:
        input_df = build_input_row(
            feature_columns=feature_columns,
            city=city,
            country_name=country_name,
            price_range=price_range,
            avg_cost_usd=avg_cost_usd,
            has_table_booking=table_booking_flag,
            has_online_delivery=online_delivery_flag,
            is_delivering_now=delivering_now_flag,
            selected_cuisines=selected_cuisines,
        )
        with st.spinner("Estimating rating..."):
            prediction = predict_rating(model, input_df)
    except Exception as e:
        st.error(
            f"Prediction failed — the input could not be scored by the model. Details: {e}")
        st.stop()

    # Error margin: use the segment-appropriate MAE, not a single blended number
    mae_to_show = metadata["mae_india"] if country_name == "India" else metadata["mae_non_india"]

    st.success(f"⭐ Estimated Rating: {prediction:.2f} ± {mae_to_show:.2f} / 5")
    st.caption(
        f"± value is the model's Mean Absolute Error on held-out {'India' if country_name == 'India' else 'non-India'} "
        "test data (docs/07_model_card.md) — this is an estimate range, not an exact prediction."
    )

    # Business Classification (based on point estimate; bucket boundaries are illustrative,
    # not statistically derived thresholds)
    if prediction >= 4.5:
        st.info("🏆 Estimated: Excellent")
    elif prediction >= 3.5:
        st.info("👍 Estimated: Good")
    elif prediction >= 2.5:
        st.warning("⚖ Estimated: Average")
    else:
        st.error("🚩 Estimated: Below Average")

    # Feature Importance (from the actual loaded model — no hardcoded model assumption)
    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "Feature": feature_columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False).head(10)

        st.subheader(
            "🔍 Top Influential Features (Model-Wide, Not Specific to This Prediction)")
        st.bar_chart(importance_df.set_index("Feature"))

st.divider()

st.caption(
    f"Model: {metadata['model_name']} | "
    f"Test R² = {metadata['r2_overall']:.3f} (India: {metadata['r2_india']:.3f}, "
    f"Non-India: {metadata['r2_non_india']:.3f}) | "
    f"MAE = {metadata['mae_overall']:.3f} | Powered by Streamlit"
)
st.caption(
    "This tool estimates ratings for restaurants that have not yet launched. It should not be "
    "used to explain an existing restaurant's current live rating — see docs/01_problem_statement.md."
)
