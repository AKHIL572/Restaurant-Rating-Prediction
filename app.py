import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ==========================================
# Page Config
# ==========================================
st.set_page_config(
    page_title="Restaurant Rating Intelligence System",
    layout="wide"
)

st.title("🍽 Restaurant Rating Intelligence System")
st.markdown(
    "Built for Restaurant Analytics Teams | Predict restaurant aggregate rating using Machine Learning"
)
st.divider()

# ==========================================
# Load Model & Assets
# ==========================================
PROJECT_ROOT = Path(__file__).parent
MODEL_PATH = PROJECT_ROOT / "Model" / "restaurant_rating_model.pkl"
FEATURE_PATH = PROJECT_ROOT / "Model" / "feature_columns.pkl"
DATA_PATH = PROJECT_ROOT / "Dataset" / "Cleaned_dataset.csv"


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURE_PATH)
    return model, features


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


model, feature_columns = load_model()
df = load_data()

# ==========================================
# Extract Dropdown Values
# ==========================================
cities = sorted(df["City"].dropna().unique())
countries = df[["Country Code"]].drop_duplicates()
country_mapping = {
    1: "India",
    14: "Australia",
    30: "Brazil",
    37: "Canada",
    94: "Indonesia",
    148: "New Zealand",
    162: "Philippines",
    166: "Qatar",
    184: "Singapore",
    189: "South Africa",
    191: "Sri Lanka",
    208: "Turkey",
    214: "UAE",
    215: "UK",
    216: "USA"
}

available_cuisines = sorted(
    list(
        set(
            cuisine.strip()
            for sublist in df["Cuisines"].dropna().str.split(",")
            for cuisine in sublist
        )
    )
)

# ==========================================
# Layout
# ==========================================
col1, col2 = st.columns(2)

with col1:
    country_name = st.selectbox(
        "Select Country", list(country_mapping.values()))
    country_code = [k for k, v in country_mapping.items() if v ==
                    country_name][0]

    city = st.selectbox("Select City", cities)

    price_range = st.selectbox(
        "Price Range",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "1 - Low",
            2: "2 - Medium",
            3: "3 - High",
            4: "4 - Premium"
        }[x]
    )

    avg_cost = st.number_input("Average Cost for Two", min_value=50, value=500)

with col2:
    has_table_booking = st.selectbox("Table Booking Available", ["No", "Yes"])
    has_online_delivery = st.selectbox(
        "Online Delivery Available", ["No", "Yes"])
    is_delivering_now = st.selectbox("Currently Delivering", ["No", "Yes"])
    votes = st.number_input("Number of Votes", min_value=1, value=100)

st.divider()

selected_cuisines = st.multiselect(
    "Select Cuisines",
    options=available_cuisines,
    default=available_cuisines[:2]
)

st.divider()

# ==========================================
# Prediction Button
# ==========================================
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

with col_btn2:
    predict = st.button("🔮 Predict Rating", use_container_width=True)

# ==========================================
# Prediction Logic
# ==========================================
if predict:

    if len(selected_cuisines) == 0:
        st.error("Please select at least one cuisine.")
        st.stop()

    has_table_booking = 1 if has_table_booking == "Yes" else 0
    has_online_delivery = 1 if has_online_delivery == "Yes" else 0
    is_delivering_now = 1 if is_delivering_now == "Yes" else 0

    cuisine_count = len(selected_cuisines)
    log_votes = np.log1p(votes)
    log_avg_cost = np.log1p(avg_cost)

    cost_category = {
        1: "Low",
        2: "Medium",
        3: "High",
        4: "Premium"
    }[price_range]

    input_dict = {
        "Country Code": country_code,
        "City": city,
        "Price range": price_range,
        "Has Table booking": has_table_booking,
        "Has Online delivery": has_online_delivery,
        "Is delivering now": is_delivering_now,
        "Cuisine_Count": cuisine_count,
        "Log_Votes": log_votes,
        "Log_Average_Cost": log_avg_cost,
        "Cost_Category": cost_category
    }

    input_df = pd.DataFrame([input_dict])

    # Add cuisine one-hot columns
    for col in feature_columns:
        if col.startswith("Cuisine_"):
            cuisine_name = col.replace("Cuisine_", "")
            input_df[col] = 1 if cuisine_name in selected_cuisines else 0

    input_df = pd.get_dummies(input_df)

    # Align columns
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[feature_columns]

    # Prediction
    with st.spinner("Predicting rating..."):
        prediction = model.predict(input_df)[0]

    st.success(f"⭐ Predicted Rating: {round(prediction, 2)} / 5")

    # Business Classification
    if prediction >= 4.5:
        st.info("🏆 Excellent Restaurant")
    elif prediction >= 3.5:
        st.info("👍 Good Restaurant")
    elif prediction >= 2.5:
        st.warning("⚖ Average Restaurant")
    else:
        st.error("🚩 Below Average Restaurant")

    # Confidence (if available)
    if hasattr(model, "predict_proba"):
        try:
            confidence = np.max(model.predict_proba(input_df))
            st.metric("Model Confidence", f"{round(confidence * 100, 2)}%")
        except:
            pass

    # Feature Importance
    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "Feature": feature_columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False).head(10)

        st.subheader("🔍 Top Influential Features")
        st.bar_chart(importance_df.set_index("Feature"))

st.divider()

st.caption(
    "Model Version: 1.0 | Built with Gradient Boosting | Powered by Streamlit")
