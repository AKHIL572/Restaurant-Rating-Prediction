"""
Pure prediction/feature-engineering logic for the Restaurant Rating Estimator.

Kept separate from app.py so it can be unit-tested without importing Streamlit
or running the UI. app.py imports and calls these functions; it should not
duplicate this logic inline.
"""
import numpy as np
import pandas as pd

MIN_RELIABLE_SAMPLES = 30  # docs/03_software_requirements.md, FR-3


def get_top_n_cuisines(feature_columns):
    """Extract the cuisine names the model actually supports from its feature list.

    Excludes 'Cuisine_Count' explicitly -- it's a numeric feature (how many
    cuisines a restaurant lists), not a one-hot cuisine flag, but it shares the
    'Cuisine_' prefix with the real cuisine columns and would otherwise be
    misread as a cuisine literally named "Count".
    """
    return [
        c.replace("Cuisine_", "")
        for c in feature_columns
        if c.startswith("Cuisine_") and c != "Cuisine_Count"
    ]


def build_input_row(
    feature_columns,
    city,
    country_name,
    price_range,
    avg_cost_usd,
    has_table_booking,
    has_online_delivery,
    is_delivering_now,
    selected_cuisines,
):
    """
    Build a single-row feature DataFrame matching the trained model's expected
    columns exactly. Any City/Country category not present as its own dummy
    column is left at 0, which correctly reproduces the drop_first=True
    baseline behavior used during training (Notebook 3) rather than being an
    omission.
    """
    if not selected_cuisines:
        raise ValueError("At least one cuisine must be selected.")
    if price_range not in (1, 2, 3, 4):
        raise ValueError("price_range must be one of 1, 2, 3, 4.")
    if avg_cost_usd <= 0:
        raise ValueError("avg_cost_usd must be positive.")

    input_row = pd.Series(0, index=feature_columns, dtype=float)

    input_row["Has Table booking"] = int(has_table_booking)
    input_row["Has Online delivery"] = int(has_online_delivery)
    input_row["Is delivering now"] = int(is_delivering_now)
    input_row["Price range"] = price_range
    input_row["Cuisine_Count"] = len(selected_cuisines)
    input_row["Log_Average_Cost_USD"] = np.log1p(avg_cost_usd)

    for cuisine in selected_cuisines:
        col = f"Cuisine_{cuisine}"
        if col in input_row.index:
            input_row[col] = 1

    city_col = f"City_{city}"
    if city_col in input_row.index:
        input_row[city_col] = 1

    country_col = f"Country Name_{country_name}"
    if country_col in input_row.index:
        input_row[country_col] = 1

    return pd.DataFrame([input_row])[feature_columns]


def predict_rating(model, input_df):
    """Run the model and clip to the valid rating range [0, 5]."""
    raw_prediction = model.predict(input_df)[0]
    return float(np.clip(raw_prediction, 0, 5))


def get_reliability_warnings(city, country_name, city_counts, country_counts, metadata):
    """Return a list of human-readable warnings for low-confidence inputs."""
    warnings = []
    city_n = city_counts.get(city, 0)

    if city_n < MIN_RELIABLE_SAMPLES:
        warnings.append(
            f"{city} has only {city_n} training examples (below the "
            f"{MIN_RELIABLE_SAMPLES} reliability threshold). Treat this estimate as low-confidence."
        )
    if country_name != "India":
        warnings.append(
            f"The model is trained on 90.6% India data. For {country_name}, the documented "
            f"test-set R2 drops from 0.41 (India) to {metadata.get('r2_non_india', 0):.2f} "
            "(non-India) -- treat this estimate as directionally indicative, not precise."
        )
    return warnings
