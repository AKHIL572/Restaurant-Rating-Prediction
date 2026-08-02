"""
Tests for Streamlit_App/predict_utils.py -- the pure logic layer behind
app.py. These exercise exactly what app.py calls, without needing to run the
Streamlit UI itself.
"""
import pytest
import numpy as np
from predict_utils import (
    get_top_n_cuisines,
    build_input_row,
    predict_rating,
    get_reliability_warnings,
    MIN_RELIABLE_SAMPLES,
)


def test_get_top_n_cuisines_returns_20(feature_columns):
    cuisines = get_top_n_cuisines(feature_columns)
    assert len(cuisines) == 20
    assert "North Indian" in cuisines


def test_build_input_row_matches_feature_columns_exactly(feature_columns):
    input_df = build_input_row(
        feature_columns=feature_columns,
        city="New Delhi",
        country_name="India",
        price_range=3,
        avg_cost_usd=15.0,
        has_table_booking=1,
        has_online_delivery=0,
        is_delivering_now=0,
        selected_cuisines=["North Indian", "Chinese"],
    )
    assert list(input_df.columns) == list(feature_columns)
    assert input_df.shape == (1, len(feature_columns))


def test_build_input_row_sets_correct_dummy_for_known_city(feature_columns):
    input_df = build_input_row(
        feature_columns=feature_columns, city="New Delhi", country_name="India",
        price_range=2, avg_cost_usd=10.0, has_table_booking=0,
        has_online_delivery=0, is_delivering_now=0, selected_cuisines=["Chinese"],
    )
    assert input_df.loc[0, "City_New Delhi"] == 1


def test_build_input_row_falls_back_to_baseline_for_dropped_category(feature_columns):
    """
    Australia and one city are the drop_first=True baseline categories from
    training (Notebook 3) -- selecting them should leave ALL Country/City dummy
    columns at 0 rather than raising a KeyError or silently doing nothing wrong.
    """
    input_df = build_input_row(
        feature_columns=feature_columns, city="New Delhi", country_name="Australia",
        price_range=2, avg_cost_usd=10.0, has_table_booking=0,
        has_online_delivery=0, is_delivering_now=0, selected_cuisines=["Chinese"],
    )
    country_dummy_cols = [
        c for c in feature_columns if c.startswith("Country Name_")]
    assert input_df.loc[0, country_dummy_cols].sum() == 0


def test_build_input_row_rejects_empty_cuisine_list(feature_columns):
    with pytest.raises(ValueError):
        build_input_row(
            feature_columns=feature_columns, city="New Delhi", country_name="India",
            price_range=2, avg_cost_usd=10.0, has_table_booking=0,
            has_online_delivery=0, is_delivering_now=0, selected_cuisines=[],
        )


def test_build_input_row_rejects_invalid_price_range(feature_columns):
    with pytest.raises(ValueError):
        build_input_row(
            feature_columns=feature_columns, city="New Delhi", country_name="India",
            price_range=7, avg_cost_usd=10.0, has_table_booking=0,
            has_online_delivery=0, is_delivering_now=0, selected_cuisines=["Chinese"],
        )


def test_build_input_row_rejects_non_positive_cost(feature_columns):
    with pytest.raises(ValueError):
        build_input_row(
            feature_columns=feature_columns, city="New Delhi", country_name="India",
            price_range=2, avg_cost_usd=0, has_table_booking=0,
            has_online_delivery=0, is_delivering_now=0, selected_cuisines=["Chinese"],
        )


def test_predict_rating_is_clipped_to_valid_range(model, feature_columns):
    input_df = build_input_row(
        feature_columns=feature_columns, city="New Delhi", country_name="India",
        price_range=4, avg_cost_usd=500.0, has_table_booking=1,
        has_online_delivery=1, is_delivering_now=1, selected_cuisines=["Italian", "Continental"],
    )
    prediction = predict_rating(model, input_df)
    assert 0 <= prediction <= 5


def test_reliability_warning_triggers_for_low_sample_city(model_metadata):
    warnings = get_reliability_warnings(
        city="SomeTinyTownNotInTraining",
        country_name="India",
        city_counts={"New Delhi": 4048},
        country_counts={"India": 6513},
        metadata=model_metadata,
    )
    assert any("training examples" in w for w in warnings)


def test_reliability_warning_triggers_for_non_india_country(model_metadata):
    warnings = get_reliability_warnings(
        city="Sydney",
        country_name="Australia",
        city_counts={"Sydney": 24},
        country_counts={"Australia": 24},
        metadata=model_metadata,
    )
    assert any("non-India" in w or "R2" in w for w in warnings)


def test_no_warnings_for_reliable_india_city(model_metadata):
    warnings = get_reliability_warnings(
        city="New Delhi",
        country_name="India",
        city_counts={"New Delhi": 4048},
        country_counts={"India": 6513},
        metadata=model_metadata,
    )
    assert warnings == []
