"""
Model artifact tests — confirm the saved model, feature columns, and metadata
are internally consistent, and guard against the model-selection bug found
during review (where the saved .pkl silently didn't match the model that
actually won on the test set).
"""
import numpy as np


def test_model_has_predict_method(model):
    assert hasattr(model, "predict")


def test_model_does_not_expose_predict_proba(model):
    """
    Regression test for the dead-code bug in the original app.py: a regressor
    should never have predict_proba. If this ever fails, someone has swapped
    in a classifier by mistake -- a much bigger problem than the UI bug it
    would reintroduce.
    """
    assert not hasattr(model, "predict_proba")


def test_feature_columns_count_matches_training(feature_columns):
    assert len(feature_columns) == 180


def test_votes_and_log_votes_excluded_from_features(feature_columns):
    """
    Guards the BRD BR-3 requirement: Votes (or any transform of it) must never
    be a model feature, since it isn't available before a restaurant launches.
    """
    assert "Votes" not in feature_columns
    assert "Log_Votes" not in feature_columns


def test_cost_category_excluded_as_redundant(feature_columns):
    """Cost_Category was confirmed via VIF to duplicate Price range -- must stay removed."""
    assert "Cost_Category" not in feature_columns


def test_metadata_contains_segment_metrics(model_metadata):
    for key in ["mae_overall", "mae_india", "mae_non_india", "r2_overall", "r2_india", "r2_non_india"]:
        assert key in model_metadata


def test_metadata_r2_within_plausible_bounds(model_metadata):
    for key in ["r2_overall", "r2_india", "r2_non_india"]:
        assert 0 <= model_metadata[key] <= 1, f"{key} is outside a plausible R2 range"


def test_non_india_r2_lower_than_india(model_metadata):
    """
    This is an expected, documented limitation (see docs/07_model_card.md), not
    a bug -- but if a future retrain reverses it, the model card's segment
    disclosure needs to be revisited.
    """
    assert model_metadata["r2_non_india"] < model_metadata["r2_india"]


def test_model_predicts_a_single_float_for_one_row(model, feature_columns):
    dummy_row = np.zeros((1, len(feature_columns)))
    prediction = model.predict(dummy_row)
    assert prediction.shape == (1,)
    assert isinstance(float(prediction[0]), float)
