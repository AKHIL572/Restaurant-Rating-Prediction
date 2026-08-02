"""
Data integrity tests — these guard the assumptions every downstream notebook,
the dashboard, and the app all depend on. If cleaned_dataset.csv is ever
regenerated (e.g. Notebook 1 is re-run against a refreshed raw extract), these
tests should be run before trusting anything built on top of it.
"""
import pandas as pd


def test_no_duplicate_restaurant_ids(cleaned_df):
    assert cleaned_df["Restaurant ID"].duplicated().sum() == 0


def test_rating_within_valid_range(cleaned_df):
    assert cleaned_df["Aggregate rating"].between(0, 5).all()


def test_price_range_within_valid_categories(cleaned_df):
    assert cleaned_df["Price range"].isin([1, 2, 3, 4]).all()


def test_latitude_within_valid_bounds(cleaned_df):
    assert cleaned_df["Latitude"].between(-90, 90).all()


def test_longitude_within_valid_bounds(cleaned_df):
    assert cleaned_df["Longitude"].between(-180, 180).all()


def test_binary_flags_are_zero_or_one(cleaned_df):
    for col in ["Has Table booking", "Has Online delivery", "Is delivering now"]:
        assert set(cleaned_df[col].unique()).issubset(
            {0, 1}), f"{col} has non-binary values"


def test_currency_normalization_column_exists(cleaned_df):
    """Guards against a future edit accidentally reverting to raw, un-normalized cost."""
    assert "Average Cost for Two (USD)" in cleaned_df.columns


def test_currency_normalization_produces_comparable_scale(cleaned_df):
    """
    Regression test for the currency bug found during review: raw local-currency
    cost ranged from ~24 (USA) to ~281,190 (Indonesia) purely due to denomination.
    After normalization, no single country's mean cost should be more than ~20x
    another's -- a much saner range than the ~11,700x raw gap.
    """
    means = cleaned_df.groupby("Country Name")[
        "Average Cost for Two (USD)"].mean()
    assert means.max() / means.min() < 20, (
        "Normalized cost spread across countries is implausibly large -- "
        "check whether raw currency values leaked back in."
    )


def test_no_missing_cuisines(cleaned_df):
    """Notebook 1 fills missing Cuisines with 'Unknown' -- should never be null downstream."""
    assert cleaned_df["Cuisines"].isnull().sum() == 0


def test_constant_column_removed(cleaned_df):
    assert "Switch to order menu" not in cleaned_df.columns


def test_india_still_dominant_share(cleaned_df):
    """
    Sanity check on the known geographic skew (90.59% India). This isn't a bug to
    fix -- it's a documented characteristic of the source data (see
    docs/06_assumptions_and_limitations.md) -- but if this ratio changes
    dramatically after a data refresh, every "reliable" (n>=40) country
    threshold decision in the docs/notebooks should be revisited.
    """
    india_share = (cleaned_df["Country Name"] == "India").mean()
    assert 0.80 <= india_share <= 0.99
