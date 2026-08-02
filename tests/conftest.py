import pandas as pd
import joblib
import pytest
import sys
from pathlib import Path

# Make Streamlit_App/ and Model/ paths importable/discoverable from tests/
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "Streamlit_App"))


@pytest.fixture(scope="session")
def project_root():
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def model():
    return joblib.load(PROJECT_ROOT / "Model" / "restaurant_rating_model.pkl")


@pytest.fixture(scope="session")
def feature_columns():
    return joblib.load(PROJECT_ROOT / "Model" / "feature_columns.pkl")


@pytest.fixture(scope="session")
def model_metadata():
    return joblib.load(PROJECT_ROOT / "Model" / "model_metadata.pkl")


@pytest.fixture(scope="session")
def cleaned_df():
    return pd.read_csv(PROJECT_ROOT / "Dataset" / "cleaned_dataset.csv")


@pytest.fixture(scope="session")
def rated_df(cleaned_df):
    return cleaned_df[cleaned_df["Aggregate rating"] > 0].copy()
