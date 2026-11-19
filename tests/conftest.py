import pandas as pd
import joblib
import pytest
import sys
from pathlib import Path

# Make streamlit_app/ and model/ paths importable/discoverable from tests/
PROJECT_ROOT = Path(__file__).parent.parent

streamlit_dir = (
    PROJECT_ROOT / "streamlit_app"
    if (PROJECT_ROOT / "streamlit_app").exists()
    else PROJECT_ROOT / "Streamlit_App"
)
sys.path.insert(0, str(streamlit_dir))

MODEL_DIR = (
    PROJECT_ROOT / "model"
    if (PROJECT_ROOT / "model").exists()
    else PROJECT_ROOT / "Model"
)
DATASET_DIR = (
    PROJECT_ROOT / "Dataset"
    if (PROJECT_ROOT / "Dataset").exists()
    else PROJECT_ROOT / "dataset"
)


@pytest.fixture(scope="session")
def project_root():
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def model():
    return joblib.load(MODEL_DIR / "restaurant_rating_model.pkl")


@pytest.fixture(scope="session")
def feature_columns():
    return joblib.load(MODEL_DIR / "feature_columns.pkl")


@pytest.fixture(scope="session")
def model_metadata():
    return joblib.load(MODEL_DIR / "model_metadata.pkl")


@pytest.fixture(scope="session")
def cleaned_df():
    return pd.read_csv(DATASET_DIR / "cleaned_dataset.csv")


@pytest.fixture(scope="session")
def rated_df(cleaned_df):
    return cleaned_df[cleaned_df["Aggregate rating"] > 0].copy()
