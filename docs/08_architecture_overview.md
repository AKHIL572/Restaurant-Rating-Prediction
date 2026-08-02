# Architecture Overview

## End-to-End Pipeline

```
                         Res_dataset.csv
                    (raw, 9,551 rows x 21 cols)
                                |
                                v
        +-------------------------------------------------+
        |  01_data_understanding_and_cleaning.ipynb        |
        |  - duplicate check, constant-column removal      |
        |  - Yes/No -> binary conversion                   |
        |  - Cuisines null handling                         |
        |  - Currency normalization (Country Code ->        |
        |    actual currency -> USD, static 2019 rates)     |
        |  - outlier review + 4 validation assertions       |
        +-------------------------------------------------+
                                |
                                v
                       cleaned_dataset.csv
                                |
                +---------------+----------------+
                v                                v
  +-----------------------------+   +---------------------------------+
  | 02_EDA.ipynb                |   | 03_modeling_and_validation.ipynb |
  | - univariate/bivariate/     |   | - Cuisine_Count, Log_Average_    |
  |   multivariate analysis     |   |   Cost_USD (top-20 cuisine       |
  | - t-test/ANOVA/chi-square   |   |   one-hot; no Log_Votes -- BR-3) |
  |   significance testing      |   | - stratified train/test split    |
  | - VIF multicollinearity     |   |   (India vs non-India)           |
  |   check (Cost_Category      |   | - CV + RandomizedSearchCV        |
  |   found redundant, removed) |   | - final model selection (bug     |
  +-----------------------------+   |   fixed: saved model matches      |
                |                   |   selection logic)               |
                |                   | - segment-split evaluation       |
                |                   |   (India / non-India)            |
                |                   +---------------------------------+
                |                                |
                |                                v
                |                    model_ready_dataset.csv
                |                    restaurant_rating_model.pkl
                |                    feature_columns.pkl
                |                    model_metadata.pkl
                v                                |
  +-----------------------------+                |
  | 04_business_insights.ipynb  |                 |
  | - quantified, confound-      |                |
  |   checked recommendations   |                 |
  | - retracted the online-      |                |
  |   delivery claim (confounded|                 |
  |   by price tier)            |                 |
  | - correlation vs. causation |                 |
  |   labeling on every claim   |                 |
  +-----------------------------+                |
                |                                |
                v                                v
  +-----------------------------+   +---------------------------------+
  | Power BI Dashboard          |   | Streamlit App (app.py)           |
  | (3 pages)                   |   | - pre-launch use case only       |
  | 1. Performance Overview     |   | - no Votes/Log_Votes input       |
  |    (KPIs, map, top cities,  |   |   (BR-3)                         |
  |    price tier donut)        |   | - error margin (+/- MAE) shown   |
  | 2. Rating Drivers &         |   |   with every prediction          |
  |    Segmentation (price tier,|   | - low-confidence warnings for    |
  |    table booking, delivery  |   |   rare cities/countries          |
  |    confound chart, cuisine, |   | - top-20-cuisine-only multiselect|
  |    sample-size-aware        |   +---------------------------------+
  |    country chart)           |
  | 3. Key Findings &           |
  |    Recommendations          |
  |    (exec summary, retraction|
  |    callout, prioritization  |
  |    quadrant, quantified     |
  |    recommendations table,   |
  |    limitations callout)     |
  +-----------------------------+
```

## Tech Stack
- **Language:** Python 3.13
- **Data processing:** pandas, numpy
- **Statistics:** scipy, statsmodels (t-tests, ANOVA, chi-square, VIF)
- **Visualization:** matplotlib, seaborn (notebooks), Power BI (dashboard)
- **Modeling:** scikit-learn (Linear Regression, Decision Tree, Random Forest, Gradient Boosting)
- **Serialization:** joblib
- **App:** Streamlit
- **Testing:** pytest (see `tests/`)
- **Docs:** Markdown (this `docs/` folder)

## Repository Layout (post-rework)
```
Restaurant-Rating-Prediction/
├── docs/                     <- this documentation set
├── Dataset/
│   ├── Res_dataset.csv               (raw)
│   ├── cleaned_dataset.csv           (post Notebook 1)
│   └── model_ready_dataset.csv       (post Notebook 3)
├── Notebook/
│   ├── 01_data_understanding_and_cleaning.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_modeling_and_validation.ipynb
│   └── 04_business_insights.ipynb
├── Dashboard/
│   ├── restaurant_dashboard.pbix     (3 pages, see docs/03_software_requirements.md)
│   └── dashboard_screenshots/
├── Model/
│   ├── restaurant_rating_model.pkl
│   ├── feature_columns.pkl
│   └── model_metadata.pkl
├── Streamlit_App/
│   └── app.py
├── Screenshots/
├── tests/                    <- unit tests (see NFR-3)
├── README.md
├── requirements.txt
└── .gitignore
```