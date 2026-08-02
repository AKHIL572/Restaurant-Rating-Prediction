# Software Requirements Specification (SRS)

## 1. Purpose
Defines functional and non-functional requirements for the two technical deliverables: the
Power BI dashboard and the Streamlit prediction app. Translates the BRD into implementable
requirements.

## 2. System Overview
```
Res_dataset.csv (raw, 9,551 rows, 21 cols)
        |
        v
  [Notebook 1: Data Understanding & Cleaning]  ->  cleaned_dataset.csv
        |
        v
  [Notebook 2: EDA + statistical testing]
        |
        v
  [Notebook 3: Feature Engineering + Modeling]  ->  model_ready_dataset.csv
        |                                            restaurant_rating_model.pkl
        |                                            feature_columns.pkl
        v
  [Notebook 4: Business Insights]
        |
        +----------------------------+
        v                            v
 [Power BI Dashboard]        [Streamlit App: app.py]
 (descriptive/diagnostic)    (predictive, pre-launch use case only)
```

## 3. Functional Requirements — Streamlit App

| ID | Requirement |
|---|---|
| FR-1 | App shall accept only pre-launch-available inputs: country, city, price range, average cost, table booking, online delivery, currently-delivering status, cuisines. `Votes` shall NOT be a user input (see BR-3). |
| FR-2 | App shall display predicted rating alongside the model's documented MAE, e.g. "Estimated Rating: 3.8 ± 0.4". |
| FR-3 | If a selected city has fewer than 30 training examples, app shall display a visible low-confidence warning. |
| FR-4 | If a selected cuisine is outside the top-20 cuisines used in training, app shall display a warning that this cuisine has no measurable effect on the current model. |
| FR-5 | App shall not display any UI element (e.g. "Model Confidence" via `predict_proba`) that does not functionally apply to the underlying regression model. |
| FR-6 | App shall handle prediction errors gracefully with a user-facing message, not a raw stack trace. |

## 4. Functional Requirements — Power BI Dashboard

| ID | Requirement |
|---|---|
| FR-7 | All cost-based KPIs and visuals shall use the currency-normalized cost field, not raw `Average Cost for Two`. |
| FR-8 | Any panel segmenting by country or city shall display or make available the underlying restaurant count (n) for that segment. |
| FR-9 | Dashboard title/labels shall not claim "trend" or "over time" language, since no time dimension exists in the source data. |
| FR-10 | Every visual panel shall render non-empty data or be removed prior to publishing. |
| FR-11 | Dashboard shall display a data "as of" date and source citation. |

## 5. Non-Functional Requirements

| ID | Requirement |
|---|---|
| NFR-1 | App shall load the model and dataset via cached resource/data loaders (already implemented via `st.cache_resource` / `st.cache_data`). |
| NFR-2 | Repository shall include a pinned `requirements.txt` for reproducibility. |
| NFR-3 | Core preprocessing and prediction logic shall have automated test coverage (unit tests). |
| NFR-4 | All file paths shall be relative to project root, not hardcoded to a local machine. |

## 6. Interfaces
- Input: `Dataset/Res_dataset.csv` (static file, not an API).
- Model artifacts: `Model/restaurant_rating_model.pkl`, `Model/feature_columns.pkl` (joblib format).
- Output: Streamlit web UI (local or deployed), Power BI `.pbix` file + exported screenshots.

## 7. Out of Scope (technical)
- Authentication/user accounts.
- Live data refresh / scheduled ETL.
- Multi-language support.
