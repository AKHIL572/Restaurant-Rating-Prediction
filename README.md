# 🍽 Restaurant Rating Prediction

A restaurant rating analysis and pre-launch estimation project built on a real-world Zomato
restaurant extract (9,551 restaurants, 15 countries). Includes statistically-grounded EDA,
leakage-checked regression modeling, a 3-page Power BI dashboard, and a Streamlit estimation tool
— all developed against a written BRD/SRS, not built feature-first.

> **Note on scope:** this project is built on a single 2018–2019 historical snapshot, 90.6%
> weighted toward India. Every finding and every tool in this repo is scoped and caveated
> accordingly — see [`docs/06_assumptions_and_limitations.md`](docs/06_assumptions_and_limitations.md)
> before drawing conclusions beyond what's documented there.

---

## Table of Contents
- [Problem Statement](#problem-statement)
- [Key Findings](#key-findings)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Setup & Installation](#setup--installation)
- [How to Run](#how-to-run)
- [Model Performance](#model-performance)
- [Dashboard](#dashboard)
- [Testing](#testing)
- [Documentation](#documentation)
- [Known Limitations](#known-limitations)
- [Future Work](#future-work)
- [License](#license)

---

## Problem Statement

Restaurant stakeholders currently rely on intuition rather than data when deciding on pricing
tier, delivery/booking services, or where to open a new location. This project supports **one
specific, clearly-scoped use case**: estimating a restaurant's likely rating *before it opens*,
using only attributes that are known in advance (location, price tier, cuisine, planned
services) — never post-launch signals like vote count, which can't exist yet for an unopened
restaurant.

Full detail: [`docs/01_problem_statement.md`](docs/01_problem_statement.md)

---

## Key Findings

- **Price tier is the strongest rating driver** — ANOVA F=522.9, p≈0, η²=0.175 (price tier alone
  explains 17.5% of rating variance). Premium restaurants average 3.89/5 vs. 3.24/5 for low-tier.
- **Table booking has a real, moderate positive association** with rating (Cohen's d=0.32,
  p=1.58e-22).
- **Retracted finding:** an earlier draft of this analysis claimed online delivery improves
  ratings. Controlling for price tier shows the effect direction actually reverses depending on
  price tier — there is no clean, defensible "delivery helps" or "delivery hurts" claim in this
  data. See [`Notebook/04_business_insights.ipynb`](Notebook/04_business_insights.ipynb) for the
  full confound analysis.
- **A currency normalization bug was found and fixed**: the raw dataset's cost field is stored in
  each restaurant's local currency with no conversion — meaning Indonesia appeared ~450x more
  "expensive" than India purely due to currency denomination, not real pricing. Fixed in
  Notebook 1 via a documented static USD conversion.
- India represents 90.6% of the dataset; findings for the other 14 countries are reported with
  explicit sample sizes and are descriptive-only where n < 40 (see BRD).

---

## Architecture

Full pipeline diagram: [`docs/08_architecture_overview.md`](docs/08_architecture_overview.md)

```
Res_dataset.csv → [Clean & Normalize] → [EDA + Stats] → [Model + Segment Eval] → [Insights]
                                                                  |
                                              +-------------------+-------------------+
                                              v                                       v
                                     Power BI Dashboard (3 pages)          Streamlit App (app.py)
```

---

## Repository Structure

```
Restaurant-Rating-Prediction/
├── docs/                     Problem statement, BRD, SRS, user stories, data dictionary,
│                             assumptions/limitations, model card, architecture overview
├── Dataset/
│   ├── Res_dataset.csv               raw Zomato extract
│   ├── cleaned_dataset.csv           post-cleaning + currency normalization (Notebook 1)
│   └── model_ready_dataset.csv       post-feature-engineering (Notebook 3)
├── Notebook/
│   ├── 01_data_understanding_and_cleaning.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_modeling_and_validation.ipynb
│   └── 04_business_insights.ipynb
├── Model/
│   ├── restaurant_rating_model.pkl
│   ├── feature_columns.pkl
│   └── model_metadata.pkl
├── Dashboard/
│   ├── restaurant_dashboard.pbix
│   └── dashboard_screenshots/
├── Streamlit_App/
│   ├── app.py                 UI layer
│   └── predict_utils.py       testable prediction logic (imported by app.py and by tests/)
├── tests/
│   ├── conftest.py
│   ├── test_data_integrity.py
│   ├── test_model_artifacts.py
│   └── test_app_logic.py
├── Screenshots/
├── pytest.ini
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup & Installation

```bash
git clone https://github.com/<your-username>/Restaurant-Rating-Prediction.git
cd Restaurant-Rating-Prediction

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## How to Run

**1. Reproduce the analysis pipeline** (run in order — each notebook depends on the previous
one's output):
```bash
jupyter lab Notebook/
```
Run `01_data_understanding_and_cleaning.ipynb` → `02_EDA.ipynb` →
`03_modeling_and_validation.ipynb` → `04_business_insights.ipynb`.

This regenerates `Dataset/cleaned_dataset.csv`, `Dataset/model_ready_dataset.csv`, and everything
in `Model/`.

**2. Run the Streamlit app:**
```bash
streamlit run Streamlit_App/app.py
```

**3. Open the dashboard:** open `Dashboard/restaurant_dashboard.pbix` in Power BI Desktop.

**4. Run the test suite:**
```bash
pytest
```

---

## Model Performance

Gradient Boosting Regressor, tuned via `RandomizedSearchCV` + 5-fold CV, evaluated on a
stratified (India/non-India) held-out test set (n=1,481).

| Metric | Overall | India | Non-India |
|---|---|---|---|
| MAE | 0.3053 | 0.3070 | 0.2931 |
| RMSE | 0.3930 | 0.3913 | 0.4050 |
| R² | 0.5045 | 0.4078 | 0.2747 |

**Note:** an earlier version of this model reached R² ≈ 0.65 by including vote count as a
feature. That was leakage — a pre-launch restaurant can't have votes yet — and was removed. 0.5045
is the correct, defensible number for the model's actual stated use case. Full detail, including
known limitations and ethical considerations: [`docs/07_model_card.md`](docs/07_model_card.md).

---

## Dashboard

3-page Power BI dashboard:
1. **Performance Overview** — currency-normalized KPIs, restaurant map, top cities, price tier mix
2. **Rating Drivers & Segmentation** — price tier effect, table booking effect, the online-delivery
   confound chart, cuisine breakdown, sample-size-aware country comparison
3. **Key Findings & Recommendations** — executive summary, retraction callout, effort/impact
   prioritization, quantified recommendations table, data limitations

Screenshots: [`Screenshots/`](Screenshots/)

---

## Testing

31 automated tests covering data integrity, model artifact consistency, and the app's prediction
logic — including regression tests for two real bugs found during development (a feature-leakage
bug and a model-selection bug where the saved model didn't match the one that actually won on the
test set).

```bash
pytest -v
```

---

## Documentation

| File | Purpose |
|---|---|
| [`01_problem_statement.md`](docs/01_problem_statement.md) | Business problem, objective, success criteria |
| [`02_business_requirements.md`](docs/02_business_requirements.md) | Scope, business rules, acceptance criteria |
| [`03_software_requirements.md`](docs/03_software_requirements.md) | Functional/non-functional requirements |
| [`04_user_stories.md`](docs/04_user_stories.md) | User stories mapped to requirements |
| [`05_data_dictionary.md`](docs/05_data_dictionary.md) | Every column, type, and known issue |
| [`06_assumptions_and_limitations.md`](docs/06_assumptions_and_limitations.md) | What this data can and can't support |
| [`07_model_card.md`](docs/07_model_card.md) | Intended use, performance, limitations, ethics |
| [`08_architecture_overview.md`](docs/08_architecture_overview.md) | Full pipeline diagram, tech stack |

---

## Known Limitations

- Single point-in-time snapshot (2018–2019) — no trend/time-series analysis possible.
- 90.6% India-weighted; 8 of 15 countries have fewer than 40 restaurants and are excluded from
  business recommendations.
- No review text — no sentiment/NLP analysis possible.
- Model reliability is lower for non-India predictions (R²=0.27 vs 0.41) — surfaced as a warning
  in the Streamlit app.

Full detail: [`docs/06_assumptions_and_limitations.md`](docs/06_assumptions_and_limitations.md).

---

## Future Work

- Geospatial density features in place of raw Latitude/Longitude.
- Investigate the true meaning of zero-rated restaurants (unrated vs. delisted).
- Retraining plan / model registry as newer data becomes available.
- CI pipeline (GitHub Actions) running `pytest` on every push.

---


## Author

Akhil T V
