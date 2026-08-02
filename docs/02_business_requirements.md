# Business Requirements Document (BRD)

## 1. Document Purpose
Defines what this project must deliver from a business standpoint, independent of how it is
technically implemented. Technical implementation is covered in `03_software_requirements.md`.

## 2. Business Objectives
| ID | Objective |
|---|---|
| BO-1 | Provide business stakeholders a reliable, currency-normalized view of restaurant performance across countries, cities, price tiers, and cuisines. |
| BO-2 | Provide a directional rating estimate for a restaurant that has not yet accumulated significant review volume, based on planned attributes. |
| BO-3 | Surface actionable, quantified (not just descriptive) recommendations that a restaurant operator could plausibly act on. |

## 3. Scope

### 3.1 In Scope
- Analysis of the 9,551-restaurant Zomato snapshot across all 15 represented countries.
- Currency-normalized cost analysis (conversion of `Average Cost for Two` to a common baseline,
  USD, using a documented static conversion table — see `06_assumptions_and_limitations.md`).
- A regression model estimating `Aggregate rating` from pre-launch-available attributes only
  (see `01_problem_statement.md`, Primary Use Case).
- A descriptive Power BI dashboard segmented with explicit sample-size disclosure.
- A Streamlit tool restricted to the pre-launch estimation use case, with visible caveats for
  low-reliability inputs (rare cities, uncommon cuisines).

### 3.2 Out of Scope
- Real-time or live Zomato data (this is a static historical extract; no refresh pipeline).
- Sentiment/text analysis of customer reviews (no review text exists in this dataset).
- Time-series or trend analysis (no date/time field exists in this dataset).
- Country-level strategic conclusions for any country with fewer than 40 restaurants in the
  dataset (Country Codes 14, 37, 94, 148, 162, 166, 184, 191, 208 — each under 0.5% of the data)
  — these will be reported descriptively only, never used to justify a business recommendation.

## 4. Business Rules
- BR-1: No cost comparison may combine multiple currencies without conversion.
- BR-2: No per-segment statistic (country, city, cuisine) may be reported without disclosing its
  underlying sample size (n).
- BR-3: The predictive model must never use `Votes` or current live rating as an input feature
  for the pre-launch use case, since these values cannot exist prior to launch.
- BR-4: Any prediction surfaced to a user must be accompanied by the model's documented error
  margin (MAE from `07_model_card.md`), not shown as a bare number.

## 5. Assumptions (business-level; technical assumptions are in file 06)
- The 2018–2019 Zomato snapshot is treated as representative of general rating-driver patterns,
  not as current real-time market data.
- India-specific findings (90.59% of data) are the most statistically reliable; non-India
  findings are directional only given small sample sizes per country.

## 6. Constraints
- No budget/infrastructure for live data ingestion — this is a portfolio/analysis project, not a
  production pipeline with a live Zomato feed.
- Currency conversion rates are static (documented, dated) rather than live FX rates, since the
  underlying data itself is a static historical snapshot.

## 7. Acceptance Criteria
- [ ] All dashboard KPIs are currency-normalized where cost is involved.
- [ ] All dashboard segment-level charts disclose sample size (n) either inline or via tooltip.
- [ ] Model documentation explicitly states the pre-launch use case and its input restrictions.
- [ ] Model performance is reported both overall and split by India / non-India.
- [ ] Every business recommendation in the insights notebook is quantified (has a number attached)
      and labeled as correlational unless a causal claim is explicitly justified.
