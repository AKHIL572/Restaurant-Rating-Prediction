# Data Dictionary — `Res_dataset.csv`

Source: Zomato public API extract, single snapshot (2018–2019 vintage, based on dataset
characteristics). 9,551 rows, 21 raw columns, 0 duplicate `Restaurant ID` values.

| Column | Type | Description | Notes / Known Issues |
|---|---|---|---|
| Restaurant ID | int | Unique restaurant identifier | No duplicates found |
| Restaurant Name | string | Restaurant's listed name | Dropped before modeling (high cardinality, leakage risk via brand recognition) |
| Country Code | int (categorical) | Numeric code for country, 15 distinct values | 90.59% of rows are Code 1 (India); see `06_assumptions_and_limitations.md` for full distribution |
| City | string | City name, 141 distinct values | Extremely long-tailed: New Delhi alone = 5,473 rows (57.3%) |
| Address | string | Full street address | Dropped before modeling (unique per row, not a usable feature) |
| Locality | string | Neighborhood-level location | Dropped before modeling (high cardinality) |
| Locality Verbose | string | Locality + city combined | Dropped before modeling (redundant with Locality + City) |
| Longitude | float | Geographic longitude | Not linearly related to rating; raw correlation is not meaningful (see `06`) |
| Latitude | float | Geographic latitude | Same caveat as Longitude |
| Cuisines | string | Comma-separated list of cuisines served | 9 missing values (0.09%), filled as "Unknown"; 145 distinct individual cuisines, top 20 used for modeling |
| Average Cost for Two | int | Cost for two people, in **local currency** | **Not currency-normalized** — values range from ~24 (USA, USD) to ~281,190 (Indonesia, IDR) mean by country. Must not be compared/aggregated across countries without conversion. |
| Currency | string | Currency label for the row's `Average Cost for Two` value | Country Code 162 (Philippines) is mislabeled "Botswana Pula(P)" in the source data — a known scrape-level bug, not a project error; documented here rather than silently corrected |
| Has Table booking | Yes/No → 0/1 | Whether the restaurant offers table booking | Converted to binary in cleaning |
| Has Online delivery | Yes/No → 0/1 | Whether the restaurant offers online delivery | Converted to binary in cleaning |
| Is delivering now | Yes/No → 0/1 | Whether delivery is currently active | 99.6% "No" — near-constant, low signal value |
| Switch to order menu | Yes/No | Legacy Zomato feature flag | Constant across all rows — dropped in cleaning |
| Price range | int, 1–4 | Coarse price tier (1=Low … 4=Premium) | Used to derive `Cost_Category` — see feature engineering note in `06` re: redundancy |
| Aggregate rating | float, 0.0–4.9 | Target variable — overall customer rating | 22.49% of rows (2,148) have a value of 0.0, meaning of which is not explicit (unrated vs. no reviews vs. delisted) — filtered out for modeling, but not root-caused |
| Rating color | string | Zomato UI color code tied to rating band | Dropped before modeling (derived directly from target — leakage) |
| Rating text | string | Zomato UI text label tied to rating band | Dropped before modeling (same leakage reason) |
| Votes | int | Number of customer votes/reviews received | Mean 202.2 / median 60 / max 10,934 among rated restaurants; **excluded as a model input** for the pre-launch use case (see BRD, BR-3) since it cannot be known before a restaurant has opened |

## Derived Fields (created in Notebook 1 / 3)
| Field | Derivation |
|---|---|
| Cuisine_Count | Count of cuisines listed per restaurant (split on ", ") |
| Log_Votes | `log1p(Votes)` — variance-stabilizing transform |
| Log_Average_Cost | `log1p(Average Cost for Two)` — **inherits the currency issue above; must be recomputed on the normalized cost field, not raw local-currency cost** |
| Cost_Category | Mapped directly from `Price range` (1→Low … 4→Premium) — **redundant with Price range if both are one-hot encoded; use only one in modeling** |
| Cuisine_<name> | One binary column per top-20 most frequent cuisine (North Indian, Chinese, Fast Food, Mughlai, Italian, and 15 others) |

## Distribution Snapshot (for reference)
- Countries: 15, but 90.59% India (Code 1); next largest is USA (Code 216) at 4.54%; nine
  countries individually represent under 0.5% of rows each.
- Rating: 22.49% zero-rated; among the 7,403 rated restaurants, mean = 3.44, median = 3.4.
- Top 5 cuisines by frequency: North Indian (3,960), Chinese (2,735), Fast Food (1,986),
  Mughlai (995), Italian (764).
