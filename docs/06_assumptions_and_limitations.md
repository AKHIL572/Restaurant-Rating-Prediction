# Assumptions & Limitations

This document exists so that anyone consuming the dashboard, the model, or the notebooks knows
exactly what this data can and cannot support. Every claim made elsewhere in this project should
be read alongside this file.

## 1. Data Vintage
The dataset is a **single point-in-time snapshot** (no scrape date field exists, but structural
evidence — currency labels, Zomato feature set — places it around 2018–2019). No trend, growth,
or "over time" claim is valid anywhere in this project. Any dashboard or notebook language
implying "trends" refers to cross-sectional patterns at one point in time, not change over time.

## 2. Geographic Skew
| Country Code | Country | Row Count | % of Data |
|---|---|---|---|
| 1 | India | 8,652 | 90.59% |
| 216 | USA | 434 | 4.54% |
| 215 | UK | 80 | 0.84% |
| 30 | Brazil | 60 | 0.63% |
| 214 | UAE | 60 | 0.63% |
| 189 | South Africa | 60 | 0.63% |
| 148 | New Zealand | 40 | 0.42% |
| 208 | Turkey | 34 | 0.36% |
| 14 | Australia | 24 | 0.25% |
| 162 | Philippines | 22 | 0.23% |
| 94, 184, 166, 191 | Indonesia, Singapore, Qatar, Sri Lanka | 20 each | 0.21% each |
| 37 | Canada | 4 | 0.04% |

**Implication:** Any statistic reported for a country below ~40 rows (9 of the 15 countries) is
descriptive only and must never be used as the basis for a business recommendation. India-level
findings are the only ones with enough volume to be considered reasonably reliable.

## 3. Currency Normalization
`Average Cost for Two` is stored in each restaurant's **local currency**, not a common unit. Mean
cost by country ranges from ~24 (USA, USD) to ~281,190 (Indonesia, IDR) — a difference driven
almost entirely by currency denomination, not real price positioning. **Any cross-country cost
comparison, aggregate KPI, or feature used in modeling must first convert this field to a common
currency (this project uses a static, dated USD conversion table — see conversion rates applied
in Notebook 1).** Numbers not yet converted (e.g., in the current dashboard) should be treated as
provisional and are being corrected as part of this documentation pass.

## 4. Known Source Data Bug
Country Code 162 (Philippines) carries the currency label "Botswana Pula(P)" in the raw data.
This is an artifact of the original Zomato API scrape, not an error introduced by this project.
It is documented here rather than silently altered, so that anyone auditing the cleaning steps
can see the discrepancy was identified and intentionally handled (mapped to the correct currency
for conversion purposes), not missed.

## 5. Zero-Rated Restaurants
2,148 restaurants (22.49%) have `Aggregate rating = 0.0`. The dataset does not indicate whether
this means "not yet reviewed," "delisted," or a data collection gap. These rows are excluded from
model training (a 0.0 rating is not a genuine low rating, it is an absence of rating), but this
is a judgment call, not a certainty, and is flagged as such.

## 6. No Review Text
The dataset contains only a pre-aggregated numeric rating, not the underlying review text. No
sentiment analysis, topic modeling, or qualitative "why" behind a rating is possible with this
data as-is.

## 7. Votes as a Feature
`Votes` (and its log-transformed version) reflects post-launch popularity and is **excluded** from
the pre-launch prediction use case (see `01_problem_statement.md`). Any historical-explanation use
case that does use `Votes` must be clearly labeled as a different, retrospective analysis — not
conflated with a "should I open this restaurant" prediction.

## 8. Model Generalization
Given the 90.59% India skew, the model's reliability for non-India predictions is inherently
lower and must be validated and reported separately (see `07_model_card.md`, Performance by
Segment). Predictions for rare cities/countries should be treated as low-confidence by default.

## 9. Non-Linearity of Geographic Coordinates
Latitude/Longitude are not linearly related to rating; raw correlation coefficients against these
fields (as computed in early EDA) are not a meaningful measure of geographic effect and should
not be cited as evidence of "no geographic effect." A proper geospatial feature (e.g., restaurant
density within a radius, distance to city center) would be required to test this properly, and
is noted as a Future Work item.
