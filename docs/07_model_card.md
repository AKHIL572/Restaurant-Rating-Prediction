# Model Card — Restaurant Rating Estimator

## Model Details
- **Type:** Gradient Boosting Regressor (scikit-learn `GradientBoostingRegressor`), selected via
  `RandomizedSearchCV` hyperparameter tuning and 5-fold cross-validation against Random Forest and
  Linear Regression baselines.
- **Version:** 1.0
- **Training data:** 7,403 restaurants (post zero-rating removal) from the Zomato snapshot
  described in `05_data_dictionary.md`.
- **Owner/maintainer:** [your name]

## Intended Use
**Primary intended use:** directional rating estimation for a restaurant at or before launch,
based on planned attributes (city, country, price range, currency-normalized cost, cuisine
selection, table booking / online delivery plans). See `01_problem_statement.md` for the full
use-case definition.

**Not intended for:**
- Predicting rating changes over time (no temporal data exists).
- Explaining an *existing* restaurant's current rating using its actual live vote count — that is
  a distinct, retrospective analysis and would require a separate model trained with `Votes` as a
  legitimate input.
- Country-level policy or investment decisions for any country with under ~40 restaurants in
  the training data (see geographic skew in `06_assumptions_and_limitations.md`).

## Training Features
Country Code, City (one-hot, high cardinality caveat noted below), Price range, currency-
normalized Average Cost (log-transformed), Has Table booking, Has Online delivery, Is delivering
now, Cuisine_Count, top-20 cuisine one-hot flags. `Votes` and any rating-derived field
(`Rating color`, `Rating text`) are excluded as leakage/out-of-scope for this use case.

## Performance
Evaluated on a stratified (India / non-India) held-out test set, n=1,481 (1,303 India / 178
non-India), using the leakage-corrected feature set (no `Votes`/`Log_Votes`) described in
Notebook 3. These are the actual results from the corrected pipeline, not placeholders.

| Metric | Overall Test Set | India Subset | Non-India Subset |
|---|---|---|---|
| MAE | 0.3053 | 0.3070 | 0.2931 |
| RMSE | 0.3930 | 0.3913 | 0.4050 |
| R² | 0.5045 | 0.4078 | 0.2747 |

**Context on these numbers:** an earlier draft of this pipeline that still included `Log_Votes`
(a post-launch popularity signal) reached R² ≈ 0.65. That number was inflated by a feature the
model could never legitimately have at prediction time for a pre-launch restaurant. **0.5045 is
the correct, defensible figure for the stated use case** — a real reviewer should be told this
directly rather than have the higher, leaky number quietly presented as the model's quality.

**Segment-split evaluation confirms a real limitation, not just a formality:** R² is notably
weaker for non-India restaurants (0.2747) than for India (0.4078), even though non-India MAE is
slightly better in absolute terms (0.2931 vs 0.3070) — the model explains less of the rating
*variance* outside India, consistent with the 90.59% India training skew. This is surfaced to
the Streamlit app user as a low-confidence warning for non-India predictions (see SRS FR-3), not
buried in a single blended metric.

**Strongest predictive signal:** `Log_Average_Cost_USD` (36.2% of feature importance), followed
by `Country Name_India` (12.0%) — consistent with the country-level rating gap found in Notebook
2 (India's average rating, 3.352, is notably lower than most other countries in this dataset).

## Known Limitations
1. City is one-hot encoded with 141 categories; predictions for cities with very few training
   examples (most cities outside the top ~10) are lower-confidence and should be flagged to the
   end user (see SRS FR-3).
2. Cuisines outside the top 20 by frequency contribute no signal to the current model (see SRS
   FR-4) — the app must disclose this rather than imply full cuisine coverage.
3. `Cost_Category` was confirmed via VIF analysis (Notebook 2) to be a perfect 1:1 remapping of
   `Price range` and has been **removed** from the final feature set — only `Price range` is used,
   avoiding the double-counting risk flagged in an earlier draft.
4. No prediction interval is currently exposed to the user beyond the point estimate ± MAE; a
   proper quantile regression or bootstrapped interval is a Future Work item for a tighter
   confidence range.

## Ethical Considerations
- `Restaurant Name` is deliberately excluded as a feature to avoid the model learning brand-name
  reputation rather than genuine attribute-driven signal, which would disadvantage new/unknown
  restaurants relative to established chains — consistent with the pre-launch use case.
- Given the India skew, users in under-represented countries should be explicitly told their
  estimate carries wider uncertainty, so the tool doesn't create a false sense of precision for
  markets it barely saw during training.

## Future Work
- Retraining plan: as newer Zomato/other-platform data becomes available, retrain and compare
  against this version's held-out metrics before replacing it (basic model versioning/registry).
- Add geospatial density features in place of raw Latitude/Longitude.
- Investigate zero-rated restaurants' true meaning (unrated vs. delisted) via any available
  metadata, rather than blanket-excluding them.