# User Stories

Format: As a [persona], I want [capability], so that [benefit]. Acceptance criteria listed below
each story map directly to the SRS functional requirements (FR-#).

---

### US-01 — Pre-launch rating estimate
**As a** restaurant owner planning a new outlet,
**I want to** input my planned city, cuisine, price range, and service options,
**so that** I can get a directional rating estimate before committing to a concept.

**Acceptance Criteria**
- Maps to FR-1, FR-2.
- Does not require me to input review/vote data I couldn't possibly have yet.
- Shows the estimate with an honest error margin, not a bare number.

---

### US-02 — Low-confidence input warning
**As a** restaurant owner entering a small or unusual city,
**I want to** be warned if my city has very little training data behind it,
**so that** I don't over-trust a prediction built on a handful of examples.

**Acceptance Criteria**
- Maps to FR-3.
- Warning appears inline, not buried in a tooltip.

---

### US-03 — Unsupported cuisine transparency
**As a** restaurant owner offering a niche cuisine,
**I want to** know if my cuisine choice isn't actually used by the model,
**so that** I don't assume my selection is influencing the prediction when it isn't.

**Acceptance Criteria**
- Maps to FR-4.

---

### US-04 — Currency-consistent benchmarking
**As a** regional operations analyst,
**I want to** view cost-related KPIs in a single normalized currency,
**so that** I can compare restaurant pricing across countries without being misled by raw local-currency values.

**Acceptance Criteria**
- Maps to FR-7.
- No dashboard KPI mixes currencies without conversion.

---

### US-05 — Statistically honest segmentation
**As a** regional operations analyst,
**I want to** see the sample size behind any per-country or per-city statistic,
**so that** I don't act on a conclusion drawn from 4 restaurants as if it were drawn from 4,000.

**Acceptance Criteria**
- Maps to FR-8, BR-2.

---

### US-06 — Actionable, quantified recommendations
**As a** business decision-maker reviewing the insights notebook,
**I want** each recommendation to include a supporting number and a correlation-vs-causation label,
**so that** I can judge how much weight to put behind each one.

**Acceptance Criteria**
- Maps to BRD Acceptance Criteria, section 7.
- Every recommendation includes at least one statistic (effect size, mean difference, or
  correlation coefficient) and states whether it is correlational or has causal support.

---

### US-07 — Reproducible analysis
**As a** technical reviewer (interviewer, teammate, or future self),
**I want** the repository to run end-to-end from a fresh clone with pinned dependencies,
**so that** I can verify the results without debugging environment issues.

**Acceptance Criteria**
- Maps to NFR-2, NFR-4.
- `pip install -r requirements.txt` followed by running notebooks in numbered order reproduces
  `cleaned_dataset.csv`, `model_ready_dataset.csv`, and the trained model artifact.
