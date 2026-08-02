# Problem Statement

## Project Title
Restaurant Rating Intelligence — Predictive & Diagnostic Analysis of Zomato Restaurant Performance

## Background
Zomato operates as a restaurant discovery and delivery platform across 15 countries. Restaurant
owners, regional operations teams, and expansion/strategy teams need to understand what
measurable factors are associated with a restaurant's customer rating (`Aggregate rating`), and
whether that rating can be reasonably estimated ahead of certain business decisions (e.g.
adjusting service offerings, evaluating a new outlet concept).

This project uses a historical extract of 9,551 restaurant records pulled from the Zomato API
(single point-in-time snapshot) to (a) diagnose which restaurant attributes are associated with
higher or lower ratings, and (b) build a supervised regression model that estimates a restaurant's
expected `Aggregate rating` from its attributes.

## Business Problem
Restaurant stakeholders currently rely on intuition rather than data when deciding whether to:
- add online delivery or table booking as a service,
- adjust pricing tier,
- expand into a new city or cuisine category.

There is no existing lightweight tool that lets a stakeholder input a restaurant's attributes and
receive an evidence-based rating estimate, nor a consolidated view of what actually drives rating
performance across the platform.

## Objective
1. Identify, with statistical support (not just visual impression), which attributes are
   significantly associated with `Aggregate rating`.
2. Build a regression model that estimates `Aggregate rating` from restaurant attributes, with a
   documented, defensible error margin.
3. Deliver two audience-specific outputs:
   - A descriptive/diagnostic **Power BI dashboard** for business stakeholders reviewing existing
     performance.
   - A **predictive Streamlit tool** for a narrower, clearly defined use case (see Scope below).

## Primary Use Case (chosen to resolve prior ambiguity)
This project supports **Use Case A: rating estimation for an existing or soon-to-launch
restaurant based on its planned service attributes (price tier, cuisine, delivery/booking
options, city)**, evaluated at the point a restaurant has just launched and has not yet
accumulated a meaningful review history.

Explicitly **out of scope**: using post-launch popularity signals (`Votes`, current live rating)
as predictive inputs for a *pre-launch* estimate — since a restaurant that hasn't opened cannot
have votes yet. Any model or app feature that requires `Votes` as user input is being used for
a *different* use case (explaining an existing restaurant's current rating) and must be labeled
as such, not conflated with pre-launch estimation.

## Success Criteria
- Model: Test-set MAE ≤ 0.40 rating points (on a 0–5 scale), R² ≥ 0.60, evaluated separately on
  India vs. non-India subsets to confirm the model isn't only reliable for the majority class.
- Dashboard: every KPI must be currency-normalized and sample-size-aware (no blended
  cross-currency averages, no per-country stat shown without its underlying n).
- App: every prediction is shown with its documented error margin, and any feature with known
  reliability limits (rare cities, non-top-20 cuisines) surfaces a visible caveat to the user.

## Stakeholders
| Stakeholder | Interest |
|---|---|
| Restaurant Owner / Franchisee | Wants a directional rating estimate before/at launch |
| Regional Ops Analyst | Wants descriptive dashboard of current performance by city/cuisine/price tier |
| Data/Analytics Lead (reviewer) | Wants a defensible, documented, reproducible analysis |

## Known Data Constraint (flagged up front)
The dataset is a single 2018–2019 snapshot, 90.59% India-weighted, with per-country currency
values that are not normalized to a common unit. These constraints directly shape scope — see
`06_assumptions_and_limitations.md` for the full list and how each is handled.
