# wc-passes-model

Predicts a **specific player's passes attempted in a specific match** and turns the
prediction into +EV plays on **Underdog Fantasy's soccer "Passes" pick'em** (Underdog
grades that prop on passes *attempted*, so the target matches the market exactly).

It's a **two-stage, minutes-aware, hierarchical Bayesian Negative-Binomial** model
trained on **national-team tournament data** (pooling across World Cups, Euros, Copa
América, AFCON), producing a full **posterior-predictive distribution** so over/unders
can be priced — not just a point estimate.

---

## Why this design (in one paragraph)

Passes attempted ≈ **minutes × pass-rate**, and the count is overdispersed, so we model
minutes as a distribution (stage 1) and a Negative-Binomial per-90 rate (stage 2), then
Monte-Carlo them together. With only ~8–9k national-team player-match rows and many
players seen just a handful of times, a **hierarchical Bayesian** model is the right tool:
partial pooling shrinks low-sample players toward their role/position mean and gives
**honest, wider uncertainty** for debutants — exactly what prevents mispriced props. A
gradient-boosted point model would overfit player identity and give no usable distribution.

See `../.claude/plans/structured-painting-forest.md` for the full rationale.

---

## Quickstart

```bash
pip3 install -r requirements.txt

python3 -m src.cli list-competitions   # verify StatsBomb comp/season ids, fill config.yaml
python3 -m src.cli ingest              # pull event data (incremental, cached)
python3 -m src.cli retrain             # ingest -> features -> fit -> save model
python3 -m src.cli backtest            # walk-forward eval (strict train/test separation)
python3 -m src.cli predict --upcoming upcoming.csv
```

## Retraining is one command

Everything is driven by **`config.yaml`** — it is the only file you edit to add data.

- **New tournament available?** Add a line under `statsbomb.competitions`.
- **New friendlies / qualifiers?** Add under `fbref.competitions`.
- Re-run `python3 -m src.cli retrain`.

Ingestion is **incremental**: any match already cached in `data/raw/` is never
re-pulled, so retraining after a new matchday only fetches the new matches and refits.
New players/teams simply appear as new factor levels in the hierarchical model — no
code changes, no schema migration.

---

## Train/test separation (enforced, not assumed)

Three structural guarantees (`src/splits.py`, `src/features.py`):

1. **Walk-forward by tournament+date** — each test fold (a whole tournament) is held
   out entirely; the fold's model trains only on matches that finished *before* the
   fold's first kickoff.
2. **Disjoint rows** — `assert_disjoint()` raises if any `(match_id, player_id)` row is
   in both train and test, and if any test match predates the newest training match.
3. **As-of-date features** — every feature for a match on date *D* uses only matches
   finished strictly before *D*, including the recency-weighted player and player×style
   rates. No test-period information can reach training.

Model selection uses **CRPS** and **log-loss** (which reward correct variance), never RMSE.

---

## Features

Built as-of-date in `src/features.py`:

| Feature | Notes |
|---|---|
| `recent_per90` | player's recency-weighted (exp. half-life) passes-per-90 |
| `style_per90_recencybiased` | player's attempts-per-90 vs the opponent **style cluster**, **recency-biased** (own half-life) |
| `role` | bucketed position (CB/FB/DM/CM/W/ST/GK) — hierarchical group |
| `position` | granular StatsBomb position — its own partially-pooled effect |
| `possession_share`, `opp_passes_allowed` | team context / "park factor" |
| opponent `style` | k-means cluster of teams (press/possession/volume), fit on train only |
| ball-depth (optional) | from StatsBomb 360 freeze-frames where available |

The hierarchical model (`src/rate_model.py`) adds player, role, **position**,
player×style, competition, and data-provider effects.

---

## Data sources (free/open)

| Source | Role | Caveat |
|---|---|---|
| **StatsBomb Open Data** (`statsbombpy`) | pass **label** + features, men's intl tournaments (~314 matches) | tournaments only; research/non-commercial license + attribution |
| **StatsBomb 360** | optional ball-depth feature | WC 2022, Euro 2020/2024, AFCON 2023 only |
| **FBref friendlies + WC qualifiers** (`soccerdata`) | recent **minutes/form** for 2026 squads | ⚠️ no per-player **passing** post-2026-01-20 (Opta feed gone) |
| **Underdog lines** | backtest target + live edge | no public API — log daily / scrape |

**Live-2026 gap:** no free *event* data during the tournament; live play uses recent-form
features + confirmed lineups (or a paid feed). **Friendlies/quals pass counts** aren't
free in 2026 — they contribute minutes/form unless you wire a paid feed
(`fbref.passing_available: true`).

---

## Underdog edge layer (`src/betting.py`)

Underdog posts one projection line; you pick MORE/LESS at fixed parlay multipliers, so
there's no two-sided price to de-vig. The bar is the **multiplier break-even**: an N-pick
parlay paying M× needs `(1/M)^(1/N)` per leg (5-pick 20× → ~0.549). We pick the legs whose
model probability most exceeds that bar.

**Edge benchmark:** Underdog's own posted line + realized hit-rate/ROI (no sharp closing
line exists for passes). The backtest warns if ROI rests on `< backtest.min_plays_warn` plays.

---

## Layout

```
config.yaml                 # the only file you edit to retrain
src/
  cli.py                    # entry point — `retrain`, `backtest`, `predict`, ...
  ingest_statsbomb.py       # incremental pass label + lineups/minutes
  ingest_fbref.py           # friendlies/quals minutes & form (rate-limited)
  ingest_underdog.py        # pick'em lines (append-only history)
  features.py               # as-of-date, recency-weighted features
  splits.py                 # walk-forward + train/test separation guarantees
  minutes_model.py          # stage-1 minutes distribution
  rate_model.py             # stage-2 hierarchical Bayesian NB (PyMC)
  predict.py                # Monte-Carlo posterior predictive
  betting.py                # Underdog break-even + edge + grading
  backtest.py               # walk-forward CRPS / log-loss / ROI
```

## Tests

```bash
python3 -m pytest tests/ -v
```

`tests/` is dependency-light (pandas/numpy/sklearn only — no pymc needed) and proves the
leakage guarantees on a synthetic 3-tournament corpus:
- `test_splits.py` — folds are chronological & disjoint; `assert_disjoint` catches a
  planted row-overlap leak and a future-dated training row.
- `test_features.py` — a player's first appearance has no history; poisoning a player's
  *last* match with an outlier leaves all earlier rows' features unchanged (as-of-date).

## Verification

- **Label sanity:** sum `Pass` events per player for a known match vs a reference box score.
- **No leakage:** `assert_disjoint` + as-of-date features (unit-test on a held-out match).
- **Model quality:** walk-forward CRPS/log-loss beat naive baselines; posterior-predictive
  intervals are calibrated (80% interval covers ~80% of actuals).
- **Pooling check:** debutants are shrunk toward role/position means with wider intervals.
- **Betting backtest:** hit-rate, ROI, and bet count on walk-forward, post-calibration picks.
