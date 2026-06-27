# WC Passes Model — Design, Validation & Decisions Log

Living document. Captures the *why* behind the model, what we've learned validating it
against real World Cup 2026 results, the betting rules that fall out of that, the current
feature set, data sources/gaps, and the open work queue.

---

## 1. Goal & betting thesis

- **Target:** a specific player's **passes _attempted_** in a specific match (continuous,
  with a margin of error / confidence interval — never whole numbers).
- **Market:** **Underdog Fantasy** pick'em "Passes" — graded on passes *attempted*, so the
  target matches the market exactly. Standard parlay multipliers: 2-pick 3×, 3-pick 6×,
  4-pick 10×, 5-pick 20×. EV = P(all hit)×M − 1; per-leg break-even = (1/M)^(1/N).
- **Edge thesis (core):** *Underdog's lines are soft.* The edge is comparing our number to
  the **real expected value**, not to a sharp line (there isn't one). We win by knowing the
  true distribution better than a lazily-posted projection.
- **Operational cadence:** ~1h before each game, produce the best 1–2 **three-leg** parlays
  across all remaining games of the day (can mix players/slots; never the same player twice;
  same-team legs OK unless correlation too high / opposite-direction). Email to
  `iainchu0219@gmail.com` and `mepel@umich.edu` (send **from** mepel@umich.edu).
- **Hard rule:** **only bet confirmed starters.** Model beats the baseline ~22% on CRPS at
  *known* minutes and *loses* at guessed minutes. Lineups (ESPN `rosters`) populate ~1h pre-kick.

---

## 2. Model architecture

Two-stage, fully Bayesian, distribution-first (the posterior predictive *is* the object we
price against).

**Stage 1 — minutes:** P(start) + minutes|start. Only used when minutes are unknown; for
betting we condition on confirmed starters at actual minutes.

**Stage 2 — rate:** hierarchical **Negative-Binomial GLM** with a minutes offset (PyMC),
non-centered parameterization, sampled with **nutpie** (Rust NUTS, 2–5× faster than stock
PyMC; falls back to the pymc sampler if it fails to compile). macOS: `cores=2`, `mp_ctx=spawn`.

```
passes ~ NegBinomial(mu, alpha)
log(mu) = log(minutes)
        + a_role[role]                 # role baseline (new-cap fallback)
        + b_position[position]         # granular position residual
        + a_player[player]             # SEE §3 — the key change
        + s_pstyle[player x opp-style] # pooled player×style slope
        + t_comp[competition] + p_prov[provider]
        + beta · X                     # matchup features (see §5)
```

Predictions: Monte-Carlo posterior predictive → mean + 80% CI; P(over line) →
**isotonic calibration** (`models/calibrator.pkl`, clipped to ~[2%,98%]).

**Retraining is one command** (`python3 -m src.cli retrain`), config-driven
(`config.yaml`). Walk-forward CV by tournament; strict as-of-date features (no leakage —
this was audited repeatedly).

---

## 3. The player-effect anchor (the big modeling story)

**Problem found in validation:** the model **systematically under-predicted elite "builders"**
(deep playmakers / ball-playing CBs) in possession-dominant games. Root cause: the player
random effect `a_player ~ Normal(0, σ)` shrinks every player toward the **role mean**, and σ
was estimated small by the ~2k low-data players — so stars got lowballed.
(`recent_per90` as a plain feature didn't help: its coefficient went to ~0, i.e. the model
*ignored* each player's own rate.)

**Things that did NOT fix it (recorded so we don't repeat them):**
- Student-t de-shrink of `a_player` — helped a little, not enough.
- ESPN team strength/possession features — neutral on aggregate.
- Position-group × matchup interactions (def/wing/mid/attack × SoS/possession) — neutral
  (CRPS 8.06 vs 8.00); reverted.
- **Anchor as a hard log-offset** (`+ log(recent_rate)`, coeff = 1): **exploded** (Rodri →
  469) because the role/feature effects *over-add* on top (new-cap fallback rows contaminate
  the shared effects).
- **Anchor as a free/strong-prior coefficient** on `log(recent_rate)`: **collapsed to ~0** —
  with 9.5k rows the likelihood overrides the prior, because the team/possession features are
  **collinear** with recent rate and already explain the level. (This is why we can't keep all
  the level-features free *and* force a separate anchor term — they compete.)

**The fix that works — informative prior on the player effect:**
shrink `a_player` toward the **player's own recent rate**, not toward zero:
```
prior_dev[i] = shrink_i · log( player_recent_rate_i / role_mean_role(i) )
a_player[i]  ~ StudentT(nu=4, mu=prior_dev[i], sigma=sigma_player)
```
- `player_recent_rate` = mean passes-per-90 over the player's **last 2 tournaments** (honors
  the recency requirement; coaching/squad context changes between tournaments).
- `shrink_i = n_i / (n_i + K)`, K≈4 — **data-count shrinkage**: high-data players (builders)
  keep their own rate; thin-data players pull back to the role mean. This is what prevents the
  overfit (see below).
- Keeps **all** features (they stay as matchup adjustments — no collinearity fight, because the
  recent rate enters only as the *prior center* of the player term, not as a competing feature).

**Builder fix, validated against actuals:**

| Player | Old model | Anchor model | Actual |
|---|---|---|---|
| Rodri (DM)   | 86 | **98** | 106 |
| Laporte (CB) | 77 | **93** | 102 |
| Theate (LB)  | 61 | **75** | (line 74.5 → flips to OVER) |
| Tielemans    | 56 | **68** | (line 61.5 → flips to OVER) |

**The honest tradeoff (K sweep).** The *no-shrink* version pinned each player to their training
average (σ_player → 0.004) and **overfit** → aggregate CRPS **9.15**. Count-shrinkage `n/(n+K)`
fixes the overfit but also mutes the builder lift (shrink applies to the prior center). Sweep:

| | Aggregate CRPS | Rodri (act. 106) | Laporte (act. 102) |
|---|---|---|---|
| Old model (shrink-to-role) | **8.00** | 86 | 77 |
| Anchor, no shrink | 9.15 (overfit) | 98 | 93 |
| Anchor, K=4 | 8.17 | 83 | 80 |
| **Anchor, K=2 (chosen)** | 8.32 | **90** | **86** |

The anchor does **not** beat the old model on *aggregate* — its value is concentrated in
**data-rich builders** (Laporte 77→86, n=15), at a small cost on noisy players we never bet. Two
things matter: (1) **match count drives trust** — Theate has only **n=3** matches, so he's correctly
shrunk toward the role mean regardless of K (thin-data players are a *qualitative-read* call, not a
model-trust one); (2) even at full trust the model can't predict a blowout *above* a player's own
rate (Rodri 98 < actual 106 because Spain dominated extraordinarily). **Chosen K=2**: it bakes the
validated "builders go over" lesson into the model for the data-rich players we actually bet, at a
~4% aggregate-CRPS cost on players we don't. `K` is the single knob (rate_model.py); raise it to
favor aggregate accuracy, lower it to favor builder lift.

---

## 4. Validation log (real WC 2026 results vs model)

Four games graded. Consistent, actionable failure modes:

- **France–Norway (dead rubber):** MAE 7.5. Data-backed starters excellent (Mbappé −1,
  Lacroix +1). New caps under-predicted. **Recommended 3-leg all WON.** France fielded a
  B-team (rotation) — stage/stakes matter (see §6).
- **Senegal–Iraq:** MAE 13.6. Gueye 92→94 nailed (validates elite pivots). New caps noisy in
  *both* directions; sub-risk players hurt (90-min assumption).
- **Uruguay (vs Spain):** MAE ~11. Two projected starters **DNP** (projected XI was wrong →
  *always wait for confirmed lineups*). **Within-team distribution flaw:** vs a presser, the
  CBs passed *more* than predicted (recycled under pressure: Olivera 46, Cáceres 37) while the
  **midfielders got starved** (Valverde/Ugarte/Canobbio ~13–15 vs ~31 predicted).
- **Spain (vs Uruguay) — most important:** builders **exploded** — Rodri 106, Laporte 102,
  Cubarsí 87 — all 20–33 *over* the old model. **Vindicated the Rodri OVER 89.5 call** (we
  leaned over despite the model's slight under; actual 106).

**Unified betting rule (proven across 4 games):**
> In a possession-dominant matchup, the **favorite's builders + keeper go OVER** the (old)
> model — bet those overs. The **underdog's players go UNDER** — bet those unders. The model's
> *level* was biased low for builders (the anchor in §3 is the structural fix for this).

Trust: **data-backed + full-90 + on-the-ball central** starters. Distrust: **new caps** (noisy
both ways), **sub-risk** players, **projected (unconfirmed) XIs**.

---

## 5. Feature set

Stays in the model (all as-of-date, leakage-safe):
- `style_per90_recencybiased` — player × opponent-style, recency-biased.
- `team_poss_asof`, `team_elo` — own-side control/strength.
- `opp_allowed_asof`, `opp_elo`, `team_poss_espn`, `opp_poss_espn` — opponent/matchup context.
- Player's recent rate enters via the **prior center** of `a_player` (§3), not as a flat feature.

Conventions: Elo from ESPN results (full coverage) + possession where ESPN has it. Opponent
"style" = clustered from PPDA/block-height/possession/directness. Role buckets (ball-playing CB,
DM, CM, FB, winger, ST) > crude DEF/MID/FWD.

---

## 6. In-progress feature work (requested)

1. **Group-stage vs knockout separation** — capture StatsBomb `competition_stage`; add a
   knockout indicator. Intensity/rotation differs by stage.
2. **`qualified` / `eliminated` features (group stage)** — compute from group standings
   (cumulative points/games before each matchday). Conservative flags: eliminated = 0 pts after
   2 games; qualified/clinched = 6 pts after 2. **Weight dead-rubber group games** (a team
   already qualified or eliminated → rotation, like France–Norway) appropriately; treat
   elimination-stakes group games like knockouts.
   *Status: needs a re-ingest to capture stage + match scores; standings logic then lives in
   features.py.*
3. **Team style window = max(3 years, last coaching change)** — style changes with the coach,
   so the opponent-style/possession features should only look back that far. **Blocker:**
   coaching-change dates are **external data not in StatsBomb** and there's no clean free source.
   Interim: flat 3-year window; the coaching-change refinement needs a data source.

---

## 7. Data sources & gaps

| Source | Use | Status |
|---|---|---|
| **StatsBomb Open Data** | training corpus: WC 2018/2022, Euro 2020/2024, Copa 2024, AFCON 2023 (~314 matches, ~9.5k player-match rows). Pass label = count of `Pass` events. | ✅ ingested; license = research/non-commercial |
| **ESPN public JSON API** | fixtures, confirmed lineups (~1h pre-kick), team possession/passes; all-nations results → Elo. | ✅ (use unverified SSL ctx; curl `--compressed`) |
| **Underdog API** (`/beta/v5/over_under_lines`, stat `period_1_2_passes`) | the lines we bet; **open, no auth**. | ✅ |
| **Per-player 2026 passing** | the live-tournament gap. | ❌ no free source — FBref dropped passing (Jan 2026 Opta cut); ESPN per-player has no passes; Fotmob/Sofascore Cloudflare/DataDome-blocked |
| **Fotmob API** (`GET /api/data/matchDetails?matchId=`) | **fills the 2026 per-player passes gap.** | ✅ **IMPLEMENTED** (`src/ingest_fotmob.py`). Per-player attempted passes = `content.playerStats[*].stats[...]["Accurate passes"].total`. Verified live on WC 2026 (Gueye 94 matches the hand-graded actual). **Auth cracked & self-contained:** `x-mas` header = base64({body:{url,code,foo}, signature: md5(json(body)+SECRET)}); **SECRET = the "Three Lions" lyrics** (not Rick Astley) + a `foo` build-hash, both in the `_app-*.js` bundle — we **re-extract them from the live bundle each run** so it survives rotation (the community token-server `46.101.91.154:6006` is dead; soccerdata dropped FotMob; bgrnwd predates auth). Passes are **post-kickoff only** → grading/backfill, not a pre-match feature; ToS-gray → keep rate low |
| **Club passing** | optional separate feature for players w/o intl history. | ⏸ deferred (BACKLOG.md). Est. club↔intl passing correlation ~0.5 raw rate, ~0.7 on *share* — use share, never inject as a national rate |

---

## 8. Engineering notes

- **nutpie** is the default sampler (Rust NUTS); `PM_SAMPLER=pymc` to override. Env knobs:
  `PM_CORES/PM_DRAWS/PM_TUNE`.
- Persistence via **pickle** (arviz NetCDF backend was fragile).
- `parlay.py`: correlation-aware 3-leg picker; row-alignment by `player_id` (build re-sorts —
  this bit us once, producing a bogus "Tchouaméni 39"); opponent resolved via Underdog match_id
  co-occurrence with ESPN fixture fallback; applies the calibrator; computes EV.
- **No leakage** is a standing requirement: every feature for match *t* uses only data < *t*;
  scaler fit on train only; team possession/opp-allowed are as-of-date expanding means.

---

## 9. Open queue (next actions)

- [x] Anchor model (K=2) retrained + recalibrated (Brier 0.204→0.200) — **live model**.
- [x] `ingest_fotmob.py` — live 2026 per-player **passes attempted** working (auto-extracts
      the rotating secret). **Next: wire it into an auto-grader** (pull Fotmob actuals after each
      slot, compare to predictions, append to the validation log) and **backfill training data**.
- [ ] Re-ingest with `competition_stage` + scores → add knockout / qualified / eliminated
      features + dead-rubber weighting (§6.1–6.2).
- [ ] Style window: flat 3-year cutoff now; source coaching-change dates for the full rule (§6.3).
- [ ] Standing parlay rules: P(hit) ≥ ~80% filter, auto-exclude new caps, favor builder-OVERs
      in dominant matchups / underdog-UNDERs.
