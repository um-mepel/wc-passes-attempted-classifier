# Backlog / deferred enhancements

## Deferred

### Club passing as a SEPARATE feature (not injected into the national-team rate)
**Status:** deferred (design agreed).

Add each player's club passing volume (e.g. club passes-per-90) as its **own input
feature**, distinct from the national-team `recent_per90`. The model then *learns*
the club→country relationship as a coefficient, instead of us assuming it or
contaminating the national-team rate.

**Why a separate feature, not injection:** club role ≠ national role (system,
teammates, possession share, build-up duties all differ). Injecting club passing
directly as a national-team rate would mislead — especially for players who are
ball-dominant at club but role-players for country (or vice-versa). Keeping it
separate lets the model down-weight it where it doesn't transfer.

**Implementation notes for later:**
- Source is the blocker: per-player passing is Opta-gated. FBref dropped passing
  (club + national) after the Jan-2026 Opta cut; Fotmob/Sofascore APIs are
  token/Cloudflare-blocked. Needs a paid feed or manual entry.
- Add as `club_per90` in `_FEATURES`; standardize with the train-only scaler.
- Most useful for NEW CAPS with no national-team history (Pedri, Cubarsí, Porro),
  who currently fall back to position/role pooling (wide intervals).
- Keep heavy regularization / let the hierarchical shrinkage handle low-coverage.

## Discussed / next
- Normalized skill metric in the backtest: CRPS vs a naive baseline (predict the
  player's historical mean) so fold-to-fold comparison is fair (raw CRPS scales
  with each tournament's pass-volume spread).
- ESPN team-possession wired into 2026 features (free 2026 upgrade).
- Underdog line logging -> betting backtest (hit-rate / ROI).
