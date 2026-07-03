#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Nightly deterministic retrain — NO Claude, NO tokens. Run by launchd.
#   1. back up current production artifacts (rollback safety)
#   2. incremental Fotmob corpus refresh (pulls newly-finished matches)
#   3. retrain the hierarchical rate model
#   4. regenerate both calibrators
# Guardrail: if any step fails, the backup is left intact and production
# artifacts are only overwritten by steps that succeed (retrain/calibrate write
# in place, so a mid-run crash could leave a half-updated model — the backup dir
# is the recovery path). Logs to data/logs/retrain_<ts>.log.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
PY="/usr/local/bin/python3"
REPO="/Users/mihirepel/wc-passes-model"
cd "$REPO"
export PYTHONPATH="$REPO"
export PYTHONUNBUFFERED=1 LOKY_MAX_CPU_COUNT=8

LOG_DIR="$REPO/data/logs"; mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/retrain_$TS.log"
exec >>"$LOG" 2>&1

echo "=== daily_retrain $TS (CST $(date)) ==="

# 1. backup current production artifacts
BK="$REPO/models/_backup_$(date +%Y%m%d)"
mkdir -p "$BK"
[ -d models/hiernb ] && cp -R models/hiernb "$BK/hiernb" || true
cp models/calibrator.pkl "$BK/" 2>/dev/null || true
cp models/calibrator_striker.pkl "$BK/" 2>/dev/null || true
echo "[backup] -> $BK"

# 2. incremental corpus refresh (DATES auto-extends to today)
echo "[1/3] corpus refresh…"
$PY scripts/build_fotmob_corpus.py --refresh

# 3. retrain rate model
echo "[2/3] retrain…"
$PY -m src.cli retrain

# 4. recalibrate
echo "[3/3] calibrate…"
$PY scripts/calibrate.py
$PY scripts/calibrate_striker.py

# prune backups older than 10 days
find "$REPO/models" -maxdepth 1 -name '_backup_*' -type d -mtime +10 -exec rm -rf {} + 2>/dev/null || true

echo "=== retrain OK $(date +%H:%M:%S) ==="
