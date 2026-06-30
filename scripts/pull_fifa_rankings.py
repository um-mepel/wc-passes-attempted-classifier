"""Scrape the FIFA ranking as-of the start of every major-tournament edition (2002->now)
from fifaranking.net, for the pre-tournament Elo pull-toward-FIFA. Caches one CSV per
edition date at data/raw/fifa_rankings/<YYYY-MM-DD>.csv with columns: rank, team, points.

Points scales differ by era (2002 old system ~max 800; 2006-2018 ~max 1900; 2018+ Elo-like
~1700) — that's fine, the Elo builder normalises each date internally.
"""
from __future__ import annotations

import gzip
import html as H
import re
import ssl
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import Config

_CTX = ssl.create_default_context(); _CTX.check_hostname = False; _CTX.verify_mode = ssl.CERT_NONE


def fetch_table(d: str) -> pd.DataFrame:
    """Return the FIFA ranking as-of date d (YYYY-MM-DD): rank, team, points."""
    url = f"https://en.fifaranking.net/ranking/index.php?d={d}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept-Encoding": "gzip"})
    raw = urllib.request.urlopen(req, timeout=30, context=_CTX).read()
    try:
        raw = gzip.decompress(raw)
    except Exception:
        pass
    html = raw.decode("utf-8", "ignore")
    rows = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S):
        cells = [H.unescape(re.sub(r"<[^>]+>", "", c)).strip()
                 for c in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
        cells = [c for c in cells if c != ""]
        if len(cells) < 3 or not cells[0].isdigit():
            continue
        rank = int(cells[0])
        # layout is [rank, points, team, ...]; team is the first non-numeric cell.
        team = next((c for c in cells[1:] if not c.replace(".", "").replace(",", "").isdigit()), None)
        pts = next((float(c.replace(",", "")) for c in cells[1:]
                    if c.replace(".", "").replace(",", "").isdigit()), None)
        if team and pts is not None:
            rows.append((rank, team, pts))
    return pd.DataFrame(rows, columns=["rank", "team", "points"])


def main():
    cfg = Config.load()
    major = pd.read_parquet(cfg.path("raw") / "major_results.parquet")
    major["date"] = pd.to_datetime(major["date"])
    # one snapshot per edition, taken at the edition's first match date
    starts = (major.assign(year=major.date.dt.year)
              .groupby(["competition", "year"]).date.min().dt.date.astype(str))
    dates = sorted(set(starts))
    out = cfg.path("raw") / "fifa_rankings"; out.mkdir(parents=True, exist_ok=True)
    print(f"{len(dates)} edition dates to snapshot", flush=True)
    for d in dates:
        f = out / f"{d}.csv"
        if f.exists():
            continue
        try:
            df = fetch_table(d)
            if len(df) < 50:
                print(f"  {d}: only {len(df)} rows — skipped", flush=True); continue
            df.to_csv(f, index=False)
            print(f"  {d}: {len(df)} teams -> {f.name}", flush=True)
            time.sleep(0.2)
        except Exception as e:
            print(f"  {d}: ERROR {e}", flush=True)


if __name__ == "__main__":
    main()
