"""Live per-player **passes attempted** from Fotmob — fills the 2026 data gap.

No free source gives per-player passing for live 2026 internationals (FBref dropped
passing post-Opta; ESPN has no per-player passes). Fotmob does, via its private
`/api/data/matchDetails` endpoint, but every request needs a self-signed `x-mas` header.

The header = base64(JSON({body, signature})) where
    body      = {"url": <path>, "code": <unix ms>, "foo": "production:<build-hash>"}
    signature = md5( json(body) + SECRET )            # SECRET = "Three Lions" lyrics (!)
Both SECRET and the `foo` build-hash live in Fotmob's `_app-*.js` bundle and **rotate on
deploys**, so we re-extract them from the live bundle each run instead of hardcoding —
that's what makes this resilient (the community token-server wrappers broke when their
single hardcoded host died). Passes only populate AFTER kickoff, so this is for
grading / training-backfill, not a pre-match feature.

Per-player attempted passes JSON path:
    content.playerStats[<pid>].stats[*].stats["Accurate passes"].stat
      -> {"value": <completed>, "total": <ATTEMPTED>}
"""
from __future__ import annotations

import base64
import concurrent.futures as cf
import gzip
import hashlib
import json
import re
import ssl
import time
import urllib.request
from pathlib import Path

import pandas as pd

_SSL = ssl.create_default_context()
_SSL.check_hostname = False
_SSL.verify_mode = ssl.CERT_NONE
_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
_BASE = "https://www.fotmob.com"


def _raw(url: str, headers: dict | None = None, timeout: int = 25) -> bytes:
    h = {"User-Agent": _UA, "Accept-Encoding": "gzip"}
    if headers:
        h.update(headers)
    raw = urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout, context=_SSL).read()
    try:
        return gzip.decompress(raw)
    except Exception:
        return raw


class Fotmob:
    """Signed Fotmob client. Re-extracts the signing secret/foo from the live bundle."""

    def __init__(self):
        self._secret, self._foo = self._extract_signing()

    def _extract_signing(self) -> tuple[str, str]:
        """Find the _app-*.js chunk and pull out the SECRET (lyrics) + foo build-hash."""
        home = _raw(_BASE + "/").decode("utf-8", "ignore")
        chunks = dict.fromkeys(re.findall(r"/_next/static/[^\"']+\.js", home))
        for c in chunks:
            js = _raw(_BASE + c).decode("utf-8", "ignore")
            anchor = js.find('",S("".concat(JSON.stringify(n)).concat(t))')
            if anchor < 0:
                continue
            i = anchor - 1                                   # walk back to the opening quote
            while js[i] != '"' or js[i - 1] == "\\":
                i -= 1
            secret = json.loads(js[i:anchor + 1])
            foo = re.search(r'foo:"(production:[0-9a-f]+)"', js)
            if not foo:
                raise RuntimeError("found secret but not the foo build-hash")
            return secret, foo.group(1)
        raise RuntimeError("could not locate Fotmob signing chunk (bundle layout changed)")

    def _xmas(self, path: str) -> str:
        body = {"url": path, "code": int(time.time() * 1000), "foo": self._foo}
        jb = json.dumps(body, separators=(",", ":"))
        sig = hashlib.md5((jb + self._secret).encode()).hexdigest()
        return base64.b64encode(
            json.dumps({"body": body, "signature": sig}, separators=(",", ":")).encode()).decode()

    def get(self, path: str) -> dict:
        return json.loads(_raw(_BASE + path, {"x-mas": self._xmas(path)}))

    def get_cached(self, path: str, cache_dir: Path, key: str) -> dict:
        """get() with an on-disk JSON cache (finished matches never change)."""
        cache_dir.mkdir(parents=True, exist_ok=True)
        f = cache_dir / f"{key}.json"
        if f.exists():
            return json.loads(f.read_text())
        d = self.get(path)
        f.write_text(json.dumps(d))
        time.sleep(0.12)                                    # be polite on live fetches
        return d

    # ---- high-level helpers -------------------------------------------------

    def matches_on(self, yyyymmdd: str) -> list[dict]:
        """All matches on a date: [{id, league, home, away, finished, started}]."""
        d = self.get(f"/api/data/matches?date={yyyymmdd}")
        out = []
        for lg in d.get("leagues", []):
            for m in lg.get("matches", []):
                st = m.get("status", {})
                out.append({"id": m.get("id"), "league": lg.get("name"),
                            "home": m.get("home", {}).get("name"), "away": m.get("away", {}).get("name"),
                            "finished": bool(st.get("finished")), "started": bool(st.get("started"))})
        return out

    def match_player_passes(self, match_id: int) -> pd.DataFrame:
        """Per-player attempted/completed passes + minutes for a (started) match.
        Columns: match_id, player_id, player, team, minutes, passes_attempted, passes_completed."""
        md = self.get(f"/api/data/matchDetails?matchId={match_id}")
        content = md.get("content", {})
        # team-id -> name, from the lineup block
        teams = {}
        lu = content.get("lineup", {}) or {}
        for side in ("homeTeam", "awayTeam"):
            t = lu.get(side) or {}
            if t.get("id") is not None:
                teams[t["id"]] = t.get("name")
        # fallback team map from general header
        for side in ("homeTeam", "awayTeam"):
            t = (md.get("general", {}) or {}).get(side) or {}
            if t.get("id") is not None:
                teams.setdefault(t["id"], t.get("name"))

        rows = []
        for pid, pl in (content.get("playerStats", {}) or {}).items():
            attempted = completed = minutes = None
            for grp in pl.get("stats", []) or []:
                s = grp.get("stats", {})
                if "Accurate passes" in s:
                    stat = s["Accurate passes"].get("stat", {})
                    completed, attempted = stat.get("value"), stat.get("total")
                if "Minutes played" in s:
                    minutes = s["Minutes played"].get("stat", {}).get("value")
            if attempted is None:
                continue                                    # didn't play / no pass data
            rows.append({"match_id": match_id, "player_id": pl.get("id", pid), "player": pl.get("name"),
                         "team": teams.get(pl.get("teamId")), "minutes": minutes,
                         "passes_attempted": attempted, "passes_completed": completed})
        return pd.DataFrame(rows)


def _pos_from_depth(x: float | None) -> str:
    """Coarse StatsBomb-style position from pitch depth (own goal=0 .. opp goal=1).
    Used only for players we can't match to a StatsBomb record (their modal position
    is better when available)."""
    if x is None:
        return "Center Midfield"
    return ("Goalkeeper" if x <= 0.15 else "Center Back" if x <= 0.38
            else "Center Midfield" if x <= 0.62 else "Center Forward")


def _match_meta(md: dict) -> dict:
    """Per-match score + ball possession from a Fotmob matchDetails payload.
    Scores: header.teams[*].score (home first, away second).
    Possession: content.stats.Periods.All, key 'BallPossesion' -> [home%, away%]."""
    h = md.get("header", {}) or {}
    teams = h.get("teams") or []
    home = teams[0] if len(teams) > 0 else {}
    away = teams[1] if len(teams) > 1 else {}
    poss_home = poss_away = None
    allp = (((md.get("content") or {}).get("stats") or {}).get("Periods") or {}).get("All") or {}
    for grp in (allp.get("stats") or []):
        for s in (grp.get("stats") or []):
            if s.get("key") == "BallPossesion" or s.get("title") == "Ball possession":
                vals = s.get("stats") or []
                if len(vals) == 2:
                    poss_home, poss_away = vals[0], vals[1]
        if poss_home is not None:
            break
    return {"home_score": home.get("score"), "away_score": away.get("score"),
            "poss_home": poss_home, "poss_away": poss_away}


# International national-team leagues we keep (everything else — domestic clubs, club cups,
# continental CLUB competitions — is dropped). Note: WC/Nations-League qualification match
# via "World Cup"/"Nations League"; we deliberately DON'T key on bare "Qualif" (that would
# also grab club Champions/Europa League qualifiers).
_INTL_KEYS = ("World Cup", "EURO", "European Championship", "Copa America",
              "Africa Cup of Nations", "AFCON", "Asian Cup", "Gold Cup", "Nations League", "Friendl")

# This model is SENIOR MEN's passes. Youth (U15-U23 / Olympics), women's, futsal and beach
# tournaments share the 'World Cup'/'EURO' names but are a different population — exclude them.
_EXCLUDE_TOKENS = ("U15", "U16", "U17", "U18", "U19", "U20", "U21", "U22", "U23",
                   "Under-", "Under ", "Women", "Womens", "Female", "Girls",
                   "Olympic", "Futsal", "Beach")


def _excluded_league(league: str) -> bool:
    return any(t in str(league) for t in _EXCLUDE_TOKENS)


def _wanted_league(league: str) -> bool:
    """True for SENIOR MEN's international leagues (cheap pre-filter on the day listing,
    so we never fetch matchDetails for domestic club, youth, or women's matches)."""
    L = str(league)
    return "Club" not in L and not _excluded_league(L) and any(k in L for k in _INTL_KEYS)


def _classify_comp(league: str, date) -> tuple:
    """Map a Fotmob leagueName -> (competition_label, is_friendly, is_qualifier).
    Returns (None, 0, 0) for club/irrelevant leagues. Tournament labels are
    edition-distinct (year-aware) so the recent-form window treats each separately and
    they line up with the StatsBomb competition names ('World Cup 2022', 'Euro 2024', …).
    Order matters: WC qualification contains 'World Cup'; Nations League contains
    'Qualification' for its league phase — neither must be mistaken for the other."""
    L = league
    yr = date.year if date is not None else None
    mo = date.month if date is not None else 1
    if "Club" in L or _excluded_league(L):         # club / youth / women's / futsal etc.
        return None, 0, 0
    if "World Cup" in L and "Qualif" in L:
        return "WC Qualifier", 0, 1
    if "Nations League" in L:                       # competitive (incl. its 'Qualification' phase)
        return "Nations League", 0, 0
    if "World Cup" in L:                            # finals (qualifier handled above)
        return (f"World Cup {yr}" if yr else "World Cup"), 0, 0
    if "EURO" in L or "European Championship" in L:
        return ("Euro 2020" if yr == 2021 else f"Euro {yr}" if yr else "Euro"), 0, 0
    if "Copa America" in L:
        return (f"Copa America {yr}" if yr else "Copa America"), 0, 0
    if "Gold Cup" in L:
        return (f"Gold Cup {yr}" if yr else "Gold Cup"), 0, 0
    if "Africa Cup of Nations" in L or "AFCON" in L:
        return ("AFCON 2023" if (yr == 2024 and mo <= 3) else f"AFCON {yr}" if yr else "AFCON"), 0, 0
    if "Asian Cup" in L:
        return ("Asian Cup 2023" if (yr == 2024 and mo <= 3) else f"Asian Cup {yr}" if yr else "Asian Cup"), 0, 0
    if "Friendl" in L:
        return "Intl Friendly", 1, 0
    return None, 0, 0


class FotmobTraining(Fotmob):
    def __init__(self, cache_dir: Path | str | None = None):
        super().__init__()
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def match_training_rows(self, mid: int) -> list[dict]:
        """Corpus-schema rows for one match: per-player passes ATTEMPTED (label) +
        minutes + position + opponent + stage + pitch depth + realized possession + score.
        International only (WC finals/quals, Euro, Copa, AFCON, Asian Cup, Nations League,
        friendlies); club matches return []. The pass label is ALWAYS attempts
        ('Accurate passes'.stat.total) — never the accurate/.value count."""
        path = f"/api/data/matchDetails?matchId={mid}"
        md = self.get_cached(path, self.cache_dir, str(mid)) if self.cache_dir else self.get(path)
        g = md.get("general", {}) or {}
        league = str(g.get("leagueName", ""))
        home, away = g.get("homeTeam", {}), g.get("awayTeam", {})
        date = pd.to_datetime(g.get("matchTimeUTCDate")).tz_localize(None) if g.get("matchTimeUTCDate") else None
        comp, is_friendly, is_qualifier = _classify_comp(league, date)
        if comp is None:
            return []                                  # club / youth / women's / non-intl
        if str(g.get("gender", "male")).lower() == "female":
            return []                                  # senior MEN's model only
        # realized possession (fraction) + score per team, for this match
        mm = _match_meta(md)
        tstat = {
            home.get("name"): {"poss_for": mm["poss_home"] / 100.0 if mm["poss_home"] is not None else None,
                               "goals_for": mm["home_score"], "goals_against": mm["away_score"]},
            away.get("name"): {"poss_for": mm["poss_away"] / 100.0 if mm["poss_away"] is not None else None,
                               "goals_for": mm["away_score"], "goals_against": mm["home_score"]},
        }
        rnd = str(g.get("leagueRoundName", "") or g.get("matchRound", ""))
        knockout = any(k in rnd for k in ("Final", "16", "32", "Quarter", "Semi", "Knockout")) or not rnd.strip().isdigit() and "Group" not in rnd and "Stage" not in rnd
        stage = "knockout" if knockout else "group"
        # per-player position + depth + team/opponent, from the lineup block
        info = {}
        lu = md.get("content", {}).get("lineup", {}) or {}
        for side, opp in (("homeTeam", away.get("name")), ("awayTeam", home.get("name"))):
            t = lu.get(side, {}) or {}
            for grp in ("starters", "subs"):
                for p in (t.get(grp, []) or []):
                    p = p[0] if isinstance(p, list) and p else p
                    if not isinstance(p, dict):
                        continue
                    x = (p.get("horizontalLayout") or {}).get("x")
                    info[p.get("id")] = {"team": t.get("name"), "opponent": opp,
                                         "pos": _pos_from_depth(x), "depth": x}
        rows = []
        for pid, pl in (md.get("content", {}).get("playerStats", {}) or {}).items():
            attempted = minutes = None
            for grpd in pl.get("stats", []) or []:
                s = grpd.get("stats", {})
                if "Accurate passes" in s:
                    attempted = s["Accurate passes"].get("stat", {}).get("total")
                if "Minutes played" in s:
                    minutes = s["Minutes played"].get("stat", {}).get("value")
            if attempted is None:
                continue
            pid_i = pl.get("id", pid)
            meta = info.get(pid_i, {})
            st = tstat.get(meta.get("team"), {})
            rows.append({"fm_player_id": pid_i, "player": pl.get("name"),
                         "team": meta.get("team"), "opponent": meta.get("opponent"),
                         "position": meta.get("pos", "Center Midfield"), "depth": meta.get("depth"),
                         "minutes": minutes, "passes_attempted": attempted, "stage": stage,
                         "is_friendly": is_friendly, "is_qualifier": is_qualifier,
                         "match_date": date, "match_id": mid, "competition": comp,
                         "possession_for": st.get("poss_for"),
                         "goals_for": st.get("goals_for"), "goals_against": st.get("goals_against"),
                         "is_home": int(meta.get("team") == home.get("name"))})
        return rows

    def wc_training_rows(self, dates: list[str], workers: int = 4) -> pd.DataFrame:
        """All finished INTERNATIONAL matches (WC finals+quals, Euro, Copa, AFCON,
        Asian Cup, Nations League, friendlies; not club) across the given YYYYMMDD dates.
        I/O-bound, so day listings AND per-match pulls run in a thread pool; per-match
        JSON is cached on disk, so re-runs only fetch new matches. The league filter is
        delegated to `match_training_rows`/`_classify_comp` (returns [] for anything we
        don't want), so the gate stays in one place."""
        # 1) gather candidate (finished, non-club) match ids across all dates, in parallel
        def _day(d):
            try:
                return [(m["id"], d, m.get("home"), m.get("away")) for m in self.matches_on(d)
                        if m["finished"] and _wanted_league(m["league"])]
            except Exception as e:
                print(f"[fm-train] matches_on {d} failed: {e}", flush=True)
                return []
        cand: dict = {}
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            for part in ex.map(_day, dates):
                for mid, d, h, a in part:
                    cand.setdefault(mid, (mid, d, h, a))   # dedup match ids across dates
        print(f"[fm-train] {len(cand)} candidate finished non-club matches across {len(dates)} dates", flush=True)
        # 2) pull per-match training rows in parallel (cached; classifier drops non-intl)
        frames = []
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(self.match_training_rows, mid): meta for mid, meta in cand.items()}
            for fut in cf.as_completed(futs):
                mid, d, h, a = futs[fut]
                try:
                    r = fut.result()
                except Exception as e:
                    print(f"[fm-train] match {mid} ({h}-{a}) failed: {e}", flush=True)
                    continue
                if r:
                    frames.append(pd.DataFrame(r))
                    print(f"[fm-train] {d} {h}-{a} ({r[0]['competition']}): {len(r)} rows", flush=True)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def grade_date(yyyymmdd: str, out_dir: str | Path | None = None) -> pd.DataFrame:
    """Pull per-player passes for every finished match on a date (for grading/backfill)."""
    fm = Fotmob()
    frames = []
    for m in fm.matches_on(yyyymmdd):
        if not m["finished"]:
            continue
        df = fm.match_player_passes(m["id"])
        df["match"] = f"{m['home']} vs {m['away']}"
        df["league"] = m["league"]
        frames.append(df)
        print(f"[fotmob] {m['home']} vs {m['away']} ({m['league']}): {len(df)} players", flush=True)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if out_dir is not None and len(out):
        p = Path(out_dir) / f"fotmob_passes_{yyyymmdd}.parquet"
        out.to_parquet(p, index=False)
        print(f"[fotmob] wrote {len(out)} player rows -> {p}", flush=True)
    return out


if __name__ == "__main__":
    import sys
    date = sys.argv[1] if len(sys.argv) > 1 else time.strftime("%Y%m%d")
    df = grade_date(date)
    if len(df):
        print(df[["player", "team", "minutes", "passes_attempted", "passes_completed"]]
              .sort_values("passes_attempted", ascending=False).head(20).to_string(index=False))
