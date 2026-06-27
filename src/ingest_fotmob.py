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


class FotmobTraining(Fotmob):
    def __init__(self, cache_dir: Path | str | None = None):
        super().__init__()
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def match_training_rows(self, mid: int) -> list[dict]:
        """Corpus-schema rows for one match: per-player passes (label) + minutes +
        position + opponent + stage + pitch depth. International (WC/friendly/qual) only."""
        path = f"/api/data/matchDetails?matchId={mid}"
        md = self.get_cached(path, self.cache_dir, str(mid)) if self.cache_dir else self.get(path)
        g = md.get("general", {}) or {}
        league = str(g.get("leagueName", ""))
        if "Club" in league or not any(k in league for k in ("World Cup", "Friendl", "Qualif")):
            return []
        is_friendly = int("Friendl" in league)
        is_qualifier = int("Qualif" in league)        # WC Qualification leagues contain
        comp = ("WC Qualifier" if is_qualifier         # "World Cup" too -> check qualifier
                else "World Cup 2026" if "World Cup" in league   # FIRST so finals != quals
                else "Intl Friendly")
        home, away = g.get("homeTeam", {}), g.get("awayTeam", {})
        date = pd.to_datetime(g.get("matchTimeUTCDate")).tz_localize(None) if g.get("matchTimeUTCDate") else None
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
            rows.append({"fm_player_id": pid_i, "player": pl.get("name"),
                         "team": meta.get("team"), "opponent": meta.get("opponent"),
                         "position": meta.get("pos", "Center Midfield"), "depth": meta.get("depth"),
                         "minutes": minutes, "passes_attempted": attempted, "stage": stage,
                         "is_friendly": is_friendly, "is_qualifier": is_qualifier,
                         "match_date": date, "match_id": mid, "competition": comp})
        return rows

    def wc_training_rows(self, dates: list[str]) -> pd.DataFrame:
        """All finished INTERNATIONAL matches (WC + friendlies + qualifiers, not club)
        across a list of YYYYMMDD dates. Minnow friendlies w/o per-player stats yield 0."""
        frames = []
        for d in dates:
            for m in self.matches_on(d):
                lg = str(m["league"])
                if not m["finished"] or "Club" in lg or not any(
                        k in lg for k in ("World Cup", "Friendl", "Qualif")):
                    continue
                r = self.match_training_rows(m["id"])
                if r:
                    frames.append(pd.DataFrame(r))
                    print(f"[fm-train] {d} {m['home']}-{m['away']} ({lg}): {len(r)} rows", flush=True)
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
