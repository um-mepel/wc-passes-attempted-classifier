"""Clean player_id quality in the assembled corpus, in two passes:

1. DE-CONFLATION — split a player_id that holds >1 distinct PERSON (cross-source id-merge
   collisions, e.g. several "Rodríguez" pooled into one id). Pooling unrelated players'
   passes poisons both history and the model's per-player effect and manufactures fake
   prop edges. The largest person-cluster keeps the id; each other gets a fresh id.
   Spelling variants of ONE person (Dayot/Dayotchanculle Upamecano, n'golo/n''golo kante)
   stay merged via prefix / one-indel / a small transliteration whitelist.

2. DE-FRAGMENTATION — merge >1 player_id that are the SAME player split across sources by
   differing names (StatsBomb full legal name vs Fotmob common name). Conservative: merge
   only when one name's tokens are an exact SUBSET of the other's (>=2 tokens), within one
   team — never on a lone first name or prefix, so brothers/namesakes split in pass 1 stay
   apart.

Call clean_player_ids(df) after corpus assembly and before writing. Idempotent.
"""
from __future__ import annotations

import unicodedata

import pandas as pd

MISSING = {"", "none", "nan", "null"}
_NEWPID0 = 950_000_000
# transliteration variants a token heuristic can't see as one person -> force-merged.
KEEP_MERGED = [
    ("ilya zabarnyi", "illia zabarnyi"),
    ("vitalii mykolenko", "vitaliy mykolenko"),
    ("hicham boudaoui", "hichem boudaoui"),
    ("anatolii trubin", "anatoliy trubin"),
    ("clatous chama", "cletous chama"),
    ("ehsan haddad", "ihsan haddad"),
    ("gismat aliyev", "qismat aliyev"),
    ("ebert martinez", "evert martinez"),
    ("srdjan babic", "srđan babic"),
    ("sahruddin mahammadaliyev", "shahrudin mahammadaliyev"),
    ("joon-ho bae", "jun-ho bae"),
    ("khayal aliyev", "xayal aliyev"),
    # NOT merged (genuinely different people): marco/mario pasalic, dayro/yairo moreno
]

_TRANSLIT = str.maketrans({"ø": "o", "Ø": "O", "æ": "ae", "Æ": "AE", "å": "a", "Å": "A",
                           "ð": "d", "Ð": "D", "þ": "th", "Þ": "TH", "ł": "l", "Ł": "L"})


def _norm(s) -> str:
    s = str(s).translate(_TRANSLIT)
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn").lower()
    s = "".join(c if (c.isalnum() or c in " -") else "" for c in s)   # drop apostrophes/punct
    return " ".join(s.split())


def _one_indel(short: str, lng: str) -> bool:
    """True if `lng` is `short` with exactly one char inserted (doubled/dropped letter)."""
    i = j = 0
    skipped = False
    while i < len(short) and j < len(lng):
        if short[i] == lng[j]:
            i += 1
            j += 1
        elif skipped:
            return False
        else:
            skipped = True
            j += 1
    return True


def _tok_match(t: str, u: str) -> bool:
    if t == u:
        return True
    if len(t) >= 3 and u.startswith(t):
        return True
    if len(u) >= 3 and t.startswith(u):
        return True
    if abs(len(t) - len(u)) == 1:
        s, l = (t, u) if len(t) < len(u) else (u, t)
        return _one_indel(s, l)
    return False


_KEEP = {frozenset(p) for p in KEEP_MERGED}


def _same_person(a: str, b: str) -> bool:
    if frozenset((a, b)) in _KEEP:
        return True
    ta, tb = a.split(), b.split()
    if not ta or not tb:
        return False
    small, big = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    used = [False] * len(big)
    for t in small:
        ok = False
        for i, u in enumerate(big):
            if used[i]:
                continue
            if _tok_match(t, u):
                used[i] = True
                ok = True
                break
        if not ok:
            return False
    return True


def _cluster(names: list[str]) -> list[list[str]]:
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if _same_person(a, b):
                parent[find(a)] = find(b)
    out: dict[str, list[str]] = {}
    for n in names:
        out.setdefault(find(n), []).append(n)
    return list(out.values())


def _subset_match(a: str, b: str) -> bool:
    """One name's tokens are an exact subset of the other's (>=2 tokens), or equal."""
    if a == b:
        return True
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return False
    small, big = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    return len(small) >= 2 and small <= big


def split_conflations(pm: pd.DataFrame) -> pd.DataFrame:
    """Pass 1: split player_ids that hold >1 distinct person."""
    n = pm["player"].map(_norm)
    rc = pm.groupby([pm["player_id"], n]).size()
    remap: dict[tuple, float] = {}
    counter = _NEWPID0
    for pid, idx in pm.groupby("player_id").groups.items():
        names = sorted(v for v in n.loc[idx].unique() if v not in MISSING)
        if len(names) <= 1:
            continue
        cl = _cluster(names)
        if len(cl) <= 1:
            continue
        cl.sort(key=lambda c: -sum(int(rc.get((pid, x), 0)) for x in c))
        for c in cl[1:]:
            counter += 1
            for x in c:
                remap[(pid, x)] = float(counter)
    if not remap:
        return pm
    keys = list(zip(pm["player_id"], n))
    newpid = pm["player_id"].astype("float64").to_numpy().copy()
    for pos, k in enumerate(keys):
        if k in remap:
            newpid[pos] = remap[k]
    pm = pm.copy()
    pm["player_id"] = newpid
    return pm


def merge_fragments(pm: pd.DataFrame) -> pd.DataFrame:
    """Pass 2: merge player_ids that are the same player split across sources."""
    n = pm["player"].map(_norm)
    team = pm["team"].map(_norm)
    rows_by_id = pm.groupby("player_id").size()
    name_by_id = (pd.DataFrame({"pid": pm["player_id"], "n": n})[~n.isin(MISSING)]
                  .groupby(["pid", "n"]).size().reset_index(name="c")
                  .sort_values("c").drop_duplicates("pid", keep="last")
                  .set_index("pid")["n"].to_dict())
    team_ids = pd.DataFrame({"team": team, "pid": pm["player_id"]}).groupby("team")["pid"].unique()

    parent: dict[float, float] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for ids in team_ids:
        ids = [i for i in ids if i in name_by_id]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if _subset_match(name_by_id[ids[i]], name_by_id[ids[j]]):
                    parent[find(ids[i])] = find(ids[j])

    groups: dict[float, set] = {}
    for i in list(parent):
        groups.setdefault(find(i), set()).add(i)
    remap: dict[float, float] = {}
    for g in groups.values():
        if len(g) <= 1:
            continue
        canon = max(g, key=lambda i: int(rows_by_id.get(i, 0)))
        for i in g:
            if i != canon:
                remap[i] = canon
    if not remap:
        return pm
    pm = pm.copy()
    pm["player_id"] = pm["player_id"].map(lambda i: remap.get(i, i))
    return pm


def clean_player_ids(pm: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Split conflations then merge fragments. Returns a corpus with one id per person."""
    n0 = pm["player_id"].nunique()
    pm = split_conflations(pm)
    n1 = pm["player_id"].nunique()
    pm = merge_fragments(pm)
    n2 = pm["player_id"].nunique()
    if verbose:
        print(f"[deconflate] player_id {n0} -> split {n1} (+{n1 - n0}) "
              f"-> merge {n2} ({n2 - n1}); net {n2 - n0}")
    return pm
