"""Shared helpers for the ADA 2026 project: predicting upsets in ATP matches.

Used by the notebooks in this folder. Everything that is needed in more than one
place lives here (requirement 4.9).

Design rule (instructor's note of 1 Nov 2026): every player feature is computed
only from matches played BEFORE the match being predicted, and train/test splits
are made by date. The history features are built in one chronological pass that
reads the state of each player first and updates it afterwards, so a match can
never see itself or anything that comes later. `check_no_lookahead` verifies this
independently.
"""
from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
FIRST_YEAR, LAST_YEAR = 2000, 2024

# Order of rounds inside a tournament (RR = round robin, played before the
# knockout stage; BR = bronze medal match, played before the final).
ROUND_ORDER = {"ER": 0, "R128": 1, "R64": 2, "R32": 3, "R16": 4, "RR": 4,
               "QF": 5, "SF": 6, "BR": 7, "F": 8}

# Scores that mean the match was not played to a result we can use as history.
# RET (retirement) is different: the match started, so it counts as played.
NOT_PLAYED_PATTERN = r"W/O|Walkover|DEF|Def|Default|unfinished|abandoned|ABD"
RETIRED_PATTERN = r"RET"

FORM_WINDOW = 10      # form = win rate over the player's last 10 matches
SURFACE_SMOOTH = 4    # surface record shrunk towards 50% with weight of 4 matches
DAYS_CAP = 365        # days since last match capped at one year (heavy right tail)

# Elo rating (overall and per surface), updated match by match
ELO_START = 1500.0
ELO_K_SCALE, ELO_K_SHIFT, ELO_K_EXP = 250.0, 5.0, 0.4   # K = 250 / (n + 5) ** 0.4


# ----------------------------------------------------------------------------
# Loading and preparation
# ----------------------------------------------------------------------------
def load_raw(data_dir: str | Path, first_year: int = FIRST_YEAR,
             last_year: int = LAST_YEAR) -> pd.DataFrame:
    """Read the yearly ATP match files (Sackmann) and stack them."""
    data_dir = Path(data_dir)
    frames = [pd.read_csv(data_dir / f"atp_matches_{y}.csv", low_memory=False)
              for y in range(first_year, last_year + 1)]
    return pd.concat(frames, ignore_index=True)


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    """Add dates, round order and played/finished flags; sort chronologically.

    The chronological order is: tournament start date, round, tournament id,
    match number. `pos` is the position in that order and is the only notion of
    "before" used anywhere in the project.

    Matches in the same tournament share `tourney_date` (the start of the event),
    so a calendar date for each match is approximated as start date plus a number
    of days that grows with the round (2 days per round at Grand Slams, 1 elsewhere).
    It is only used to measure days of inactivity.
    """
    df = raw.copy()
    df["tourney_date"] = pd.to_datetime(df["tourney_date"].astype(int).astype(str),
                                        format="%Y%m%d")
    df["round_order"] = df["round"].map(ROUND_ORDER)
    if df["round_order"].isna().any():
        unknown = df.loc[df["round_order"].isna(), "round"].unique()
        raise ValueError(f"Unknown round labels: {unknown}")

    score = df["score"].fillna("")
    df["played"] = ~score.str.contains(NOT_PLAYED_PATTERN, regex=True)
    df["retired"] = score.str.contains(RETIRED_PATTERN, regex=True)
    df["finished"] = df["played"] & ~df["retired"]

    df = df.sort_values(["tourney_date", "round_order", "tourney_id", "match_num"],
                        kind="mergesort").reset_index(drop=True)
    step = np.where(df["tourney_level"] == "G", 2, 1)
    df["approx_date"] = df["tourney_date"] + pd.to_timedelta(
        np.maximum(df["round_order"] - 1, 0) * step, unit="D")
    df["pos"] = np.arange(len(df))
    return df


# ----------------------------------------------------------------------------
# History features (one chronological pass, state read before update)
# ----------------------------------------------------------------------------
HIST_COLS = ["days", "form", "surf", "n_prior", "elo", "selo"]


def _elo_k(n_matches: int) -> float:
    """Learning rate of the Elo update: large for players with few matches."""
    return ELO_K_SCALE / (n_matches + ELO_K_SHIFT) ** ELO_K_EXP


def _elo_expected(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """For winner (w_) and loser (l_) of each match compute, using earlier
    matches only: days since last match, form (win rate over last 10), win rate
    on the match surface (smoothed), number of earlier matches; plus head-to-head
    wins of each side against the other. It also tracks an Elo rating (overall,
    `elo`) and an Elo rating per surface (`selo`), both read before the match and
    updated after it.

    `df` must come from `prepare`. Matches flagged not played neither get
    features nor enter anyone's history. Retirements do enter the history.
    """
    n = len(df)
    out = {f"{side}_{c}": np.full(n, np.nan) for side in ("w", "l") for c in HIST_COLS}
    out["h2h_w"] = np.full(n, np.nan)
    out["h2h_l"] = np.full(n, np.nan)

    last_date: dict = {}
    recent = defaultdict(lambda: deque(maxlen=FORM_WINDOW))
    n_played = defaultdict(int)
    surf_rec = defaultdict(lambda: [0, 0])     # (player, surface) -> [wins, matches]
    h2h = defaultdict(int)                      # (a, b) -> wins of a over b
    elo = defaultdict(lambda: ELO_START)        # player -> rating
    selo = defaultdict(lambda: ELO_START)       # (player, surface) -> rating
    selo_n = defaultdict(int)                   # (player, surface) -> matches played

    winner, loser = df["winner_id"].to_numpy(), df["loser_id"].to_numpy()
    surface = df["surface"].to_numpy(dtype=object)
    date = df["approx_date"].to_numpy()
    played = df["played"].to_numpy()

    for i in range(n):
        if not played[i]:
            continue
        w, l, s, d = winner[i], loser[i], surface[i], date[i]
        has_surface = isinstance(s, str)

        # 1) read the state (strictly earlier matches only)
        for side, p in (("w", w), ("l", l)):
            if p in last_date:
                out[f"{side}_days"][i] = max((d - last_date[p]) / np.timedelta64(1, "D"), 0.0)
            if recent[p]:
                out[f"{side}_form"][i] = sum(recent[p]) / len(recent[p])
            if has_surface:
                wins, tot = surf_rec[(p, s)]
                out[f"{side}_surf"][i] = (wins + SURFACE_SMOOTH * 0.5) / (tot + SURFACE_SMOOTH)
            out[f"{side}_n_prior"][i] = n_played[p]
            out[f"{side}_elo"][i] = elo[p]
            out[f"{side}_selo"][i] = selo[(p, s)] if has_surface else elo[p]
        out["h2h_w"][i] = h2h[(w, l)]
        out["h2h_l"][i] = h2h[(l, w)]

        # 2) only now update the state with this match
        # Elo: both updates use the ratings read above (before the match)
        e_w = _elo_expected(elo[w], elo[l])
        k_w, k_l = _elo_k(n_played[w]), _elo_k(n_played[l])
        elo_w_new = elo[w] + k_w * (1.0 - e_w)
        elo_l_new = elo[l] - k_l * (1.0 - e_w)
        if has_surface:
            es_w = _elo_expected(selo[(w, s)], selo[(l, s)])
            ks_w, ks_l = _elo_k(selo_n[(w, s)]), _elo_k(selo_n[(l, s)])
            selo[(w, s)] += ks_w * (1.0 - es_w)
            selo[(l, s)] -= ks_l * (1.0 - es_w)
            selo_n[(w, s)] += 1
            selo_n[(l, s)] += 1
        elo[w], elo[l] = elo_w_new, elo_l_new
        for p, won in ((w, 1), (l, 0)):
            last_date[p] = d
            recent[p].append(won)
            n_played[p] += 1
            if has_surface:
                rec = surf_rec[(p, s)]
                rec[0] += won
                rec[1] += 1
        h2h[(w, l)] += 1

    return df.assign(**out)


# ----------------------------------------------------------------------------
# Modelling table: favourite / underdog
# ----------------------------------------------------------------------------
def build_model_table(featured: pd.DataFrame) -> pd.DataFrame:
    """Keep finished matches with both rankings and distinct ranks; express
    everything from the favourite's (better rank) and underdog's point of view.

    upset = 1 when the underdog (worse-ranked player) wins. Roles come from the
    pre-match ranking, so no shuffling of player order is needed.
    """
    d = featured
    keep = (d["finished"] & d["winner_rank"].notna() & d["loser_rank"].notna()
            & (d["winner_rank"] != d["loser_rank"]))
    d = d.loc[keep].copy()
    fav_w = (d["winner_rank"] < d["loser_rank"]).to_numpy()

    def pick(col_w, col_l, fav=True):
        a, b = d[col_w].to_numpy(), d[col_l].to_numpy()
        return np.where(fav_w, a, b) if fav else np.where(fav_w, b, a)

    t = pd.DataFrame(index=d.index)
    t["pos"] = d["pos"]
    t["tourney_id"] = d["tourney_id"]
    t["tourney_date"] = d["tourney_date"]
    t["year"] = d["tourney_date"].dt.year
    t["tourney_level"] = d["tourney_level"]
    t["round"] = d["round"]
    t["surface"] = d["surface"]
    t["best_of"] = d["best_of"]
    t["fav_id"] = pick("winner_id", "loser_id")
    t["und_id"] = pick("winner_id", "loser_id", fav=False)

    t["fav_rank"] = pick("winner_rank", "loser_rank")
    t["und_rank"] = pick("winner_rank", "loser_rank", fav=False)
    t["fav_points"] = pick("winner_rank_points", "loser_rank_points")
    t["und_points"] = pick("winner_rank_points", "loser_rank_points", fav=False)
    for c in HIST_COLS:
        t[f"fav_{c}"] = pick(f"w_{c}", f"l_{c}")
        t[f"und_{c}"] = pick(f"w_{c}", f"l_{c}", fav=False)
    t["fav_h2h"] = pick("h2h_w", "h2h_l")
    t["und_h2h"] = pick("h2h_w", "h2h_l", fav=False)

    # derived features
    t["rank_gap"] = t["und_rank"] - t["fav_rank"]
    t["log_rank_ratio"] = np.log(t["und_rank"] / t["fav_rank"])
    t["log_points_ratio"] = np.log(t["fav_points"].clip(lower=1) / t["und_points"].clip(lower=1))
    for who in ("fav", "und"):
        t[f"{who}_days"] = t[f"{who}_days"].clip(upper=DAYS_CAP)
        t[f"{who}_no_history"] = t[f"{who}_n_prior"].eq(0).astype(int)
    t["form_diff"] = t["fav_form"] - t["und_form"]
    t["surf_diff"] = t["fav_surf"] - t["und_surf"]
    t["h2h_diff"] = t["fav_h2h"] - t["und_h2h"]
    t["h2h_total"] = t["fav_h2h"] + t["und_h2h"]
    t["elo_diff"] = t["fav_elo"] - t["und_elo"]      # favourite minus underdog
    t["selo_diff"] = t["fav_selo"] - t["und_selo"]

    t["upset"] = (~fav_w).astype(int)
    return t.reset_index(drop=True)


# ----------------------------------------------------------------------------
# Splits by date
# ----------------------------------------------------------------------------
def year_splits(table: pd.DataFrame, first_test_year: int, last_test_year: int):
    """Expanding-window splits by calendar year: for each test year Y the train
    set is every match of years before Y and the test set is year Y.
    Yields (Y, train_positions, test_positions) as integer row positions."""
    years = table["year"].to_numpy()
    for y in range(first_test_year, last_test_year + 1):
        yield y, np.flatnonzero(years < y), np.flatnonzero(years == y)


# ----------------------------------------------------------------------------
# Verification that no feature uses future information
# ----------------------------------------------------------------------------
def _brute_force(prep: pd.DataFrame, pos: int, player, opponent, surface) -> dict:
    """Recompute one player's features for the match at `pos` by filtering the
    full data frame to strictly earlier played matches (independent of the loop)."""
    prior = prep.iloc[:pos]
    prior = prior[prior["played"] & ((prior["winner_id"] == player) | (prior["loser_id"] == player))]
    now = prep.at[pos, "approx_date"]
    res = {"days": np.nan, "form": np.nan, "surf": np.nan, "n_prior": float(len(prior)),
           "h2h": 0.0}
    if prior.empty:
        res["surf"] = 0.5 if isinstance(surface, str) else np.nan
        return res
    won = (prior["winner_id"] == player).to_numpy()
    res["days"] = max((now - prior["approx_date"].iloc[-1]) / pd.Timedelta(days=1), 0.0)
    res["form"] = won[-FORM_WINDOW:].mean()
    if isinstance(surface, str):
        on_s = (prior["surface"] == surface).to_numpy()
        res["surf"] = (won[on_s].sum() + SURFACE_SMOOTH * 0.5) / (on_s.sum() + SURFACE_SMOOTH)
    vs = prior[(prior["winner_id"] == opponent) | (prior["loser_id"] == opponent)]
    res["h2h"] = float((vs["winner_id"] == player).sum())
    return res


def _elo_reference(prep: pd.DataFrame) -> dict:
    """Second, independent implementation of the Elo ratings (plain loop over
    match tuples, no shared code with add_history_features). Returns
    {pos: (elo_w, elo_l, selo_w, selo_l)} as read before each match."""
    rating, srating, count, scount = {}, {}, {}, {}
    ref = {}
    for r in prep.itertuples(index=False):
        if not r.played:
            continue
        w, l, s = r.winner_id, r.loser_id, r.surface
        surf = isinstance(s, str)
        rw, rl = rating.get(w, 1500.0), rating.get(l, 1500.0)
        sw = srating.get((w, s), 1500.0) if surf else rw
        sl = srating.get((l, s), 1500.0) if surf else rl
        ref[r.pos] = (rw, rl, sw, sl)
        pw = 1 / (1 + 10 ** ((rl - rw) / 400))
        kw = 250 / (count.get(w, 0) + 5) ** 0.4
        kl = 250 / (count.get(l, 0) + 5) ** 0.4
        rating[w], rating[l] = rw + kw * (1 - pw), rl - kl * (1 - pw)
        count[w], count[l] = count.get(w, 0) + 1, count.get(l, 0) + 1
        if surf:
            pws = 1 / (1 + 10 ** ((sl - sw) / 400))
            ksw = 250 / (scount.get((w, s), 0) + 5) ** 0.4
            ksl = 250 / (scount.get((l, s), 0) + 5) ** 0.4
            srating[(w, s)], srating[(l, s)] = sw + ksw * (1 - pws), sl - ksl * (1 - pws)
            scount[(w, s)], scount[(l, s)] = scount.get((w, s), 0) + 1, scount.get((l, s), 0) + 1
    return ref


def check_no_lookahead(prep: pd.DataFrame, table: pd.DataFrame, n_samples: int = 300,
                       n_truncations: int = 5, seed: int = 0) -> dict:
    """Two independent checks, raising AssertionError on any mismatch.

    1. Brute force: for a random sample of matches, recompute both players'
       features by filtering to strictly earlier matches and compare with the table.
       The Elo ratings are compared with a second, independent implementation.
    2. Truncation: cut the data right after a random match, rebuild the features
       from scratch and compare that match's features with the full run. If any
       feature used later matches, the values would differ.
    """
    rng = np.random.default_rng(seed)

    # 1) brute force against the final table (also checks the favourite/underdog mapping)
    sample = table.iloc[rng.choice(len(table), size=min(n_samples, len(table)), replace=False)]
    for row in sample.itertuples(index=False):
        for who, opp in (("fav", "und"), ("und", "fav")):
            exp = _brute_force(prep, row.pos, getattr(row, f"{who}_id"),
                               getattr(row, f"{opp}_id"), row.surface)
            for c in ("days", "form", "surf", "n_prior"):
                got = getattr(row, f"{who}_{c}")
                if c == "days" and not np.isnan(exp["days"]):
                    exp_v = min(exp["days"], DAYS_CAP)
                else:
                    exp_v = exp[c]
                assert np.isclose(got, exp_v, equal_nan=True), (
                    f"{who}_{c} mismatch at pos {row.pos}: table {got}, brute force {exp_v}")
            assert row.__getattribute__(f"{who}_h2h") == exp["h2h"], (
                f"{who}_h2h mismatch at pos {row.pos}")

    # 1b) Elo against an independent second implementation (all sampled matches)
    ref = _elo_reference(prep)
    for row in sample.itertuples(index=False):
        rw, rl, sw, sl = ref[row.pos]
        fav_is_w = row.fav_id == prep.at[row.pos, "winner_id"]
        exp = (rw, rl, sw, sl) if fav_is_w else (rl, rw, sl, sw)
        got = (row.fav_elo, row.und_elo, row.fav_selo, row.und_selo)
        assert np.allclose(got, exp), f"Elo mismatch at pos {row.pos}: {got} vs {exp}"

    # 2) truncation invariance
    full = add_history_features(prep)
    positions = rng.choice(prep.index[prep["played"]].to_numpy(), size=n_truncations, replace=False)
    cols = [f"{s}_{c}" for s in ("w", "l") for c in HIST_COLS] + ["h2h_w", "h2h_l"]
    for p in positions:
        cut = add_history_features(prep.iloc[: p + 1].copy())
        a, b = cut.loc[p, cols].to_numpy(dtype=float), full.loc[p, cols].to_numpy(dtype=float)
        assert np.allclose(a, b, equal_nan=True), f"Truncation changed features at pos {p}"

    return {"brute_force_matches": len(sample), "truncation_points": len(positions)}


# ----------------------------------------------------------------------------
# Modelling (used by notebook 03)
# ----------------------------------------------------------------------------
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss, log_loss,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANK_FEATURES = ["log_rank_ratio", "log_points_ratio"]
CONTEXT_NUM = ["best_of"]
CONTEXT_CAT = ["surface", "round", "tourney_level"]
HISTORY_FEATURES = ["fav_days", "und_days", "fav_form", "und_form", "form_diff",
                    "fav_surf", "und_surf", "surf_diff", "h2h_diff", "h2h_total"]

# name -> (numeric columns, categorical columns)
ELO_FEATURES = ["elo_diff", "selo_diff"]

FEATURE_SETS = {
    "rank": (RANK_FEATURES, []),
    "rank+context": (RANK_FEATURES + CONTEXT_NUM, CONTEXT_CAT),
    "rank+elo": (RANK_FEATURES + ELO_FEATURES, []),
    "all": (RANK_FEATURES + CONTEXT_NUM + HISTORY_FEATURES, CONTEXT_CAT),
    "all+elo": (RANK_FEATURES + CONTEXT_NUM + HISTORY_FEATURES + ELO_FEATURES, CONTEXT_CAT),
}

# groups of columns used for the permutation importance
FEATURE_GROUPS = {
    "ranking gap": RANK_FEATURES,
    "surface, round, level, best of": CONTEXT_NUM + CONTEXT_CAT,
    "days since last match": ["fav_days", "und_days"],
    "recent form": ["fav_form", "und_form", "form_diff"],
    "surface record": ["fav_surf", "und_surf", "surf_diff"],
    "head-to-head": ["h2h_diff", "h2h_total"],
    "Elo ratings": ELO_FEATURES,
}


def make_model(kind: str, **params):
    """Return an unfitted pipeline. kind is 'logreg', 'knn' or 'gboost'."""
    num, cat = params.pop("num"), params.pop("cat")
    scaled = kind in ("logreg", "knn")
    if scaled:
        num_tf = Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                           ("sc", StandardScaler())])
    else:
        num_tf = "passthrough"          # gradient boosting handles NaN natively
    parts = [("num", num_tf, num)]
    if cat:
        cat_tf = Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="missing")),
            ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=50,
                                 sparse_output=False))])
        parts.append(("cat", cat_tf, cat))
    pre = ColumnTransformer(parts)
    if kind == "logreg":
        est = LogisticRegression(C=params.get("C", 1.0), max_iter=2000)
    elif kind == "knn":
        est = KNeighborsClassifier(n_neighbors=params.get("k", 100), n_jobs=-1)
    elif kind == "gboost":
        est = HistGradientBoostingClassifier(
            learning_rate=params.get("lr", 0.05), max_iter=params.get("n", 200),
            max_depth=params.get("depth", 3), min_samples_leaf=100,
            l2_regularization=1.0, random_state=0)
    else:
        raise ValueError(kind)
    return Pipeline([("pre", pre), ("est", est)])


def scores(y, p, threshold: float) -> dict:
    """Metrics for the upset class (label 1) given predicted probabilities p."""
    y = np.asarray(y)
    pred = (p >= threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0)
    return {"auc": roc_auc_score(y, p), "log_loss": log_loss(y, p),
            "brier": brier_score_loss(y, p), "accuracy": float((pred == y).mean()),
            "precision": prec, "recall": rec, "f1": f1}


def best_f1_threshold(y, p) -> float:
    """Probability threshold that maximises F1 for the upset class."""
    grid = np.arange(0.10, 0.70, 0.01)
    f1s = [precision_recall_fscore_support(y, (p >= t).astype(int), average="binary",
                                           zero_division=0)[2] for t in grid]
    return float(grid[int(np.argmax(f1s))])
