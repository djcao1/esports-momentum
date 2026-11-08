"""Version 4: best subset selection for the CS2 win-probability model.

Plan from the Nov 2 meeting:
  - Forward selection adds covariates one at a time, so it can miss combinations that work
    well together. Instead, test EVERY subset of covariates (best subset selection), so the
    order covariates are added in doesn't matter.
  - Look at subsets of 4 up to 10 covariates and find the best one at each size.
  - Expect diminishing returns; use the results to judge whether this dataset is good enough.

Candidates: the economy and skill covariates from v3 plus four new ones (round in half,
biggest single buy, typical player buy, typical money left). Score difference and loss
streaks are left out because they are built from earlier round results (momentum);
add --with-state to include them anyway.

Every subset is scored with the same 5-fold cross-validation grouped by map as v3, using
logistic regression. Accuracy, AUC and log loss are averages over the 5 folds.

Run:  python logit_v4.py data                  (sizes 4-10, about 3,800 subsets)
      python logit_v4.py data --min 1 --max 10 (also sizes 1-3)
      python logit_v4.py data --with-state     (add score difference and loss streaks)
"""
import itertools
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
args = sys.argv[1:]
DATA = next((a for a in args if not a.startswith("--") and not a.isdigit()), "data")
def opt(name, default):
    return int(args[args.index(name) + 1]) if name in args else default
MIN_K, MAX_K = opt("--min", 4), opt("--max", 10)
WITH_STATE = "--with-state" in args
OUT = "out"
os.makedirs(OUT, exist_ok=True)

# =============================================================================
# 1. Build one row per round (CT's point of view) with every candidate covariate
# =============================================================================
maps = pd.read_parquet(f"{DATA}/full_maps.parquet")
rounds = pd.read_parquet(f"{DATA}/full_rounds.parquet")
econ = pd.read_parquet(f"{DATA}/full_team_round_economy.parquet")

maps = maps[maps["analysis_eligible_v1"]]
rounds = rounds[rounds["map_uid"].isin(maps["map_uid"])]
econ = econ[econ["map_uid"].isin(maps["map_uid"])].copy()
econ["team"] = econ["team_name_resolved"].fillna(econ["roster_uid"]).astype(str)

# Per-side values, then CT and T side by side
cols = {"team_current_equip_value": "equip",            # equipment after buying (what they play with)
        "team_round_start_equip_value": "kept",         # equipment carried over from last round
        "team_spend": "spend",                           # money spent this round
        "team_balance": "money",                         # money left after buying
        "armor_players": "armor", "helmet_players": "helmet", "defuser_players": "kits",
        "max_player_spend": "maxbuy",                    # most one player spent (an AWP shows up here)
        "median_player_spend": "medbuy",                 # what a typical player spent
        "median_balance": "medmoney"}                    # what a typical player has left
side = econ[["map_uid", "round_num", "side", "team"] + list(cols)].rename(columns=cols)
wide = side.pivot_table(index=["map_uid", "round_num"], columns="side",
                        values=list(cols.values()), aggfunc="first")
wide.columns = [f"{s}_{v}" for v, s in wide.columns]
teams = side.pivot_table(index=["map_uid", "round_num"], columns="side", values="team", aggfunc="first")
teams.columns = [f"{s}_team" for s in teams.columns]
df = (rounds[["map_uid", "round_num", "map_name", "winner_side"]]
      .merge(wide.reset_index(), on=["map_uid", "round_num"])
      .merge(teams.reset_index(), on=["map_uid", "round_num"]))
df = df.dropna(subset=["ct_equip", "t_equip", "ct_team", "t_team"]).sort_values(["map_uid", "round_num"])
df["ct_win"] = (df["winner_side"] == "ct").astype(int)
df["map_name"] = df["map_name"].str.replace("de_", "").str.title()
df["half"] = np.where(df["round_num"] <= 12, 1, np.where(df["round_num"] <= 24, 2, 3))

# --- Economy covariates (known before the round starts) ---------------------
df["equip_gap"] = (df["ct_equip"] - df["t_equip"]) / 1000
df["kept_gap"] = (df["ct_kept"] - df["t_kept"]) / 1000       # carried-over gear
df["spend_gap"] = (df["ct_spend"] - df["t_spend"]) / 1000     # new purchases
df["money_gap"] = (df["ct_money"] - df["t_money"]) / 1000     # money saved for later
df["armor_gap"] = df["ct_armor"] - df["t_armor"]
df["helmet_gap"] = df["ct_helmet"] - df["t_helmet"]
df["ct_kits"] = df["ct_kits"]
df["max_buy_gap"] = (df["ct_maxbuy"] - df["t_maxbuy"]) / 1000
df["median_buy_gap"] = (df["ct_medbuy"] - df["t_medbuy"]) / 1000
df["median_money_gap"] = (df["ct_medmoney"] - df["t_medmoney"]) / 1000
df["round_in_half"] = np.where(df["round_num"] <= 12, df["round_num"], df["round_num"] - 12)
tiers = [-1, 4999, 9999, 19999, 1e9]
names = ["Eco", "SemiEco", "SemiBuy", "Full"]
df["buy_matchup"] = (pd.cut(df["ct_equip"], tiers, labels=names).astype(str) + "_v_" +
                     pd.cut(df["t_equip"], tiers, labels=names).astype(str))

# --- Game-state covariates (built from earlier round results) ---------------
# Careful: these use past results, so they could absorb momentum later on.
long = pd.concat([
    df[["map_uid", "round_num", "half", "ct_team", "ct_win"]].rename(columns={"ct_team": "team", "ct_win": "win"}),
    df[["map_uid", "round_num", "half", "t_team"]].rename(columns={"t_team": "team"}).assign(win=1 - df["ct_win"].values),
]).sort_values(["map_uid", "team", "round_num"])
long["score"] = long.groupby(["map_uid", "team"])["win"].cumsum() - long["win"]     # rounds won before this one
lost = 1 - long["win"]
# wins BEFORE this round (exclude the current result, or the model would see the answer)
grp = long.groupby(["map_uid", "team", "half"])["win"].cumsum() - long["win"]
long["loss_streak"] = (lost.groupby([long["map_uid"], long["team"], long["half"], grp]).cumsum() - lost).clip(upper=4)
# = losses since the team's last win in this half, counted before this round (capped at 4)
for s in ["ct", "t"]:
    df = df.merge(long[["map_uid", "round_num", "team", "score", "loss_streak"]]
                  .rename(columns={"team": f"{s}_team", "score": f"{s}_score", "loss_streak": f"{s}_loss_streak"}),
                  on=["map_uid", "round_num", f"{s}_team"], how="left")
df["score_diff"] = df["ct_score"] - df["t_score"]

# --- Team skill: pre-map Elo from earlier maps only --------------------------
mp = maps[["map_uid", "match_date", "map_index"]]
res = df.groupby("map_uid").agg(a=("ct_team", "first"), b=("t_team", "first")).reset_index().merge(mp, on="map_uid")
wins = long.groupby(["map_uid", "team"])["win"].sum()
res["a_wins"] = [wins.get((m, t), 0) for m, t in zip(res["map_uid"], res["a"])]
res["b_wins"] = [wins.get((m, t), 0) for m, t in zip(res["map_uid"], res["b"])]
res = res.sort_values(["match_date", "map_index", "map_uid"])
elo, K, pre = {}, 32, {}
for m, a, b, wa, wb in zip(res["map_uid"], res["a"], res["b"], res["a_wins"], res["b_wins"]):
    ra, rb = elo.get(a, 1500.0), elo.get(b, 1500.0)
    pre[(m, a)], pre[(m, b)] = ra, rb
    ea = 1 / (1 + 10 ** ((rb - ra) / 400))
    sa = 1.0 if wa > wb else 0.0
    elo[a], elo[b] = ra + K * (sa - ea), rb - K * (sa - ea)
df["elo_gap"] = [(pre.get((m, c), 1500) - pre.get((m, t), 1500)) / 100
                 for m, c, t in zip(df["map_uid"], df["ct_team"], df["t_team"])]   # per 100 Elo

# --- Same rounds as v2: non-pistol regulation rounds -------------------------
df = df[df["round_num"].between(2, 12) | df["round_num"].between(14, 24)].reset_index(drop=True)
counts = df["buy_matchup"].value_counts()
df["buy_matchup"] = df["buy_matchup"].where(df["buy_matchup"].map(counts) >= 100, "Rare")
print(f"{len(df):,} non-pistol regulation rounds from {df['map_uid'].nunique():,} maps\n")

# =============================================================================
# 2. Candidate covariates (each one may stand for several columns)
# =============================================================================
CANDIDATES = {
    "equipment gap":        ["equip_gap"],
    "buy matchup":          ["C:buy_matchup"],
    "map":                  ["C:map_name"],
    "armor + helmets":      ["armor_gap", "helmet_gap"],
    "defuse kits":          ["ct_kits"],
    "kept vs. bought gear": ["kept_gap"],
    "money left":           ["money_gap"],
    "team skill (Elo)":     ["elo_gap"],
    "round in half":        ["round_in_half"],
    "biggest single buy":   ["max_buy_gap"],       # new: an AWP or big rifle buy
    "typical player buy":   ["median_buy_gap"],    # new
    "typical money left":   ["median_money_gap"],  # new
}
if WITH_STATE:
    CANDIDATES["score difference *"] = ["score_diff"]
    CANDIDATES["loss streaks *"] = ["ct_loss_streak", "t_loss_streak"]
NAMES = list(CANDIDATES)

# Build every column once, and remember which columns belong to which covariate
parts, col_idx, pos = [], {}, 0
for name, cols_ in CANDIDATES.items():
    idx = []
    for c in cols_:
        block = (pd.get_dummies(df[c[2:]], prefix=c[2:], drop_first=True, dtype=float)
                 if c.startswith("C:") else df[[c]].astype(float))
        parts.append(block)
        idx += list(range(pos, pos + block.shape[1])); pos += block.shape[1]
    col_idx[name] = idx
X_all = pd.concat(parts, axis=1).fillna(0).values
y = df["ct_win"].values

# Same 5 folds for every subset; scale with each fold's training rounds only
FOLDS = []
for tr, te in GroupKFold(n_splits=5).split(X_all, y, groups=df["map_uid"]):
    sc = StandardScaler().fit(X_all[tr])
    FOLDS.append((sc.transform(X_all[tr]), y[tr], sc.transform(X_all[te]), y[te]))


def score(subset):
    """Mean held-out accuracy, AUC and log loss of logistic regression on this subset."""
    idx = [i for n in subset for i in col_idx[n]]
    acc, auc, ll = [], [], []
    for Xtr, ytr, Xte, yte in FOLDS:
        m = LogisticRegression(max_iter=2000).fit(Xtr[:, idx], ytr)
        p = np.clip(m.predict_proba(Xte[:, idx])[:, 1], 1e-6, 1 - 1e-6)
        acc.append(accuracy_score(yte, p > 0.5)); auc.append(roc_auc_score(yte, p)); ll.append(log_loss(yte, p))
    return {"size": len(subset), "covariates": " + ".join(subset),
            "accuracy": np.mean(acc), "AUC": np.mean(auc), "log loss": np.mean(ll),
            "accuracy sd": np.std(acc)}


# =============================================================================
# 3. Score every subset of size MIN_K..MAX_K
# =============================================================================
base = score(["equipment gap"])
subsets = [s for k in range(MIN_K, min(MAX_K, len(NAMES)) + 1) for s in itertools.combinations(NAMES, k)]
print(f"{len(NAMES)} candidate covariates, testing all {len(subsets):,} subsets of size {MIN_K}-{MAX_K} "
      f"(5 folds each, all CPU cores)...")
t0 = time.time()
results = Parallel(n_jobs=-1, batch_size=8)(delayed(score)(s) for s in subsets)
res = pd.DataFrame(results)
res.to_csv(f"{OUT}/v4_all_subsets.csv", index=False)
print(f"done in {time.time() - t0:.0f}s, saved {OUT}/v4_all_subsets.csv\n")

# =============================================================================
# 4. Best subset at each size
# =============================================================================
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 140)
print(f"Baseline, equipment gap only: accuracy {base['accuracy']:.4f}, AUC {base['AUC']:.4f}, log loss {base['log loss']:.4f}\n")
for metric, asc in [("accuracy", False), ("AUC", False)]:
    best = res.sort_values(metric, ascending=asc).groupby("size").head(1).sort_values("size")
    print(f"BEST SUBSET AT EACH SIZE, ranked by {metric}:")
    print(best[["size", "accuracy", "AUC", "log loss", "covariates"]].to_string(index=False, float_format="%.4f"))
    print()
best_acc = res.sort_values("accuracy", ascending=False).groupby("size").head(1).sort_values("size")
best_auc = res.sort_values("AUC", ascending=False).groupby("size").head(1).sort_values("size")
best_acc.to_csv(f"{OUT}/v4_best_by_size.csv", index=False)

# How often each covariate appears in the top 10 subsets of each size (by accuracy)
top = res.sort_values("accuracy", ascending=False).groupby("size").head(10)
freq = pd.DataFrame({n: top.groupby("size")["covariates"].apply(lambda s, n=n: s.str.contains(n, regex=False).sum())
                     for n in NAMES}).T
print("How many of the top 10 subsets (by accuracy) include each covariate, per size:")
print(freq.to_string())
noise = res["accuracy sd"].mean()
print(f"\nNote: accuracy varies by about {noise * 100:.1f} points from fold to fold, so differences of a few")
print("tenths of a point between the best subsets are within noise. Look for the size where the curve flattens.")

# =============================================================================
# 5. Chart: best performance vs number of covariates (diminishing returns)
# =============================================================================
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    a1.plot(best_acc["size"], best_acc["accuracy"] * 100, marker="o", color="#2a78d6", lw=2)
    a1.axhline(base["accuracy"] * 100, color="#b8b7b1", ls="--", lw=1)
    b = base["accuracy"] * 100
    a1.axhspan(b - noise * 100, b + noise * 100, color="#b8b7b1", alpha=0.2, lw=0)
    a1.set_ylim(b - 1, b + 1)   # +/-1 point, so fold-to-fold noise is visible in scale
    a1.set_title("Best accuracy at each size", loc="left", fontweight="bold")
    a1.set_xlabel("Number of covariates"); a1.set_ylabel("Held-out accuracy (%)")
    a2.plot(best_auc["size"], best_auc["AUC"], marker="o", color="#eb6834", lw=2)
    a2.axhline(base["AUC"], color="#b8b7b1", ls="--", lw=1)
    a2.set_title("Best AUC at each size", loc="left", fontweight="bold")
    a2.set_xlabel("Number of covariates"); a2.set_ylabel("Held-out AUC")
    for a in (a1, a2):
        a.spines[["top", "right"]].set_visible(False); a.grid(alpha=0.3)
    fig.text(0.01, 0.01, "Dashed line = equipment gap alone; grey band = typical fold-to-fold accuracy noise. Best of all subsets at each size, 5-fold CV grouped by map.",
             fontsize=8, color="#52514e")
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(f"{OUT}/v4_best_by_size.png", dpi=200)
    print(f"Saved {OUT}/v4_best_by_size.csv, {OUT}/v4_best_by_size.png")
except ImportError:
    pass