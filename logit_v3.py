"""Version 3: improve the CS2 win-probability model.

Goals this week (CS2 only, no momentum yet):
  1. Try more covariates and measure how much each improves held-out AUC and accuracy.
  2. Drop covariates that add little.
  3. Compare logistic regression with other models (logistic model tree, random forest, XGBoost).

Every number is a 5-fold cross-validation grouped by map (each map is tested once, never
trained on in the same fold), reported as the mean over folds. This is steadier than the
single 80/20 split used in v1/v2, which matters when gains are as small as +0.001 AUC.

Same rounds as logit_v2.py: non-pistol regulation rounds (2-12, 14-24), from CT's point of view.

Run:  python logit_v3.py data            (add --quick to skip the slower model comparison)
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
DATA = next((a for a in sys.argv[1:] if not a.startswith("--")), "data")
QUICK = "--quick" in sys.argv
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
        "armor_players": "armor", "helmet_players": "helmet", "defuser_players": "kits"}
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

# Economy covariates (known before the round starts) 
df["equip_gap"] = (df["ct_equip"] - df["t_equip"]) / 1000
df["kept_gap"] = (df["ct_kept"] - df["t_kept"]) / 1000       # carried-over gear
df["spend_gap"] = (df["ct_spend"] - df["t_spend"]) / 1000     # new purchases
df["money_gap"] = (df["ct_money"] - df["t_money"]) / 1000     # money saved for later
df["armor_gap"] = df["ct_armor"] - df["t_armor"]
df["helmet_gap"] = df["ct_helmet"] - df["t_helmet"]
df["ct_kits"] = df["ct_kits"]
tiers = [-1, 4999, 9999, 19999, 1e9]
names = ["Eco", "SemiEco", "SemiBuy", "Full"]
df["buy_matchup"] = (pd.cut(df["ct_equip"], tiers, labels=names).astype(str) + "_v_" +
                     pd.cut(df["t_equip"], tiers, labels=names).astype(str))

# Game-state covariates (built from earlier round results)
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

# Team skill: pre-map Elo from earlier maps only
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

#Same rounds as v2: non-pistol regulation rounds
df = df[df["round_num"].between(2, 12) | df["round_num"].between(14, 24)].reset_index(drop=True)
counts = df["buy_matchup"].value_counts()
df["buy_matchup"] = df["buy_matchup"].where(df["buy_matchup"].map(counts) >= 100, "Rare")
print(f"{len(df):,} non-pistol regulation rounds from {df['map_uid'].nunique():,} maps\n")

# =============================================================================
# 2. Helpers
# =============================================================================
class LogisticModelTree:
    """Simple logistic model tree: a shallow decision tree splits rounds into groups
    (e.g. big vs. small equipment gaps), then a separate logistic regression is fit
    in each group. Like Weka's LMT in spirit, kept small so it is easy to explain."""

    def __init__(self, max_depth=2, min_leaf=0.1):
        self.max_depth, self.min_leaf = max_depth, min_leaf

    def fit(self, X, y):
        from sklearn.tree import DecisionTreeClassifier
        self.tree = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_leaf=self.min_leaf,
                                           random_state=0).fit(X, y)
        leaves = self.tree.apply(X)
        self.models = {}
        for leaf in np.unique(leaves):
            m = leaves == leaf
            if len(np.unique(y[m])) < 2:                 # one class only: predict its rate
                self.models[leaf] = float(y[m].mean())
            else:
                self.models[leaf] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X[m], y[m])
        return self

    def predict_proba(self, X):
        leaves = self.tree.apply(X)
        p = np.zeros(len(X))
        for leaf, model in self.models.items():
            m = leaves == leaf
            if m.any():
                p[m] = model if isinstance(model, float) else model.predict_proba(X[m])[:, 1]
        return np.column_stack([1 - p, p])


# =============================================================================
# 2b. Design matrix + grouped cross-validation
# =============================================================================
CANDIDATES = {                        # name -> columns it adds (categoricals become 0/1 dummies)
    "map":                  ["C:map_name"],
    "buy matchup":          ["C:buy_matchup"],
    "kept vs. bought split": ["kept_gap"],          # with equip_gap in, this = kept gear vs. purchases
    "money left":           ["money_gap"],
    "armor + helmets":      ["armor_gap", "helmet_gap"],
    "defuse kits":          ["ct_kits"],
    "team skill (Elo)":     ["elo_gap"],
    "score difference *":   ["score_diff"],
    "loss streaks *":       ["ct_loss_streak", "t_loss_streak"],
}
STAR_NOTE = "* built from earlier round results: may absorb momentum, check before keeping"


def design(cols_):
    parts = []
    for c in cols_:
        if c.startswith("C:"):
            parts.append(pd.get_dummies(df[c[2:]], prefix=c[2:], drop_first=True, dtype=float))
        else:
            parts.append(df[[c]].astype(float))
    return pd.concat(parts, axis=1).fillna(0)


FOLDS = list(GroupKFold(n_splits=5).split(df, groups=df["map_uid"]))
y = df["ct_win"].values


def cv(cols_, make_model=None):
    """Mean held-out AUC / accuracy / log loss over the 5 map-grouped folds, plus per-fold AUCs."""
    X = design(cols_).values
    aucs, accs, lls = [], [], []
    for tr, te in FOLDS:
        model = make_model() if make_model else make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        model.fit(X[tr], y[tr])
        p = np.clip(model.predict_proba(X[te])[:, 1], 1e-6, 1 - 1e-6)
        aucs.append(roc_auc_score(y[te], p)); accs.append(accuracy_score(y[te], p > 0.5)); lls.append(log_loss(y[te], p))
    return {"AUC": np.mean(aucs), "accuracy": np.mean(accs), "log loss": np.mean(lls), "fold_aucs": np.array(aucs)}


# =============================================================================
# 3. Goal 1: add each covariate to the baseline, one at a time
# =============================================================================
BASE = ["equip_gap"]
base = cv(BASE)
rows = []
for name, extra in CANDIDATES.items():
    r = cv(BASE + extra)
    gain = r["fold_aucs"] - base["fold_aucs"]
    rows.append({"covariate added to equip_gap": name, "AUC": r["AUC"], "AUC gain": gain.mean(),
                 "folds improved": f"{(gain > 0).sum()}/5", "accuracy": r["accuracy"],
                 "accuracy gain": r["accuracy"] - base["accuracy"], "log loss": r["log loss"]})
single = pd.DataFrame(rows).sort_values("AUC gain", ascending=False)
print(f"Baseline (equip_gap only): AUC {base['AUC']:.4f}, accuracy {base['accuracy']:.4f}, log loss {base['log loss']:.4f}\n")
print("GOAL 1 - each covariate added to the baseline on its own (5-fold CV, grouped by map):")
print(single.to_string(index=False, float_format=lambda v: f"{v:+.4f}" if abs(v) < 0.05 else f"{v:.4f}"))
print(STAR_NOTE + "\n")
single.to_csv(f"{OUT}/v3_covariates.csv", index=False)

# =============================================================================
# 4. Goal 2: forward selection - keep covariates that clearly help, drop the rest
#    Start from equip_gap. Each step, add the covariate that raises AUC the most.
#    Stop when the best one adds less than +0.001 AUC or helps in fewer than 4/5 folds.
#    Starred (game-state) covariates are left out of the main model and shown separately.
# =============================================================================
MIN_GAIN, MIN_FOLDS = 0.001, 4
pool = [n for n in CANDIDATES if "*" not in n]
keep, cur_cols, cur = [], list(BASE), base
steps = [{"step": "start", "covariate": "equip_gap", "AUC": base["AUC"], "AUC gain": np.nan, "folds improved": ""}]
while pool:
    tries = {n: cv(cur_cols + CANDIDATES[n]) for n in pool}
    best = max(tries, key=lambda n: tries[n]["AUC"])
    gain = tries[best]["fold_aucs"] - cur["fold_aucs"]
    if gain.mean() < MIN_GAIN or (gain > 0).sum() < MIN_FOLDS:
        break
    keep.append(best); pool.remove(best); cur_cols += CANDIDATES[best]; cur = tries[best]
    steps.append({"step": f"+{len(keep)}", "covariate": best, "AUC": cur["AUC"], "AUC gain": gain.mean(),
                  "folds improved": f"{(gain > 0).sum()}/5"})
drop = pool
SELECTED = cur_cols
WITH_STATE = SELECTED + [c for n in CANDIDATES if "*" in n for c in CANDIDATES[n]]
ALL = BASE + [c for n in CANDIDATES if "*" not in n for c in CANDIDATES[n]]
print(f"GOAL 2 - forward selection (stop when the best addition gains < +{MIN_GAIN} AUC or < {MIN_FOLDS}/5 folds)")
print(pd.DataFrame(steps).to_string(index=False, float_format=lambda v: f"{v:+.4f}" if abs(v) < 0.05 else f"{v:.4f}"))
print("  kept:", keep or "nothing beyond equip_gap")
print("  dropped:", drop)
summary = []
for label, cols_ in [("A  equip_gap only", BASE), ("B  selected covariates", SELECTED),
                     ("C  all economy + skill covariates", ALL), ("D  selected + game state *", WITH_STATE)]:
    r = cv(cols_)
    summary.append({"model": label, "AUC": r["AUC"], "accuracy": r["accuracy"], "log loss": r["log loss"]})
summary = pd.DataFrame(summary)
print(summary.to_string(index=False, float_format="%.4f"))
print("  B vs. C: did the dropped covariates matter when all kept together?  D: what game state adds (*).\n")

# =============================================================================
# 5. Goal 3: other model types, same covariates (selected set)
# =============================================================================
if not QUICK:
    from xgboost import XGBClassifier
    MODELS = {
        "Logistic regression": None,
        "Logistic model tree (depth 2)": lambda: LogisticModelTree(max_depth=2, min_leaf=0.1),
        "Random forest": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=50, n_jobs=-1, random_state=0),
        "XGBoost": lambda: XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                                         colsample_bytree=0.8, eval_metric="logloss", random_state=0, n_jobs=-1),
    }
    comp = []
    for name, make in MODELS.items():
        for label, cols_ in [("selected", SELECTED), ("all", ALL)]:
            r = cv(cols_, make)
            comp.append({"model": name, "covariates": label, "AUC": r["AUC"], "accuracy": r["accuracy"], "log loss": r["log loss"]})
    comp = pd.DataFrame(comp)
    print("GOAL 3 - model types (5-fold CV, grouped by map):")
    print(comp.to_string(index=False, float_format="%.4f"))
    comp.to_csv(f"{OUT}/v3_models.csv", index=False)
    print("  Higher AUC and accuracy are better; lower log loss is better.")

# =============================================================================
# 6. Chart of Goal 1
# =============================================================================
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = single.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#b8b7b1" if "*" in n else ("#2a78d6" if n in keep else "#eb6834")
              for n, g in zip(s["covariate added to equip_gap"], s["AUC gain"])]
    ax.barh(s["covariate added to equip_gap"], s["AUC gain"], color=colors)
    ax.axvline(0, color="#52514e", lw=1); 
    ax.set_xlabel("Held-out AUC gain over equip_gap alone (5-fold CV)")
    ax.set_title("Which covariates improve the CS2 win-probability model?", loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.01, 0.01, "Each bar: that covariate added alone. Blue = kept by forward selection, orange = dropped, grey = uses earlier results. "
             f"Baseline AUC {base['AUC']:.3f}.", fontsize=7.5, color="#52514e")
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(f"{OUT}/v3_covariate_gains.png", dpi=200)
    print(f"\nSaved {OUT}/v3_covariates.csv, {OUT}/v3_covariate_gains.png" + ("" if QUICK else f", {OUT}/v3_models.csv"))
except ImportError:
    pass