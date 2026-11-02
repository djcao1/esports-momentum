"""Version 2: build the CS2 win-probability model up step by step, then a first momentum check.

Each model predicts P(CT wins the round). Same data and same held-out maps as basic_logit.py.

  Model A  equip_gap                         (the basic model)
  Model B  + map                             (CT is stronger on some maps than others)
  Model C  + buy matchup                     (eco / semi-eco / semi-buy / full buy, both teams)
  Model D  ct_won_prev only                  (raw "momentum": did CT win the last round?)
  Model E  Model C + ct_won_prev             (momentum after accounting for economy and map)

Compare D and E: if ct_won_prev's effect shrinks a lot once economy is in the model,
most of the apparent momentum is really economy carrying over.

Run:  python logit_v2.py data
"""
import sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score

DATA = sys.argv[1] if len(sys.argv) > 1 else "data"

# 1. Load and build one row per round (CT's point of view)
maps = pd.read_parquet(f"{DATA}/full_maps.parquet")
rounds = pd.read_parquet(f"{DATA}/full_rounds.parquet")
econ = pd.read_parquet(f"{DATA}/full_team_round_economy.parquet")

good = maps.loc[maps["analysis_eligible_v1"], "map_uid"]
rounds = rounds[rounds["map_uid"].isin(good)]
equip = econ.pivot_table(index=["map_uid", "round_num"], columns="side",
                         values="team_current_equip_value").reset_index()
df = rounds[["map_uid", "round_num", "map_name", "winner_side"]].merge(equip, on=["map_uid", "round_num"])
df = df.dropna(subset=["ct", "t"]).sort_values(["map_uid", "round_num"])
df["ct_win"] = (df["winner_side"] == "ct").astype(int)
df["equip_gap"] = (df["ct"] - df["t"]) / 1000
df["map_name"] = df["map_name"].str.replace("de_", "").str.title()

# Buy tier per side (same cut-offs used in the win-rate charts)
tiers = [-1, 4999, 9999, 19999, 1e9]
names = ["Eco", "SemiEco", "SemiBuy", "Full"]
df["ct_buy"] = pd.cut(df["ct"], tiers, labels=names).astype(str)
df["t_buy"] = pd.cut(df["t"], tiers, labels=names).astype(str)
df["buy_matchup"] = df["ct_buy"] + "_v_" + df["t_buy"]

# Did CT win the previous round? Only within a half of regulation, where the CT team
# is the same team as last round (sides swap after round 12).
df["ct_won_prev"] = df.groupby("map_uid")["ct_win"].shift(1)
same_half = df["round_num"].between(2, 12) | df["round_num"].between(14, 24)
df = df[same_half & df["ct_won_prev"].notna()].copy()   # also drops pistol rounds 1 and 13
# Matchups with fewer than 100 rounds (e.g. eco vs. eco) are too rare to estimate; group them
counts = df["buy_matchup"].value_counts()
df["buy_matchup"] = df["buy_matchup"].where(df["buy_matchup"].map(counts) >= 100, "Rare")
print(f"{len(df):,} non-pistol regulation rounds from {df['map_uid'].nunique():,} maps\n")

# 2. Same train/test split by map as before
tr, te = next(GroupShuffleSplit(test_size=0.2, random_state=0).split(df, groups=df["map_uid"]))
train, test = df.iloc[tr], df.iloc[te]

MODELS = {
    "A  equip_gap":                    "ct_win ~ equip_gap",
    "B  + map":                        "ct_win ~ equip_gap + C(map_name)",
    "C  + buy matchup":                "ct_win ~ equip_gap + C(map_name) + C(buy_matchup)",
    "D  ct_won_prev only":             "ct_win ~ ct_won_prev",
    "E  C + ct_won_prev":              "ct_win ~ equip_gap + C(map_name) + C(buy_matchup) + ct_won_prev",
}

# 3. Fit each model; errors clustered by map 
rows, fits = [], {}
for name, formula in MODELS.items():
    m = smf.logit(formula, data=train).fit(disp=0, maxiter=200, cov_type="cluster",
                                           cov_kwds={"groups": pd.factorize(train["map_uid"])[0]})
    fits[name] = m
    p = m.predict(test)
    rows.append({"model": name, "held-out AUC": round(roc_auc_score(test["ct_win"], p), 3),
                 "accuracy": round(((p > 0.5) == test["ct_win"]).mean(), 3),
                 "pseudo R2": round(m.prsquared, 3)})
print(pd.DataFrame(rows).to_string(index=False))

# 4. Momentum check: ct_won_prev with and without economy
print("\nEffect of CT winning the previous round:")
for name in ["D  ct_won_prev only", "E  C + ct_won_prev"]:
    m = fits[name]
    b, se = m.params["ct_won_prev"], m.bse["ct_won_prev"]
    print(f"  {name:22s} coef {b:+.3f}  odds ratio {np.exp(b):.2f} "
          f"(95% CI {np.exp(b - 1.96 * se):.2f}-{np.exp(b + 1.96 * se):.2f})  p = {m.pvalues['ct_won_prev']:.3g}")

# Same thing as plain win rates: actual vs. what Model C (no momentum term) expects
test = test.assign(expected=fits["C  + buy matchup"].predict(test))
t = test.groupby("ct_won_prev").agg(actual=("ct_win", "mean"), expected=("expected", "mean"), rounds=("ct_win", "size"))
t.index = ["CT lost last round", "CT won last round"]
print("\nHeld-out maps, CT win rate this round:")
print(t.round(3).to_string())