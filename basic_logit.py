"""The most basic CS2 round model: does the CT side's equipment advantage predict who wins the round?

    P(CT wins round) = 1 / (1 + exp(-(b0 + b1 * equip_gap)))

equip_gap = CT team's equipment value minus T team's, in thousands of $, after buying.
One row per round (from the CT side's point of view), so no round is counted twice.

Run:  python basic_logit.py data
"""
import sys
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score

DATA = sys.argv[1] if len(sys.argv) > 1 else "data"

# 1. Load the three files from the Hugging Face dataset
maps = pd.read_parquet(f"{DATA}/full_maps.parquet")
rounds = pd.read_parquet(f"{DATA}/full_rounds.parquet")
econ = pd.read_parquet(f"{DATA}/full_team_round_economy.parquet")

# 2. Keep only maps the dataset marks as clean enough for analysis
good = maps.loc[maps["analysis_eligible_v1"], "map_uid"]
rounds = rounds[rounds["map_uid"].isin(good)]

# 3. One row per round: CT equipment, T equipment, and whether CT won
equip = econ.pivot_table(index=["map_uid", "round_num"], columns="side",
                         values="team_current_equip_value").reset_index()
df = rounds[["map_uid", "round_num", "winner_side"]].merge(equip, on=["map_uid", "round_num"])
df = df.dropna(subset=["ct", "t"])
df["ct_win"] = (df["winner_side"] == "ct").astype(int)
df["equip_gap"] = (df["ct"] - df["t"]) / 1000
print(f"{len(df):,} rounds from {df['map_uid'].nunique():,} maps\n")

# 4. Hold out 20% of maps for testing (whole maps, so test rounds are truly unseen)
train_idx, test_idx = next(GroupShuffleSplit(test_size=0.2, random_state=0)
                           .split(df, groups=df["map_uid"]))
train, test = df.iloc[train_idx], df.iloc[test_idx]

# 5. Fit the logistic regression on the training maps
model = sm.Logit(train["ct_win"], sm.add_constant(train[["equip_gap"]])).fit(disp=0)
print(model.summary())

# 6. Check it on the held-out maps
p = model.predict(sm.add_constant(test[["equip_gap"]]))
print(f"\nHeld-out maps: accuracy {accuracy_score(test['ct_win'], p > 0.5):.3f}, "
      f"AUC {roc_auc_score(test['ct_win'], p):.3f}")
print(f"Baseline (always guess CT wins): accuracy {test['ct_win'].mean():.3f}")
