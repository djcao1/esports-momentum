# Esports momentum: starter setup

A first, minimal logistic regression on the CS2 round-economy dataset.

## 1. Install Python
Python 3.10 or newer. Check with `python --version` (or `python3 --version` on Mac).

## 2. Create a virtual environment and install packages
From inside this folder:

**Mac / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
You'll see `(.venv)` at the start of your terminal prompt when it's active.
Run the `activate` line again each time you open a new terminal.

## 3. Get the data
Download these three files from the Hugging Face dataset
**sneakyzero/cs2-pro-round-economy** (Files and versions tab) and put them in `data/`:

```
data/full_maps.parquet
data/full_rounds.parquet
data/full_team_round_economy.parquet
```
License: CC BY 4.0 — cite the dataset in anything you publish.

## 4. Run the model
```bash
python basic_logit.py data
```

## What it does
One row per round, from the CT side's point of view:

- **y** = 1 if CT won the round, 0 if T won
- **x** = equipment gap = CT equipment value − T equipment value (thousands of $, after buying)

Model: `P(CT wins) = 1 / (1 + exp(-(b0 + b1 * equip_gap)))`

It trains on 80% of maps and tests on the other 20%.

## Reading the output
- **const (b0)**: CT's log-odds of winning when both teams have equal equipment.
  Above 0 means CT is favored at an even buy.
- **equip_gap (b1)**: change in log-odds per extra $1k of equipment advantage.
  `exp(b1)` = how much each $1k multiplies the odds of winning.
- **P>|z|**: p-value; below 0.05 means the effect is unlikely to be chance.
- **Accuracy / AUC** on held-out maps: how well it predicts rounds it never saw.
  AUC 0.5 = coin flip, 1.0 = perfect.

## Next steps (later)
Add more inputs one at a time (buy matchup, map, pistol rounds, team skill),
then recent-round results for the momentum test.
