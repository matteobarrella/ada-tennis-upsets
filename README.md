# Predicting upsets in ATP tennis matches

Semester project for Advanced Data Analytics (HEC Lausanne, autumn 2026).
Author: Matteo Barrella. Proposal approved by the teaching assistant on 1 November 2026.

**Status: work in progress.** Stages 1 to 3 are done (data, cleaning and features, models).
Figures, paper and recording are still to be added.

## Research question

Beyond the ranking gap, what predicts an upset (the lower-ranked player winning) in an ATP match:
surface, days of inactivity, tournament round and level, recent form, head-to-head and surface
record? Does a non-linear classifier capture interactions that a logistic regression misses?

## Data

ATP tour-level matches 2000-2024 (74,906 rows, 70,564 after cleaning), compiled by Jeff Sackmann.
The original GitHub repository was taken down in September 2026, so the files come from an
archival mirror. Source, commit and licence (CC BY-NC-SA 4.0) are in [`data/SOURCE.md`](data/SOURCE.md).

## Repository layout

```
README.md
requirements.txt
code/
  01_get_data.py      checks the data files, downloads missing ones from the pinned mirror commit
  02_clean.ipynb      cleaning, features built from past matches only, leakage checks
  03_models.ipynb     baseline, logistic regression, k-NN, gradient boosting, evaluation
  helpers.py          code shared by the notebooks
data/
  atp_matches_YYYY.csv    raw files, 2000-2024
  processed/              table written by 02_clean.ipynb
  SOURCE.md
results/                  tables written by 03_models.ipynb (metrics, CV results, predictions)
```

To be added: `04_figures.ipynb`, `figures/`, `paper.pdf`, `recording.mp4` (or a link).

## How to reproduce

From the repository root, in this order:

```
pip install -r requirements.txt
python code/01_get_data.py
cd code
jupyter nbconvert --to notebook --execute --inplace 02_clean.ipynb
jupyter nbconvert --to notebook --execute --inplace 03_models.ipynb
```

Each notebook runs top to bottom in a fresh kernel from the folder `code/`.
Running time: `02_clean.ipynb` under one minute, `03_models.ipynb` about 6 minutes.
Nothing beyond the course workspace (Nuvolos) is needed.

Note for Nuvolos: its notebook kernels start in `/files`, not in the notebook's folder. Run
`%cd` into the `code/` folder in a first cell (or start the notebooks from a terminal in
`code/`) so that the relative paths `../data` work.

## No look-ahead

Every player feature (days since last match, win rate over the last 10 matches, record on the
match surface, head-to-head, Elo rating overall and per surface) is computed only from matches that come earlier in a fixed
chronological order (tournament start date, round, match number). Train and test sets are split
by date. `helpers.check_no_lookahead` tests this in two ways: a brute-force recomputation on a
random sample of matches (the Elo ratings are compared with a second, independent implementation),
and a truncation test that rebuilds the features on data cut at a match. Both are run in `02_clean.ipynb`.

## Tools and attribution

* Libraries: pandas, NumPy, scikit-learn, matplotlib, Jupyter (used as intended, no attribution needed).
* Data: Jeff Sackmann, CC BY-NC-SA 4.0 (see above).
* Generative AI: Claude (Anthropic) was used to write the first version of `helpers.py`,
  `01_get_data.py`, `02_clean.ipynb` and `03_models.ipynb`, and to draft this README. The author reviewed the code.
  This list is the basis for the declaration in the paper's appendix and is kept up to date.
* Code from the course: none reused so far.
