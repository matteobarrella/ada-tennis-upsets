# Predicting upsets in ATP tennis matches

Semester project for Advanced Data Analytics (HEC Lausanne, autumn 2026).
Author: Matteo Barrella. Proposal approved by the teaching assistant on 1 November 2026.

**Status: work in progress.** Stages 1 and 2 are done. Modelling, figures, paper and recording
are still to be added.

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
  helpers.py          code shared by the notebooks
data/
  atp_matches_YYYY.csv    raw files, 2000-2024
  processed/              table written by 02_clean.ipynb
  SOURCE.md
```

To be added: `03_model.ipynb`, `04_figures.ipynb`, `figures/`, `paper.pdf`, `recording.mp4` (or a link).

## How to reproduce

From the repository root, in this order:

```
pip install -r requirements.txt
python code/01_get_data.py
cd code
jupyter nbconvert --to notebook --execute --inplace 02_clean.ipynb
```

Each notebook runs top to bottom in a fresh kernel from the folder `code/`.
Running time of `02_clean.ipynb`: under one minute. Nothing beyond the course workspace
(Nuvolos) is needed.

## No look-ahead

Every player feature (days since last match, win rate over the last 10 matches, record on the
match surface, head-to-head) is computed only from matches that come earlier in a fixed
chronological order (tournament start date, round, match number). Train and test sets are split
by date. `helpers.check_no_lookahead` tests this in two ways: a brute-force recomputation on a
random sample of matches, and a truncation test that rebuilds the features on data cut at a
match. Both are run in `02_clean.ipynb`.

## Tools and attribution

* Libraries: pandas, NumPy, scikit-learn, matplotlib, Jupyter (used as intended, no attribution needed).
* Data: Jeff Sackmann, CC BY-NC-SA 4.0 (see above).
* Generative AI: Claude (Anthropic) was used to write the first version of `helpers.py`,
  `01_get_data.py` and `02_clean.ipynb`, and to draft this README. The author reviewed the code.
  This list is the basis for the declaration in the paper's appendix and is kept up to date.
* Code from the course: none reused so far.
