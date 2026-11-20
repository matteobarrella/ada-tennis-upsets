# Data source

ATP tour-level match files `atp_matches_2000.csv` to `atp_matches_2024.csv`, compiled by
Jeff Sackmann (Tennis Abstract, https://www.tennisabstract.com).

* Original repository: `JeffSackmann/tennis_atp` on GitHub. It was taken down by its owner in
  September 2026 and is no longer available.
* Copy used here: the archival mirror `Aneeshers/tennis-sackmann-archive`, folder `atp/`,
  pinned to commit `83733587353df8a41f2fd4f516147d5aa83f5a8d`. The mirror keeps Sackmann's
  original README as provenance (`UPSTREAM_README.md`).
* Licence: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
  (CC BY-NC-SA 4.0). The files in this folder, and the table derived from them in
  `processed/`, are redistributed under the same licence. Credit: Jeff Sackmann.
* `code/01_get_data.py` re-downloads any missing file from the pinned commit.

`processed/matches_model.csv.gz` is produced by `code/02_clean.ipynb`.
