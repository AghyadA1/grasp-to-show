# Grasp-to-Show: data and trained models

Data and trained models behind the results in the README, so nothing has to be regenerated.
Extract either archive in the repository root (it creates `data/`).

**data-linux.tgz** (the run the README tables come from, Linux/GCC):
- `grasps/` grasp candidates + simulated lift labels (5,225 grasps, 40 objects)
- `feasibility/` checks 2–5 for every grasp
- `episodes/` full pick-and-show episodes: validation + policy comparison (160 scenarios)
- `models/` the two learned scorers (`scorer.pt`, `scorer_checks.pt`) and their metrics
- `episodes_learned/` the scorers as policies on the 10 unseen objects (80 scenarios)
- `heatmaps/` way-A heatmap scores for the 10 unseen objects

**data-windows.tgz** (the independent Windows/MSVC run):
- `grasps/`, `feasibility/`, `episodes/` as above (reproducibility table in the README)
- `dataset/` the 20,902-sample training set for the learned scorer (+ `clouds.npz`)
- `models/` copies of the trained scorers

Download: open this branch on GitHub, click each `.tgz`, then "Download raw file". Or: `git fetch origin data && git checkout origin/data -- data-linux.tgz`.
