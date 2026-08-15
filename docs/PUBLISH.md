# Publishing this repository on GitHub

The zip you downloaded is a complete, self-contained git-ready tree. It does
**not** contain SEA-RAFT itself: upstream is added as a submodule so your
history stays clean and the BSD-3 licensed code is never re-distributed here.

## 1. Create the repository

```bash
unzip ua-searaft-v0.1.0.zip
cd ua-searaft

git init
git add .
git commit -m "UA-SEA-RAFT: uncertainty-aware adaptive stopping for SEA-RAFT"
```

Then create an empty repo named `ua-searaft` on GitHub (no README, no license —
both already exist here) and push:

```bash
git branch -M main
git remote add origin https://github.com/USERNAME/ua-searaft.git
git push -u origin main
```

Replace `USERNAME` in `pyproject.toml` and in the README clone command too.

## 2. Add upstream as a submodule

```bash
git submodule add --depth 1 https://github.com/princeton-vl/SEA-RAFT.git third_party/SEA-RAFT
git commit -m "Add SEA-RAFT as a submodule (unmodified upstream)"
git push
```

People cloning your repo then run:

```bash
git clone --recurse-submodules https://github.com/USERNAME/ua-searaft.git
```

If you prefer not to use submodules, `scripts/setup_upstream.sh` clones upstream
on demand and is what the README documents — both paths work.

## 3. What must never be committed

`.gitignore` already covers these, but check `git status` before the first push:

- `outputs/` artefacts (`*.npz` traces, `*.json` tables, `*.pdf` figures) — keep
  the directory, not its contents. Publish figures via a release instead.
- Checkpoints (`*.pth`, `*.ckpt`) — they are downloaded from Hugging Face.
- Datasets (`data_scene_flow.zip`, `kitti15/`, Sintel) — never in git.
- `third_party/SEA-RAFT` as plain files. If you ever committed it by accident:
  `git rm -r --cached third_party/SEA-RAFT`.

GitHub rejects files above 100 MB, and a committed KITTI zip is ~2 GB.

## 4. Make it reproducible for a reader

```bash
python -m pytest tests -q            # numpy-only tests need no GPU
python scripts/selftest.py           # 7 hard assertions, needs the model
```

Commit a filled-in `docs/RESULTS.md` together with the `outputs/tables/*.json`
that produced it (they are small) so the numbers in your report are traceable.
Every JSON already carries a provenance block: `ua_stop` version, git hash of
this repo **and** of upstream, UTC timestamp, torch/CUDA/GPU.

## 5. Tag a release

```bash
git tag -a v0.1.0 -m "First public release: global uncertainty-saturation stopping"
git push origin v0.1.0
```

Attach `outputs/figs/*.pdf` and the trace `.npz` to the GitHub release page —
that is the right home for binary artefacts, not the git history.

## 6. Repository description and topics

Suggested one-liner:

> Training-free uncertainty-aware adaptive stopping for SEA-RAFT optical flow —
> replay every threshold offline from a single cached forward pass.

Topics: `optical-flow`, `sea-raft`, `raft`, `early-exit`, `adaptive-computation`,
`uncertainty-estimation`, `conformal-prediction`, `pytorch`.

## 7. Before you claim a speed-up in the README

1. `scripts/selftest.py` passes 7/7 on a real pair (not noise).
2. T4 confirms `scale` was applied — otherwise every latency number is 4× off.
3. The headline is stated against a **cost-matched fixed budget**, with a CI.
4. `d_only` is in the table.
5. Tile-level numbers are labelled *modelled*, not measured.

Items 2 and 5 are the two mistakes that already happened once in this project.
