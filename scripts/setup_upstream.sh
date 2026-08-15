#!/usr/bin/env bash
# Fetch SEA-RAFT into third_party/ and install the two extra dependencies.
#
# Nothing in this repository is a fork of SEA-RAFT: upstream is used unmodified,
# as a submodule, so the results cannot be explained away by "they changed the
# model". Every adaptation (the scale fix, the pre-clamp read-out) lives in
# ua_stop/ and is applied from the outside.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM="${ROOT}/third_party/SEA-RAFT"
REPO_URL="${SEA_RAFT_URL:-https://github.com/princeton-vl/SEA-RAFT.git}"

echo "repo root : ${ROOT}"
echo "upstream  : ${UPSTREAM}"

mkdir -p "${ROOT}/third_party" "${ROOT}/outputs/traces" "${ROOT}/outputs/tables" \
         "${ROOT}/outputs/figs" "${ROOT}/outputs/logs"

if [ -d "${UPSTREAM}/.git" ] || [ -f "${UPSTREAM}/custom.py" ]; then
  echo "upstream already present, skipping clone"
else
  echo "cloning ${REPO_URL} (depth 1) ..."
  git clone --depth 1 "${REPO_URL}" "${UPSTREAM}"
fi

# SEA-RAFT needs almost nothing beyond torch; einops is the only hard extra,
# huggingface_hub is what pulls the released checkpoint.
echo "installing python dependencies ..."
pip install -q einops "huggingface_hub>=0.20"

# Fail loudly here rather than deep inside a training loop.
for f in custom.py core/raft.py core/utils/utils.py config/parser.py; do
  if [ ! -f "${UPSTREAM}/${f}" ]; then
    echo "ERROR: expected ${UPSTREAM}/${f} -- upstream layout changed?" >&2
    exit 1
  fi
done

if [ ! -d "${UPSTREAM}/config/eval" ]; then
  echo "ERROR: ${UPSTREAM}/config/eval is missing -- cannot resolve model configs" >&2
  exit 1
fi

cat <<'EOF'

upstream ready.

next:
  python scripts/selftest.py            # 7 hard assertions, ~1 min on a T4
  python scripts/diagnose.py            # is the uncertainty signal alive?
  python scripts/run_latency.py         # T(n) = a + b n cost model
  python scripts/run_trace.py           # one full-budget pass per sample
  python scripts/run_sweep.py           # the experiment (CPU only)
  python scripts/run_calibrate.py       # RCPS risk certificate
  python scripts/make_figures.py        # fig1..fig4 as PDF

note: config/eval/spring-M.json ships scale=-1, and the checkpoint is trained
at 540x960. ua_stop applies that scale itself; selftest T4 fails loudly if it
is ever silently ignored.
EOF
