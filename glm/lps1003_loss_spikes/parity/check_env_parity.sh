#!/usr/bin/env bash
# Diff a LIVE devbox trainer's environment/libs against the canonical prod
# fingerprint. Run on the devbox node that hosts rank 0 while a trainer runs.
# usage: check_env_parity.sh [prod_fingerprint.json]
set -euo pipefail
PAR=/root/.cache/user_artifacts/lps1003/parity
REF="${1:-$PAR/fingerprint_prod_reference.json}"
[ -f "$REF" ] || { echo "prod reference fingerprint not found: $REF" >&2; exit 2; }
TMP=$(mktemp -d)
V=/root/.cache/user_artifacts/trainers_main/server/.venv/bin/python
python3 "$PAR/fingerprint.py" --python "$V" --no-hash --out "$TMP/fp_devbox.json"
python3 "$PAR/fingerprint_diff.py" "$REF" "$TMP/fp_devbox.json" --labels PROD DEVBOX
echo
echo "(env DIFF lines above are the go/no-go list; .so sha diffs suppressed by --no-hash)"
