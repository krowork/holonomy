#!/usr/bin/env bash
# Ejecuta todas las comprobaciones y regenera datos y figuras.
# Uso: ./run_all.sh [--rapido]   (--rapido omite los guiones de más de ~5 min)
set -u
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}

FAST=(verify_sym verify_step3 verify_unique tri witness_check xxz sec ba verify_holonomia)
SLOW=(table62 edge phistar primalU1 rank witness_n chk baedge2)

scripts=("${FAST[@]}")
[[ "${1:-}" == "--rapido" ]] || scripts+=("${SLOW[@]}")

failed=()
for s in "${scripts[@]}"; do
    echo "=== $s"
    start=$SECONDS
    if ! "$PY" "scripts/$s.py"; then failed+=("$s"); fi
    echo "--- $s: $((SECONDS - start)) s"
done
"$PY" scripts/bafit.py
"$PY" scripts/figures.py

if ((${#failed[@]})); then
    echo "FALLAN: ${failed[*]}"
    exit 1
fi
echo "Todas las comprobaciones correctas."
