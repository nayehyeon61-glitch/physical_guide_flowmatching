#!/usr/bin/env bash
# ERA5 2.8125 deg (WB2 128x64) for blend_sr S1: kai6h + scoring files in data/era5_2p8125, idle priority, resumes
# (finished years are skipped). WB2 has no skt or tisr: skt := t2m, tisr computed. WB2 ends 2021-12-31.
#   PROCS=4 YEARS="2016 2017" bash scripts/fetch_wb2_2p8.sh
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-data/era5_2p8125}; PROCS=${PROCS:-4}; YEARS=${YEARS:-"2016 2017 2018 2020 $(seq -s ' ' 2015 -1 1979)"}
export DASK_NUM_WORKERS=${DASK_NUM_WORKERS:-4} OMP_NUM_THREADS=2
[ -z "${SSL_CERT_FILE:-}" ] && [ -e /etc/pki/tls/certs/ca-bundle.crt ] && export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
PY=${PY:-python}
idle() { chrt -i 0 nice -n 19 ionice -c3 "$@"; }
one() {  # one <year>: up to 5 tries with a pause
    local y=$1 t
    for t in 1 2 3 4 5; do
        idle "$PY" scripts/wb2_to_kai.py --out "$OUT" --all-hours --grid 128x64 --years "$y" 2>&1 | grep -E 'done$|Error' | tail -2 | cut -c1-200
        [ "${PIPESTATUS[0]}" = 0 ] && return 0
        echo "$y: try $t failed $(date '+%H:%M')"; rm -f "$OUT/kai6h/$y.nc.tmp"; sleep 60
    done
    echo "$y FAILED"; return 1
}
set -- $YEARS
one "$1" || exit 1   # the first year also writes the constants, before anything runs in parallel
shift
fail=0
for y in "$@"; do
    while [ "$(jobs -rp | wc -l)" -ge "$PROCS" ]; do wait -n || fail=1; done
    one "$y" &
done
while [ "$(jobs -rp | wc -l)" -gt 0 ]; do wait -n || fail=1; done
echo "FETCH_EXIT=$fail $(date '+%m-%d %H:%M')  $(ls "$OUT"/kai6h/*.nc | wc -l) years"
exit $fail
