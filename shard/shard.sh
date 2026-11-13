#!/usr/bin/env bash
# One worker machine of the DR1 string scan.   usage:  bash shard.sh NUM_MACHINES MACHINE_INDEX [MAX_PATCHES_PER_PROCESS]
# Runs PROCS (default 2, one per core) dr1run.py processes, taking every (PROCS*M)-th patch: offsets K, K+M, K+2M, ... Patches already done on your own
# machine are skipped if you list their tags (one per line, e.g. P0003) in done_patches.txt.
set -euo pipefail
M=$1; K=$2; MAXP=${3:-1000000}; PROCS=${PROCS:-2}; SRC=$(cd "$(dirname "$0")" && pwd); W=/home/claude/strings
[ -f "$SRC/tilecenters_dr1.csv" ] || { echo "ERROR: tilecenters_dr1.csv missing next to shard.sh"; exit 1; }
sudo mkdir -p /home/claude 2>/dev/null || mkdir -p /home/claude; sudo chown "$(id -u)" /home/claude 2>/dev/null || true
mkdir -p $W/csrc $W/dr1; cp -r "$SRC"/mf/. $W/; cp "$SRC"/dr1pipe.py "$SRC"/ka_vet_all.py "$SRC"/tilecenters_dr1.csv $W/
cp "$SRC"/hough_fast.c $W/csrc/hough_fast.c
[ -f $W/detect.py ] || printf 'twin_pairs = None\nlines = None\n' > $W/detect.py   # import shim: detect.py was not in the zip
touch $W/tiles_dr1.csv                                    # dr1run only needs it to exist
gcc -O3 -fPIC -shared -o $W/csrc/libhough.so $W/csrc/hough_fast.c -lm
if [ -f "$SRC/done_patches.txt" ]; then while read -r t; do [ -n "$t" ] && mkdir -p $W/dr1/$t && echo '{}' > $W/dr1/$t/summary.json; done < "$SRC/done_patches.txt"; fi
cd $W; export OMP_NUM_THREADS=1
for i in $(seq 0 $((PROCS-1))); do
  python3 dr1run.py dr1.mer_catalogue_wide_survey $MAXP $((PROCS*M)) $((K+i*M)) > $W/run_$i.log 2>&1 &
done
wait; echo "shard $K finished: $(ls $W/dr1/P*/summary.json | wc -l) summaries"
