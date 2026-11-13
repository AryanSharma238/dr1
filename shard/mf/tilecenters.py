"""Tile centres for a MER catalogue table: one cheap TOP-1 query per tile, in parallel. Usage: tilecenters.py TABLE tiles.csv out.csv"""
import sys, subprocess, csv, os
from concurrent.futures import ThreadPoolExecutor
TABLE, TL, OUT = sys.argv[1:4]
tiles = [l.split(",")[0] for l in open(TL).read().split()[1:]]
done = {r[0] for r in csv.reader(open(OUT))} if os.path.exists(OUT) else set()
def one(t):
    if t in done: return None
    r = subprocess.run(["curl", "-s", "--max-time", "120", "https://irsa.ipac.caltech.edu/TAP/sync", "--data-urlencode",
                        f"QUERY=SELECT TOP 50 ra,dec FROM {TABLE} WHERE tileid={t}", "--data-urlencode", "FORMAT=csv"], capture_output=True, text=True).stdout.split()[1:]
    if not r: return None
    ra = [float(v.split(",")[0]) for v in r]; de = [float(v.split(",")[1]) for v in r]
    import numpy as np; return t, float(np.median(ra)), float(np.median(de))
with ThreadPoolExecutor(8) as ex:
    with open(OUT, "a") as f:
        for res in ex.map(one, tiles):
            if res: f.write("%s,%.5f,%.5f\n" % res); f.flush()
