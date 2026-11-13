"""DR1 orchestrator. Usage: python3 dr1run.py TABLE [max_patches]
1) tile list (GROUP BY tileid) -> tiles_dr1.csv ; 2) tile centres -> tilecenters_dr1.csv ; 3) 4x4 deg patches on a 3.5 deg grid
4) for each patch not yet done: dr1pipe (bright), delete raw, record summary; every 10th patch also a scrambled copy for calibration.
State lives in dr1/ so the run can be resumed after interruption."""
import sys, os, subprocess, json, glob, numpy as np, shutil
TABLE = sys.argv[1]; MAXP = int(sys.argv[2]) if len(sys.argv) > 2 else 10 ** 6
STRIDE, OFF = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (1, 0)
os.chdir("/home/claude/strings"); env = dict(os.environ, OMP_NUM_THREADS="1", TILECENTERS="tilecenters_dr1.csv")
if not os.path.exists("tiles_dr1.csv"):
    subprocess.run(["curl", "-s", "--max-time", "3600", "https://irsa.ipac.caltech.edu/TAP/sync", "--data-urlencode",
                    f"QUERY=SELECT tileid, COUNT(*) AS n FROM {TABLE} GROUP BY tileid", "--data-urlencode", "FORMAT=csv", "-o", "tiles_dr1.csv"])
#subprocess.run(["python3", "tilecenters.py", TABLE, "tiles_dr1.csv", "tilecenters_dr1.csv"])
tc = np.array([[float(v) for v in l.split(",")[1:]] for l in open("tilecenters_dr1.csv").read().split()])
print("DR1 tiles with centres:", len(tc), flush=True)
patches = set()
for ra, de in tc:   # snap each tile to a patch centre on a 3.5 deg grid (dec rows, RA step scaled by cos dec)
    dj = round(de / 3.5) * 3.5; step = 3.5 / max(np.cos(np.radians(dj)), 0.1); ri = (round(ra / step) * step) % 360
    patches.add((round(ri, 3), round(dj, 3)))
# priority order: patches nearest the Q1 fields first (lets Q1 candidate lines be extended), then the large contiguous southern block
Q1C = [(52.9, -28.1), (61.2, -48.4), (269.7, 66.0)]
def sep(a, b, c, d): a, b, c, d = map(np.radians, (a, b, c, d)); return np.degrees(np.arccos(np.clip(np.sin(b)*np.sin(d)+np.cos(b)*np.cos(d)*np.cos(a-c), -1, 1)))
patches = sorted(patches, key=lambda p: (min(sep(p[0], p[1], *q) for q in Q1C) > 8, p[1] > 0, min(sep(p[0], p[1], *q) for q in Q1C)))
print("patches:", len(patches), "first:", patches[:5], flush=True)
done = 0
for k, (ra, de) in enumerate(patches):
    tag = f"P{k:04d}"
    if k % STRIDE != OFF: continue
    if os.path.exists(f"dr1/{tag}/summary.json"): continue
    if done >= MAXP: break
    subprocess.run(["python3", "dr1pipe.py", TABLE, str(ra), str(de), "4", tag], env=env)
    if k % 10 == 0 and os.path.exists(f"dr1/{tag}/summary.json"):
        subprocess.run(["python3", "dr1pipe.py", TABLE, str(ra), str(de), "4", tag], env=dict(env, SCRAMBLE=str(k)))
    s = json.load(open(f"dr1/{tag}/summary.json")) if os.path.exists(f"dr1/{tag}/summary.json") else None
    keep = s and s["max_S"] >= 6.8   # lower discovery bar; every kept candidate must then pass extend.py (validated: real strings give S~9-12 over 6 deg)
    if not keep: shutil.rmtree(f"dr1/{tag}/raw", ignore_errors=True)    # stream: keep raw data only for candidate patches
    print(tag, ra, de, "max_S", s and round(s["max_S"], 2), "KEEP for vetting" if keep else "", flush=True); done += 1
