"""Streaming patch pipeline for Euclid DR1 (rehearsed on Q1).
  python3 dr1pipe.py TABLE PATCH_RA PATCH_DEC SIZE_DEG TAG [faint]
For one patch: find tiles (TAP), download only needed columns per tile (parallel curl), convert to compact npy, delete CSV,
local-mean shape subtraction, optimal weights, weighted blind C scan, vet top lines, save summary JSON."""
import numpy as np, sys, os, json, subprocess, time, glob, pickle
from concurrent.futures import ThreadPoolExecutor
TABLE, RA0, DEC0, SIZE, TAG = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
FAINT = len(sys.argv) > 6 and sys.argv[6] == "faint"
WD = f"/home/claude/strings/dr1/{TAG}"; os.makedirs(WD + "/raw", exist_ok=True)
COLS = "object_id,right_ascension,declination,flux_vis_2fwhm_aper,fluxerr_vis_2fwhm_aper,semimajor_axis,ellipticity,position_angle"
FCUT = "flux_vis_2fwhm_aper>0.04" if FAINT else "flux_vis_2fwhm_aper>0.23"
def tap(q, out):
    r = subprocess.run(["curl", "-s", "--max-time", "1200", "https://eas.esac.esa.int/tap-server/tap/sync", "--data-urlencode", "REQUEST=doQuery", "--data-urlencode", "LANG=ADQL", "--data-urlencode", "QUERY=" + q,
                        "--data-urlencode", "FORMAT=csv", "-o", out], capture_output=True); return os.path.exists(out) and os.path.getsize(out) > 0
t0 = time.time(); h = SIZE / 2; cosd = np.cos(np.radians(DEC0))
box = f"ra BETWEEN {RA0 - h / cosd} AND {RA0 + h / cosd} AND dec BETWEEN {DEC0 - h} AND {DEC0 + h}"
TC = os.environ.get("TILECENTERS", "tilecenters_q1.csv")
tc = [l.split(",") for l in open(TC).read().split()]
subs = [t for t, tr, td in tc if abs(float(td) - DEC0) < h + 0.35 and abs((float(tr) - RA0 + 180) % 360 - 180) * cosd < h + 0.35]
print(f"[{TAG}] {len(subs)} tiles in patch", flush=True)
def get(t):
    out = f"{WD}/raw/t{t}.csv"
    if os.path.exists(out + ".npy"): return t
    q = f"SELECT {COLS} FROM {TABLE} WHERE tile_index={t} AND {FCUT} AND spurious_flag=0 AND point_like_prob<0.5"
    for attempt in range(3):
        if tap(q, out): break
    try:
        a = np.genfromtxt(out, delimiter=",", skip_header=1, usecols=range(1, 8), dtype=np.float64)
        a = a.reshape(-1, 7); a = a[np.all(np.isfinite(a[:, :5]), 1)]
        a = a[(np.abs(a[:, 1] - DEC0) < h) & (np.abs((a[:, 0] - RA0 + 180) % 360 - 180) * cosd < h)]
        np.save(out + ".npy", a); os.remove(out)
    except Exception as e: print("tile", t, "failed", e, flush=True)
    return t
with ThreadPoolExecutor(6) as ex: list(ex.map(get, subs))
A = np.concatenate([np.load(f) for f in glob.glob(f"{WD}/raw/*.npy")]); u, ui = np.unique(A[:, :2].round(7), axis=0, return_index=True); A = A[np.sort(ui)] if glob.glob(f"{WD}/raw/*.npy") else np.zeros((0, 7))
print(f"[{TAG}] downloaded {len(A)} galaxies in {(time.time() - t0) / 60:.1f} min", flush=True)
if len(A) < 1000: sys.exit(0)
ra, dec, flux, ferr, a, e, pa = A.T
# gnomonic projection about patch centre
r, d, r0, d0 = map(np.radians, (ra, dec, RA0, DEC0)); c = np.sin(d0) * np.sin(d) + np.cos(d0) * np.cos(d) * np.cos(r - r0)
x = np.degrees(np.cos(d) * np.sin(r - r0) / c) * 3600; y = np.degrees((np.cos(d0) * np.sin(d) - np.sin(d0) * np.cos(d) * np.cos(r - r0)) / c) * 3600
m = (flux > 0) & (a > 0); mag = -2.5 * np.log10(np.where(flux > 0, flux, 1)) + 23.9; m &= (mag > 20) & (mag < 27)
x, y, flux, ferr, a, e, pa, mag = [v[m] for v in (x, y, flux, ferr, a, e, pa, mag)]
import pandas as pd
D = pd.DataFrame(dict(x=x, y=y, m=mag, flux_vis_2fwhm_aper=flux, fluxerr_vis_2fwhm_aper=np.where(np.isfinite(ferr), ferr, flux * 0.1),
                      semimajor_axis=a, ellipticity=np.clip(np.nan_to_num(e), 0, 0.95), position_angle=np.nan_to_num(pa)))
from blind import scan, CELL
from weights import weights
aa = D.semimajor_axis.values * 0.1; ee = D.ellipticity.values; chi = (1 - (1 - ee) ** 2) / (1 + (1 - ee) ** 2); ang = np.radians(90 - D.position_angle.values)
c1 = chi * np.cos(2 * ang); c2 = chi * np.sin(2 * ang)
key = np.floor(x / CELL).astype(np.int64) * 1000003 + np.floor(y / CELL).astype(np.int64); _, inv = np.unique(key, return_inverse=True); inv = inv.ravel(); nn = np.bincount(inv)
c1 = c1 - (np.bincount(inv, c1) / nn)[inv]; c2 = c2 - (np.bincount(inv, c2) / nn)[inv]; sig = float(np.sqrt(np.mean(c1 ** 2)))
w, var, _ = weights(D, c1, c2)
X = np.ascontiguousarray(x); Y = np.ascontiguousarray(y)
dt = float(os.environ.get("DT_SCALE", 0.30)) / (np.max(np.hypot(X, Y)) + 1)   # was 0.15; 0.30 matches pipeinj.py (2.9e-5 rad), recovery ~unchanged in my 5-injection test
mode = os.environ.get("SCRAMBLE")
if mode:
    rng = np.random.default_rng(int(mode)); ph = rng.uniform(0, 2 * np.pi, len(c1)); c1, c2 = c1 * np.cos(ph) - c2 * np.sin(ph), c1 * np.sin(ph) + c2 * np.cos(ph)
t1 = time.time(); R = scan(X, Y, c1, c2, sig, 0.0, np.pi, dt, wt=w, var=var)
lines = []
for S, th, rho, ww in R["top"]:
    if S < -1e8: continue
    if any(abs(th - l[1]) < 1e-3 and abs(rho - l[2]) < 10 for l in lines): continue
    lines.append((float(S), float(th), float(rho), int(ww)))
out = dict(tag=TAG, table=TABLE, ra0=RA0, dec0=DEC0, size=SIZE, faint=FAINT, scramble=mode, N=int(len(X)), dt=dt, scan_min=(time.time() - t1) / 60,
           max_S=float(R["S"].max()), lines=lines[:20], angle_profile=[float(R["S"][(np.degrees(R["theta"]) >= b) & (np.degrees(R["theta"]) < b + 30)].mean()) for b in range(0, 180, 30)])
json.dump(out, open(f"{WD}/summary{'_scr' + mode if mode else ''}.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "lines"}), "top lines", lines[:3], flush=True)
