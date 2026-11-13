"""Vet top candidate lines from a full scan: cluster cells into distinct lines, then kill tests."""
import numpy as np, sys, pickle
from blind import prep
from weights import weights
name, dirs, scanfile = sys.argv[1], tuple(sys.argv[2].split(",")), sys.argv[3]; K = int(sys.argv[4]) if len(sys.argv) > 4 else 6
R = pickle.load(open(scanfile, "rb")); top = R["top"]
lines = []
for S, th, rho, w in top:
    if S < -1e8: continue
    if any(abs(th - l[1]) < 1e-3 and abs(rho - l[2]) < 10 for l in lines): continue
    lines.append((S, th, rho, int(w)))
print(f"{name}: data max S {R['S'].max():.2f}; distinct candidate lines (S, theta deg, rho, w):", [(round(l[0], 2), round(np.degrees(l[1]), 3), round(l[2], 1), l[3]) for l in lines[:K]], flush=True)
d, x, y, c1, c2, sig = prep(name, dirs); wt, var, resp = weights(d, c1, c2); faint = d.m.values > 25.5
try: P = pickle.load(open(f"pairs_{name}_deep.pkl", "rb"))["F"]
except Exception: P = None
def Sstat(m, th):
    cp = -(c1[m] * np.cos(2 * th) + c2[m] * np.sin(2 * th)); return (wt[m] * cp).sum() / np.sqrt((wt[m] ** 2 * var[m]).sum() + 1e-30)
for S0, th, rho, w in lines[:K]:
    n = np.array([-np.sin(th), np.cos(th)]); t = np.array([np.cos(th), np.sin(th)])
    dist = x * n[0] + y * n[1] - rho; along = x * t[0] + y * t[1]; hw = w * 0.1 + 0.05
    m = np.abs(dist) < hw; tm = np.median(along[m])
    prof = [Sstat(np.abs(dist - o) < hw, th) for o in np.arange(-2.0, 2.01, 0.4)]
    ctrl = (np.abs(dist) > 1.0) & (np.abs(dist) < 3.0)
    dmag = np.median(d.m.values[m]) - np.median(d.m.values[ctrl])
    tile_ang = min(abs(((np.degrees(th) + 45) % 90) - 45), 45)
    tw = 0
    if P is not None:
        pd_ = np.abs(P[:, 0] * n[0] + P[:, 1] * n[1] - rho); tw = int(((pd_ < 1.0) & (np.abs(((P[:, 2] - np.degrees(th) + 90) % 180) - 90) < 10)).sum())
    print(f"\nLINE S={S0:.2f} theta={np.degrees(th):.3f}deg rho={rho:.1f} w={hw:.2f}\" N={m.sum()} length~{np.ptp(along[m]):.0f}\"")
    print(f"  bright-only S {Sstat(m & ~faint, th):.2f} | faint-only S {Sstat(m & faint, th):.2f}")
    print(f"  half A S {Sstat(m & (along < tm), th):.2f} | half B S {Sstat(m & (along >= tm), th):.2f}")
    print(f"  offset profile (-2..2\" step 0.4): {np.round(prof, 1).tolist()}")
    print(f"  median mag strip - control: {dmag:+.3f} (merged twins predict negative)")
    print(f"  angle from tile axes: {tile_ang:.2f} deg | resolved twin pairs on line: {tw}")
