"""Resolved-twin-pair cosmic-string search for one patch (complements the merged-twin Hough scan; sensitive at Gmu >~ 3e-7, tolerant of wiggles).
Called from dr1pipe.py after download. Writes twin.json: top (theta,rho) bins, their pair lists, and a scrambled-orientation control max."""
import numpy as np, json
from scipy.spatial import cKDTree
RMIN, RMAX, DTH, DRHO = 0.35, 4.0, np.radians(1.0), 3.0
def pair_lines(X, Y, F, FE, AA):
    P = cKDTree(np.c_[X, Y]).query_pairs(RMAX, output_type="ndarray"); i, j = P[:, 0], P[:, 1]
    dx, dy = X[j]-X[i], Y[j]-Y[i]; s = np.hypot(dx, dy)
    ok = (s > RMIN)&(np.abs(F[i]-F[j]) < 2.5*np.hypot(FE[i], FE[j]))&(np.abs(np.log(AA[i]/AA[j])) < 0.3)
    i, j, dx, dy = i[ok], j[ok], dx[ok], dy[ok]
    th = (np.arctan2(dy, dx)-np.pi/2) % np.pi; mx, my = (X[i]+X[j])/2, (Y[i]+Y[j])/2
    return i, j, th, -mx*np.sin(th)+my*np.cos(th)
def grid(th, rho, R):
    H, _, _ = np.histogram2d(th, rho, bins=[np.arange(0, np.pi+DTH, DTH), np.arange(-R, R+DRHO, DRHO)])
    return H[:-1, :-1]+H[1:, :-1]+H[:-1, 1:]+H[1:, 1:]
def run(X, Y, F, FE, AA, out, ra0, dec0, seed=0):
    FE = np.where(np.isfinite(FE)&(FE > 0), FE, F*.1); R = float(np.max(np.hypot(X, Y)))+10
    i, j, th, rho = pair_lines(X, Y, F, FE, AA); H = grid(th, rho, R)
    rng = np.random.default_rng(seed); thS = rng.uniform(0, np.pi, len(th)); mx = (X[i]+X[j])/2; my = (Y[i]+Y[j])/2
    Hs = grid(thS, -mx*np.sin(thS)+my*np.cos(thS), R)
    top = []
    for k in np.argsort(H.ravel())[::-1][:10]:
        a, b = np.unravel_index(k, H.shape); t0 = (a+1)*DTH; r0 = -R+(b+1)*DRHO
        sel = (np.abs(((th-t0+np.pi/2) % np.pi)-np.pi/2) < DTH)&(np.abs(rho-r0) < DRHO)
        top.append(dict(count=float(H[a, b]), theta_deg=float(np.degrees(t0)), rho=float(r0),
                        pairs=[[round(float(X[p]),2), round(float(Y[p]),2), round(float(X[q]),2), round(float(Y[q]),2)] for p, q in zip(i[sel][:200], j[sel][:200])]))
    res = dict(npairs=int(len(th)), max=float(H.max()), median=float(np.median(H)), scramble_max=float(Hs.max()), ra0=ra0, dec0=dec0, top=top)
    json.dump(res, open(out, "w")); return res
