"""Blind recovery of injected strings (image-level response x0.6) using the weighted C scan over +-0.25 deg around the true angle."""
import numpy as np, sys, time
from blind import prep, scan
from weights import weights
from imgmodel import lensed_change, SPSF
from scipy.integrate import quad
name, dirs, NI = sys.argv[1], tuple(sys.argv[2].split(",")), int(sys.argv[3]); rng = np.random.default_rng(17)
d, x, y, c1, c2, sig = prep(name, dirs); w, var, resp = weights(d, c1, c2); N = len(x)
a = d.semimajor_axis.values * 0.1; e = d.ellipticity.values.clip(0, .95); pa_xy = np.radians(90 - d.position_angle.values)
smaj = np.sqrt(np.clip(a ** 2 - SPSF ** 2, .02 ** 2, .6 ** 2)); smin = np.sqrt(np.clip((a * (1 - e)) ** 2 - SPSF ** 2, .015 ** 2, .6 ** 2))
COMP = np.load("completeness.npy"); zz = np.linspace(0, 6, 400); cz = np.array([quad(lambda u: 1/np.sqrt(.315*(1+u)**3+.685), 0, z)[0] for z in zz])
zs = rng.gamma(3.0, 0.33, N); Cs = np.interp(zs, zz, cz); span = min(np.ptp(x), np.ptp(y))
# null for this +-0.25deg window: same scan on random windows of the real data (no injection)
nullmax = []
for _ in range(6):
    t0 = rng.uniform(0, np.pi); r = scan(x, y, c1, c2, sig, t0 - 0.00436, t0 + 0.00436, 1.5e-5, wt=w, var=var, ntop=5); nullmax.append(r["S"].max())
print(name, "window-null max S:", np.round(nullmax, 2), flush=True)
for zl in (0.3, 0.6):
    Cl = np.interp(zl, zz, cz); ratio = np.clip((Cs - Cl) / Cs, 0, None) * (zs > zl)
    for gmu in (1e-7, 1.5e-7, 2e-7):
        rec = []
        for _ in range(NI):
            th = rng.uniform(0, np.pi); rho = rng.uniform(-0.3, 0.3) * span; sini = np.sqrt(1 - rng.uniform() ** 2)
            n = np.array([-np.sin(th), np.cos(th)]); dist = x * n[0] + y * n[1] - rho; dl = 8 * np.pi * gmu * sini * ratio * 206264.8
            aff = np.where((dl > 0.01) & (np.abs(dist) < dl / 2 + 3 * smaj))[0]
            split = rng.uniform(0, 1, len(aff)) < np.clip(np.interp(dl[aff], COMP[:, 0], COMP[:, 1]), 0, 1) * (np.abs(dist[aff]) < dl[aff] / 2)
            aff = aff[~split]; k1, k2, xi, yi = c1.copy(), c2.copy(), x.copy(), y.copy(); psi = th + np.pi / 2
            for i in aff:
                dchi, _, dc, _ = lensed_change(smaj[i], smin[i], pa_xy[i] - psi, dist[i], dl[i]); dchi *= 0.6
                k1[i] += dchi * np.cos(2 * psi); k2[i] += dchi * np.sin(2 * psi); xi[i] += dc * n[0]; yi[i] += dc * n[1]
            r = scan(xi, yi, k1, k2, sig, th - 0.00436, th + 0.00436, 1.5e-5, wt=w, var=var, ntop=50)
            near = [tp for tp in r["top"] if abs(tp[1] - th) < 3e-4 and abs(tp[2] - rho) < 3]
            rec.append((near[0][0] if near else 0.0, r["S"].max()))
        rec = np.array(rec)
        print("zl %.1f Gmu %.1e | recovered-at-truth S median %.2f [%s] | window max median %.2f" % (zl, gmu, np.median(rec[:, 0]), " ".join("%.1f" % v for v in sorted(rec[:, 0])), np.median(rec[:, 1])), flush=True)
