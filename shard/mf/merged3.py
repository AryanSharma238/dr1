"""Known-line recovery with IMAGE-LEVEL partial-doubling model. Same field, bright-only vs bright+faint via argv."""
import numpy as np, sys, time
from field import load_field
from scipy.integrate import quad
from imgmodel import SPSF
from isomodel import lensed_change_iso
name, dirs, NI, NNULL = sys.argv[1], tuple(sys.argv[2].split(",")), int(sys.argv[3]), int(sys.argv[4]); rng = np.random.default_rng(11)
d, _ = load_field(name, dirs); N = len(d); X = d[["x", "y"]].values
a = d.semimajor_axis.values * 0.1; e = d.ellipticity.values.clip(0, 0.95); pa = np.radians(d.position_angle.values); ok = a > 0
a2, b2 = a ** 2, (a * (1 - e)) ** 2
c1 = (a2 - b2) / (a2 + b2) * np.cos(2 * pa); c2 = (a2 - b2) / (a2 + b2) * np.sin(2 * pa); sig = np.sqrt(np.mean(c1[ok] ** 2))
smaj = np.sqrt(np.clip(a2 - SPSF ** 2, 0.02 ** 2, 0.6 ** 2)); smin = np.sqrt(np.clip(b2 - SPSF ** 2, 0.015 ** 2, 0.6 ** 2))
snr = np.clip(d.flux_vis_2fwhm_aper.values / d.fluxerr_vis_2fwhm_aper.values, 3, 300)
print(name, dirs, "N", N, "shape noise %.3f" % sig, flush=True)
COMP = np.load("completeness.npy"); zz = np.linspace(0, 6, 400); cz = np.array([quad(lambda u: 1/np.sqrt(.315*(1+u)**3+.685), 0, z)[0] for z in zz])
zs = rng.gamma(3.0, 0.33, N); Cs = np.interp(zs, zz, cz); span = min(np.ptp(X[:, 0]), np.ptp(X[:, 1])); W = (0.2, 0.3, 0.5)
def S(cp_all, pos, th, rho, w):
    n = np.array([-np.sin(np.radians(th)), np.cos(np.radians(th))]); m = (np.abs(pos @ n - rho) < w) & ok
    return cp_all[m].sum() / (sig * np.sqrt(max(m.sum(), 1)))
def proj(th):
    psi = np.radians(th + 90); return c1 * np.cos(2 * psi) + c2 * np.sin(2 * psi)
t0 = time.time(); null = {w: [] for w in W}
for _ in range(NNULL):
    th = rng.uniform(0, 180); rho = rng.uniform(-0.3, 0.3) * span; cp = proj(th)
    for w in W: null[w].append(S(cp, X, th, rho, w))
for w in W:
    v = np.array(null[w]); print("null w=%.1f mean %.2f std %.2f | frac>3: %.4f (gauss .0013) frac>4: %.4f max %.2f" % (w, v.mean(), v.std(), (v > 3).mean(), (v > 4).mean(), v.max()), flush=True)
print("null %.0fs" % (time.time() - t0), flush=True)
for zl in (0.3, 0.6):
    Cl = np.interp(zl, zz, cz); ratio = np.clip((Cs - Cl) / Cs, 0, None) * (zs > zl)
    for gmu in (1e-7, 1.5e-7):
        res = {w: [] for w in W}; naff = []
        for _ in range(NI):
            th = rng.uniform(0, 180); rho = rng.uniform(-0.3, 0.3) * span; sini = np.sqrt(1 - rng.uniform() ** 2)
            n = np.array([-np.sin(np.radians(th)), np.cos(np.radians(th))]); dist = X @ n - rho
            dl = 8 * np.pi * gmu * sini * ratio * 206264.8
            aff = np.where(ok & (dl > 0.01) & (np.abs(dist) < dl / 2 + 3 * smaj))[0]
            split = rng.uniform(0, 1, len(aff)) < np.clip(np.interp(dl[aff], COMP[:, 0], COMP[:, 1]), 0, 1) * (np.abs(dist[aff]) < dl[aff] / 2)
            aff = aff[~split]                                      # deblended twins belong to the resolved-pair search
            cp = proj(th).copy(); pos = X.copy(); psi = np.radians(th + 90)
            for i in aff:
                ang = pa[i] - psi
                dchi, _, dc, _ = lensed_change_iso(smaj[i], smin[i], ang, dist[i], dl[i], snr[i], rng)
                cp[i] += dchi; pos[i] += dc * n                      # centroid moves along split axis
            naff.append(len(aff))
            for w in W: res[w].append(S(cp, pos, th, rho, w))
        print("zl %.1f Gmu %.1e affected/line %d | %s" % (zl, gmu, np.median(naff), " ".join(
            "w=%.1f medS %.2f det>null99 %.2f" % (w, np.median(res[w]), (np.array(res[w]) > np.percentile(null[w], 99)).mean()) for w in W)), flush=True)
