"""End-to-end check of the streaming pipeline: inject strings into raw patch shapes BEFORE local-mean subtraction, then clean, weight, blind-scan."""
import numpy as np, glob, sys, pandas as pd
from blind import scan, CELL
from weights import weights
from imgmodel import lensed_change, SPSF
from scipy.integrate import quad
TAG, RA0, DEC0 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]); rng = np.random.default_rng(99)
A = np.concatenate([np.load(f) for f in glob.glob(f"dr1/{TAG}/raw/*.npy")]); ra, dec, flux, ferr, a, e, pa = A.T
r, d, r0, d0 = map(np.radians, (ra, dec, RA0, DEC0)); c = np.sin(d0)*np.sin(d)+np.cos(d0)*np.cos(d)*np.cos(r-r0)
x = np.degrees(np.cos(d)*np.sin(r-r0)/c)*3600; y = np.degrees((np.cos(d0)*np.sin(d)-np.sin(d0)*np.cos(d)*np.cos(r-r0))/c)*3600
mag = -2.5*np.log10(np.where(flux > 0, flux, 1))+23.9; m = (flux > 0)&(a > 0)&(mag > 20)&(mag < 27)
x, y, flux, ferr, a, e, pa, mag = [v[m] for v in (x, y, flux, ferr, a, e, pa, mag)]; e = np.clip(np.nan_to_num(e), 0, .95)
D = pd.DataFrame(dict(x=x, y=y, m=mag, flux_vis_2fwhm_aper=flux, fluxerr_vis_2fwhm_aper=np.where(np.isfinite(ferr), ferr, flux*.1), semimajor_axis=a, ellipticity=e, position_angle=pa))
chi = (1-(1-e)**2)/(1+(1-e)**2); ang = np.radians(90-pa); C1 = chi*np.cos(2*ang); C2 = chi*np.sin(2*ang)
sa = a*0.1; smaj = np.sqrt(np.clip(sa**2-SPSF**2, .02**2, .6**2)); smin = np.sqrt(np.clip((sa*(1-e))**2-SPSF**2, .015**2, .6**2))
COMP = np.load("completeness.npy"); zz = np.linspace(0, 6, 400); cz = np.array([quad(lambda u: 1/np.sqrt(.315*(1+u)**3+.685), 0, z)[0] for z in zz])
zs = rng.gamma(3.0, .33, len(x)); Cs = np.interp(zs, zz, cz); key = np.floor(x/CELL).astype(np.int64)*1000003+np.floor(y/CELL).astype(np.int64)
_, inv = np.unique(key, return_inverse=True); inv = inv.ravel(); nn = np.bincount(inv)
def clean(c1, c2): return c1-(np.bincount(inv, c1)/nn)[inv], c2-(np.bincount(inv, c2)/nn)[inv]
c1, c2 = clean(C1, C2); sig = float(np.sqrt(np.mean(c1**2))); w, var, _ = weights(D, c1, c2)
for zl, gmu in ((0.3, 1e-7), (0.6, 1.5e-7)):
    Cl = np.interp(zl, zz, cz); ratio = np.clip((Cs-Cl)/Cs, 0, None)*(zs > zl); rec = []
    for _ in range(6):
        th = rng.uniform(0, np.pi); rho = rng.uniform(-1500, 1500); sini = np.sqrt(1-rng.uniform()**2)
        n = np.array([-np.sin(th), np.cos(th)]); dist = x*n[0]+y*n[1]-rho; dl = 8*np.pi*gmu*sini*ratio*206264.8
        aff = np.where((dl > .01)&(np.abs(dist) < dl/2+3*smaj))[0]
        split = rng.uniform(0, 1, len(aff)) < np.clip(np.interp(dl[aff], COMP[:, 0], COMP[:, 1]), 0, 1)*(np.abs(dist[aff]) < dl[aff]/2); aff = aff[~split]
        k1, k2, xi, yi = C1.copy(), C2.copy(), x.copy(), y.copy(); psi = th+np.pi/2
        for i in aff:
            dchi, _, dc, _ = lensed_change(smaj[i], smin[i], np.radians(90-pa[i])-psi, dist[i], dl[i]); dchi *= .6
            k1[i] += dchi*np.cos(2*psi); k2[i] += dchi*np.sin(2*psi); xi[i] += dc*n[0]; yi[i] += dc*n[1]
        k1, k2 = clean(k1, k2)
        R = scan(np.ascontiguousarray(xi), np.ascontiguousarray(yi), k1, k2, sig, th-.00436, th+.00436, 2.9e-5, wt=w, var=var, ntop=50)
        near = [t for t in R["top"] if abs(t[1]-th) < 3e-4 and abs(t[2]-rho) < 3]; rec.append(round(near[0][0], 1) if near else 0.0)
    print(f"[{TAG}] zl {zl} Gmu {gmu}: blind-recovered S {sorted(rec)} (line length ~2 deg patch)", flush=True)
