"""Line-extension test for a candidate: given a great-circle line (RA, Dec of a point, PA E of N), download all tiles within 0.1 deg of the
line out to +-L deg, measure the weighted split-axis shape statistic in 30' segments along it, and the cumulative S.
A real straight string should give positive S in most segments and S growing ~sqrt(length); noise should not.
Usage: extend.py TABLE TILECENTERS RA DEC PA HALF_LENGTH_DEG RHO_HALFWIDTH_ARCSEC TAG"""
import numpy as np, sys, os, subprocess, pandas as pd
from concurrent.futures import ThreadPoolExecutor
from weights import weights
TABLE, TC, RA, DEC, PA, HL, HW, TAG = sys.argv[1], sys.argv[2], *map(float, sys.argv[3:8]), sys.argv[8]
WD = f"dr1/ext_{TAG}"; os.makedirs(WD, exist_ok=True)
def unit(ra, de): ra, de = np.radians(ra), np.radians(de); return np.array([np.cos(de) * np.cos(ra), np.cos(de) * np.sin(ra), np.sin(de)])
p0 = unit(RA, DEC); north = np.array([0, 0, 1.0]); e_ = np.cross(north, p0); e_ /= np.linalg.norm(e_); n_ = np.cross(p0, e_)
pa = np.radians(PA); tdir = np.sin(pa) * e_ + np.cos(pa) * n_; pole = np.cross(p0, tdir); pole /= np.linalg.norm(pole)   # great-circle pole
tc = [l.split(",") for l in open(TC).read().split()]
def gc_coords(ra, de):
    v = np.stack([np.cos(np.radians(de)) * np.cos(np.radians(ra)), np.cos(np.radians(de)) * np.sin(np.radians(ra)), np.sin(np.radians(de))], -1)
    perp = np.degrees(np.arcsin(np.clip(v @ pole, -1, 1)))                               # deg from great circle
    along = np.degrees(np.arctan2(v @ tdir, v @ p0)); return along, perp
tiles = [t for t, r, d in tc if (lambda a, p: abs(p) < 0.45 and abs(a) < HL + 0.45)(*gc_coords(float(r), float(d)))]
print(f"[{TAG}] {len(tiles)} tiles along the line", flush=True)
COLS = "right_ascension,declination,flux_vis_2fwhm_aper,fluxerr_vis_2fwhm_aper,semimajor_axis,ellipticity,position_angle"
def get(t):
    out = f"{WD}/t{t}.npy"
    if os.path.exists(out): return
    r = subprocess.run(["curl", "-s", "--max-time", "600", "https://eas.esac.esa.int/tap-server/tap/sync", "--data-urlencode", "REQUEST=doQuery", "--data-urlencode", "LANG=ADQL", "--data-urlencode",
                        f"QUERY=SELECT {COLS} FROM {TABLE} WHERE tile_index={t} AND flux_vis_2fwhm_aper>0.04 AND spurious_flag=0 AND point_like_prob<0.5",
                        "--data-urlencode", "FORMAT=csv"], capture_output=True, text=True).stdout
    try:
        a = np.genfromtxt(r.splitlines()[1:], delimiter=",").reshape(-1, 7); a = a[np.all(np.isfinite(a[:, :5]), 1)]
        al, pp = gc_coords(a[:, 0], a[:, 1]); np.save(out, a[np.abs(pp) * 3600 < 600])     # keep +-10' strip
    except Exception as ex: print("tile", t, ex)
with ThreadPoolExecutor(6) as ex: list(ex.map(get, tiles))
A = np.concatenate([np.load(f"{WD}/t{t}.npy") for t in tiles if os.path.exists(f"{WD}/t{t}.npy")])
ra, de, fl, fe, a, e, pa_c = A.T; al, pp = gc_coords(ra, de); pp *= 3600
e = np.clip(np.nan_to_num(e), 0, .95); chi = (1 - (1 - e) ** 2) / (1 + (1 - e) ** 2)
INJ = os.environ.get("INJ")
if INJ:   # plant a string ON this line (image-level response x0.6), to validate the extension test
    from imgmodel import lensed_change, SPSF
    from scipy.integrate import quad
    vals = list(map(float, INJ.split(","))); gmu, zl = vals[:2]; SINI = vals[2] if len(vals) > 2 else None; rng = np.random.default_rng(5)
    zz = np.linspace(0, 6, 300); cz = np.array([quad(lambda u: 1/np.sqrt(.315*(1+u)**3+.685), 0, z)[0] for z in zz])
    zs = rng.gamma(3.0, .33, len(ra)); ratio = np.clip((np.interp(zs, zz, cz) - np.interp(zl, zz, cz)) / np.interp(zs, zz, cz), 0, None) * (zs > zl)
    dl = 8 * np.pi * gmu * (SINI if SINI else np.sqrt(1 - rng.uniform() ** 2)) * ratio * 206264.8; sa = a * 0.1
    smaj = np.sqrt(np.clip(sa**2 - SPSF**2, .02**2, .6**2)); smin = np.sqrt(np.clip((sa*(1-e))**2 - SPSF**2, .015**2, .6**2))
    INJ_AFF = np.where((dl > .01) & (np.abs(pp) < dl / 2 + 3 * smaj))[0]; INJ_DL = dl; INJ_SMAJ, INJ_SMIN = smaj, smin

# local position angle of the great circle at each galaxy, in that galaxy's (E,N) frame
v = np.stack([np.cos(np.radians(de)) * np.cos(np.radians(ra)), np.cos(np.radians(de)) * np.sin(np.radians(ra)), np.sin(np.radians(de))], -1)
E = np.cross(north, v); E /= np.linalg.norm(E, axis=1)[:, None]; Nn = np.cross(v, E); t_loc = np.cross(pole, v)
th_loc = np.arctan2(t_loc @ np.zeros(3) + np.sum(t_loc * Nn, 1), np.sum(t_loc * E, 1))   # string direction angle from E toward N
ang = np.radians(90 - pa_c); cp = -(chi * np.cos(2 * ang) * np.cos(2 * th_loc) + chi * np.sin(2 * ang) * np.sin(2 * th_loc))
if INJ:
    for i in INJ_AFF:
        dchi = lensed_change(INJ_SMAJ[i], INJ_SMIN[i], ang[i] - (th_loc[i] + np.pi / 2), pp[i], INJ_DL[i])[0] * 0.6; cp[i] += dchi
    dd = np.array([lensed_change(INJ_SMAJ[i], INJ_SMIN[i], ang[i] - (th_loc[i] + np.pi / 2), pp[i], INJ_DL[i])[0] * 0.6 for i in INJ_AFF[:200]])
    print(f"[{TAG}] injected Gmu={gmu} zl={zl}: {len(INJ_AFF)} affected; in |pp|<{HW}: {(np.abs(pp[INJ_AFF])<HW).sum()}; mean dchi {dd.mean():.3f}; median dl {np.median(INJ_DL[INJ_AFF]):.3f}", flush=True)
# local-mean subtraction using galaxies 2'-10' from the line (same strip), per 30' segment
D = pd.DataFrame(dict(m=-2.5 * np.log10(np.clip(fl, 1e-6, None)) + 23.9, flux_vis_2fwhm_aper=fl, fluxerr_vis_2fwhm_aper=np.nan_to_num(fe, nan=0.1) + 1e-9,
                      semimajor_axis=a, ellipticity=e, position_angle=pa_c))
w, var, _ = weights(D, chi * np.cos(2 * ang), chi * np.sin(2 * ang))
seg = np.floor(al / 0.5); cumS = []; out = []
for s in np.unique(seg):
    ms = seg == s; bg = ms & (np.abs(pp) > 120); cpm = cp - (np.average(cp[bg], weights=w[bg]) if bg.sum() > 50 else 0)
    k = ms & (np.abs(pp) < HW)
    if k.sum() < 3: continue
    S = (w[k] * cpm[k]).sum() / np.sqrt((w[k] ** 2 * var[k]).sum()); out.append((s * 0.5, int(k.sum()), S))
out = np.array(out); Stot = (out[:, 2] * np.sqrt(1)).sum() / np.sqrt(len(out))
print(f"[{TAG}] segments (along deg, N, S): " + " ".join(f"{a_:+.1f}:{s_:+.1f}" for a_, n_, s_ in out))
print(f"[{TAG}] total S over {len(out)} segments ({len(out)*0.5:.1f} deg): {Stot:.2f}; fraction of segments with S>0: {(out[:, 2] > 0).mean():.2f}")
