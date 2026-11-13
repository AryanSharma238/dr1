"""Field-level string search: merge all tiles of a Euclid deep field into one gnomonic plane (arcsec),
find twin pairs per tile, then search for lines spanning the whole field.  Hidden-region test: lines are found
using pairs in the WEST half only; the EAST half is opened afterwards to count pairs on the predicted continuation."""
import numpy as np, pandas as pd, glob, os
from detect import twin_pairs, lines

FIELDS = {"EDF-N": lambda dec: dec > 55, "EDF-F": lambda dec: (dec < -20) & (dec > -40), "EDF-S": lambda dec: dec < -40}

def gnomonic(ra, dec, ra0, dec0):
    ra, dec, ra0, dec0 = map(np.radians, (ra, dec, ra0, dec0))
    c = np.sin(dec0) * np.sin(dec) + np.cos(dec0) * np.cos(dec) * np.cos(ra - ra0)
    x = np.cos(dec) * np.sin(ra - ra0) / c
    y = (np.cos(dec0) * np.sin(dec) - np.sin(dec0) * np.cos(dec) * np.cos(ra - ra0)) / c
    return np.degrees(x) * 3600, np.degrees(y) * 3600

LEAN = ["object_id","ra","dec","flux_vis_2fwhm_aper","fluxerr_vis_2fwhm_aper","semimajor_axis","ellipticity","position_angle"]
def load_field(name, dirs=("cat", "catf"), lean=False):
    frames = []
    for dd in dirs:
        for p in glob.glob(f"{dd}/*.csv"):
            if os.path.getsize(p) < 1000: continue
            t = pd.read_csv(p, usecols=LEAN) if lean else pd.read_csv(p)
            if lean:
                for c in LEAN[3:]: t[c] = t[c].astype(np.float32)
            if len(t) and FIELDS[name](t.dec.median()): frames.append(t)
    d = pd.concat(frames, ignore_index=True).drop_duplicates("object_id")
    d = d[(d.flux_vis_2fwhm_aper > 0) & np.isfinite(d.ra) & np.isfinite(d.dec)].copy()
    for c in ("ellipticity", "position_angle", "semimajor_axis"): d[c] = d[c].fillna(0)
    d["fluxerr_vis_2fwhm_aper"] = d.fluxerr_vis_2fwhm_aper.fillna(d.flux_vis_2fwhm_aper * 0.1)
    ra0, dec0 = d.ra.median(), d.dec.median()
    d["x"], d["y"] = gnomonic(d.ra.values, d.dec.values, ra0, dec0)
    d["m"] = -2.5 * np.log10(d.flux_vis_2fwhm_aper) + 23.9
    d["dm"] = 1.0857 * d.fluxerr_vis_2fwhm_aper.clip(upper=1e3) / d.flux_vis_2fwhm_aper
    for b in ("yjh" if not lean else ""):
        f = d[f"flux_{b}_templfit"]; d["mag_" + b] = np.where(f > 0, -2.5 * np.log10(f.where(f > 0)) + 23.9, np.nan)
    if not lean: d["c1"] = d.mag_y - d.mag_j; d["c2"] = d.mag_j - d.mag_h
    d = d[(d.m > 20) & (d.m < 27.0)].reset_index(drop=True)
    return d, (ra0, dec0)

def field_pairs(d, smin=0.3, smax=1.0, chunk=1800.0):
    """twin pairs computed on overlapping spatial chunks (KD-tree per chunk), deduplicated"""
    out = {}
    xs = np.arange(d.x.min(), d.x.max() + chunk, chunk); ys = np.arange(d.y.min(), d.y.max() + chunk, chunk)
    for x0 in xs:
        for y0 in ys:
            sel = (d.x >= x0 - 2) & (d.x < x0 + chunk + 2) & (d.y >= y0 - 2) & (d.y < y0 + chunk + 2)
            sub = d[sel]
            if len(sub) < 2: continue
            P, F = twin_pairs(sub.reset_index(drop=True), smin, smax)
            ids = sub.index.values
            for (i, j), f in zip(P, F): out[(ids[i], ids[j])] = f
    keys = np.array(list(out.keys())).reshape(-1, 2); F = np.array(list(out.values())).reshape(-1, 4)
    return keys, F

def line_members(F, th0, rho0, dth=6.0, drho=0.6, mask=None):
    n = np.array([-np.sin(np.radians(th0)), np.cos(np.radians(th0))])
    ok = (np.abs(((F[:, 2] - th0) + 90) % 180 - 90) < dth) & (np.abs(F[:, :2] @ n - rho0) < drho)
    if mask is not None: ok &= mask
    return np.where(ok)[0]
