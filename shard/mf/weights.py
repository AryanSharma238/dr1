"""Optimal per-galaxy weights w = R(size)/var(size,snr): R = expected split-axis ellipticity response (image model x 0.6, averaged over
offsets |u0|<0.3" and orientations, delta=0.2"), var = empirical shape-noise variance per component in (size, S/N) bins."""
import numpy as np
from imgmodel import lensed_change, SPSF
rng = np.random.default_rng(0)
sgrid = np.array([0.02, 0.04, 0.07, 0.1, 0.15, 0.2, 0.3, 0.45, 0.6]); R = []
for s in sgrid:
    v = [lensed_change(s, s * 0.7, rng.uniform(0, np.pi), rng.uniform(-0.3, 0.3), 0.2)[0] for _ in range(40)]
    R.append(0.6 * np.mean(v))
R = np.array(R); np.save("resp_table.npy", np.c_[sgrid, R]); print("response vs intrinsic sigma:", dict(zip(sgrid, np.round(R, 3))))
def weights(d, c1, c2):
    a = d.semimajor_axis.values * 0.1; sint = np.sqrt(np.clip(a ** 2 - SPSF ** 2, 0.02 ** 2, 0.6 ** 2))
    resp = np.interp(sint, sgrid, R); snr = d.flux_vis_2fwhm_aper.values / d.fluxerr_vis_2fwhm_aper.values
    lb = np.digitize(np.log10(sint), np.linspace(-1.7, -0.2, 12)); sb = np.digitize(np.log10(np.clip(snr, 1, 1e4)), np.linspace(0.5, 3, 12))
    key = lb * 20 + sb; var = np.zeros(len(a))
    for k in np.unique(key):
        m = key == k; var[m] = 0.5 * (np.mean(c1[m] ** 2) + np.mean(c2[m] ** 2)) if m.sum() > 50 else np.mean(c1 ** 2)
    return resp / var, var, resp
