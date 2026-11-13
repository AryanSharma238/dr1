"""SExtractor-like measurement: render intrinsic Gaussian (+string map) at 0.02", PSF-convolve, rebin to 0.1" VIS pixels, add noise
set by catalogue S/N, detect footprint (PSF-filtered image > 1.5 sigma, connected to centre), unweighted isophotal moments.
Same noise realisation for lensed/unlensed so the difference is the response."""
import numpy as np
from scipy.ndimage import gaussian_filter, label
from imgmodel import render, PIX, SPSF
REB = 5; NP = 160 // REB; gp = (np.arange(NP) - NP / 2 + 0.5) * 0.1; UP, VP = np.meshgrid(gp, gp, indexing="ij")
def iso_moments(img, noise, thr=1.5):
    det = gaussian_filter(img, SPSF / 0.1); dn = noise * np.sqrt(1 / (4 * np.pi * (SPSF / 0.1) ** 2))
    lab, _ = label(det > thr * dn); c = lab[NP // 2, NP // 2]
    if c == 0:   # centre not above threshold: take largest component
        if lab.max() == 0: return None
        c = np.bincount(lab.ravel())[1:].argmax() + 1
    m = lab == c; I = np.where(m, img, 0).clip(0)
    f = I.sum()
    if f <= 0: return None
    if m.sum() < 3: return None
    cu = (I * UP).sum() / f; cv = (I * VP).sum() / f; du, dv = UP - cu, VP - cv
    uu = (I * du * du).sum() / f; vv = (I * dv * dv).sum() / f; uv = (I * du * dv).sum() / f; T = uu + vv
    if not np.isfinite(T) or T <= 0: return None
    return (uu - vv) / T, 2 * uv / T, cu, f
def lensed_change_iso(sig_maj, sig_min, ang, u0, delta, snr, rng):
    c, s = np.cos(ang), np.sin(ang); a2, b2 = sig_maj ** 2, sig_min ** 2
    su = a2 * c * c + b2 * s * s; sv = a2 * s * s + b2 * c * c; suv = (a2 - b2) * s * c
    imgs = [gaussian_filter(render(su, sv, suv, u0, dl), SPSF / PIX).reshape(NP, REB, NP, REB).sum((1, 3)) for dl in (0.0, delta)]
    # normalise so the unlensed galaxy has aperture S/N = snr (aperture radius ~0.3")
    ap = UP ** 2 + VP ** 2 < 0.3 ** 2; amp = 1.0 / imgs[0][ap].sum(); noise = 1.0 / (snr * np.sqrt(ap.sum()))
    nz = rng.normal(0, noise, (NP, NP))
    r = [iso_moments(im * amp + nz, noise) for im in imgs]
    if r[0] is None or r[1] is None: return 0.0, 0.0, 0.0, 1.0
    return r[1][0] - r[0][0], r[1][1] - r[0][1], r[1][2] - r[0][2], r[1][3] / r[0][3]
if __name__ == "__main__":
    from imgmodel import lensed_change
    rng = np.random.default_rng(0)
    for sm in (0.05, 0.1, 0.2, 0.35):
        for snr in (10, 30):
            for dl in (0.15, 0.25):
                offs = np.linspace(-sm * 2.5 - dl / 2, sm * 2.5 + dl / 2, 9)
                iso = np.array([np.mean([lensed_change_iso(sm, sm * .7, 0.0, u, dl, snr, rng)[0] for _ in range(8)]) for u in offs])
                ad = np.array([lensed_change(sm, sm * .7, 0.0, u, dl)[0] for u in offs])
                print("sig %.2f snr %2d delta %.2f | iso %s | adaptive %s | iso/adaptive (summed) %.2f" % (sm, snr, dl, np.round(iso, 2), np.round(ad, 2), iso.sum() / ad.sum()), flush=True)
