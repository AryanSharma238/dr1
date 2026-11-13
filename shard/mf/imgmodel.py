"""Image-level string lensing of extended galaxies: render intrinsic Gaussian, apply exact conical map (duplicate band of width delta),
convolve PSF, measure adaptive (Gaussian-weighted) moments. Returns change in chi (along split axis), centroid shift, flux ratio."""
import numpy as np
from scipy.ndimage import gaussian_filter
PIX = 0.02; NG = 160; SPSF = 0.16 / 2.355
g = (np.arange(NG) - NG / 2 + 0.5) * PIX; U, V = np.meshgrid(g, g, indexing="ij")   # U = perp to string, V = along
def render(su, sv, suv, u0, delta):
    """intrinsic Gaussian (cov [[su,suv],[suv,sv]]) centred at u=u0 (source plane); string at u=0 in image plane."""
    us = U - np.sign(U) * delta / 2 if delta > 0 else U          # image u -> source u
    det = su * sv - suv ** 2; du = us - u0
    return np.exp(-0.5 * (sv * du ** 2 - 2 * suv * du * V + su * V ** 2) / det)
def admom(I, it=30):
    """adaptive Gaussian-weighted moments (HSM-like)."""
    f = I.sum(); cu = (I * U).sum() / f; cv = (I * V).sum() / f
    Muu = Mvv = 0.05; Muv = 0.0
    for _ in range(it):
        du, dv = U - cu, V - cv; det = Muu * Mvv - Muv ** 2
        w = np.exp(-0.5 * (Mvv * du ** 2 - 2 * Muv * du * dv + Muu * dv ** 2) / det); wi = w * I; s = wi.sum()
        cu += (wi * du).sum() / s; cv += (wi * dv).sum() / s
        nuu = 2 * (wi * du * du).sum() / s; nvv = 2 * (wi * dv * dv).sum() / s; nuv = 2 * (wi * du * dv).sum() / s
        if abs(nuu - Muu) < 1e-7 and abs(nvv - Mvv) < 1e-7: Muu, Mvv, Muv = nuu, nvv, nuv; break
        Muu, Mvv, Muv = nuu, nvv, nuv
    T = Muu + Mvv; return (Muu - Mvv) / T, 2 * Muv / T, cu, f
def lensed_change(sig_maj, sig_min, ang, u0, delta):
    """ang = galaxy major-axis angle relative to the U (split) axis. Returns dchi_split, dchi_cross, centroid shift along U, flux ratio."""
    c, s = np.cos(ang), np.sin(ang); a2, b2 = sig_maj ** 2, sig_min ** 2
    su = a2 * c * c + b2 * s * s; sv = a2 * s * s + b2 * c * c; suv = (a2 - b2) * s * c
    out = []
    for dl in (0.0, delta):
        I = gaussian_filter(render(su, sv, suv, u0, dl), SPSF / PIX); out.append(admom(I))
    (x0, y0, c0, f0), (x1, y1, c1, f1) = out
    return x1 - x0, y1 - y0, c1 - c0, f1 / f0
if __name__ == "__main__":
    # compare against the old "full copy" approximation for typical sizes
    for sm in (0.05, 0.1, 0.2, 0.35):
        for dl in (0.15, 0.25):
            r = [lensed_change(sm, sm * 0.7, 0.0, u0, dl) for u0 in np.linspace(-sm * 2.5, sm * 2.5, 11)]
            dchi = np.array([x[0] for x in r]); fr = np.array([x[3] for x in r])
            print("sigma %.2f\" delta %.2f\": dchi_split by offset %s | flux ratio max %.2f | sum dchi over offsets(per arcsec) %.3f" %
                  (sm, dl, np.round(dchi, 3), fr.max(), dchi.sum() * (5 * sm / 10)))
