import json, numpy as np, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/claude/strings")
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter, label, map_coordinates
from imgmodel import render, PIX, SPSF
rows = json.load(open("/home/claude/strings/realval_rows.json")); rng = np.random.default_rng(5)
def moments(img, noise, thr=1.5):
    ny, nx = img.shape; yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    det = gaussian_filter(img, SPSF / 0.1); dn = noise * np.sqrt(1 / (4 * np.pi * (SPSF / 0.1) ** 2))
    lab, _ = label(det > thr * dn); c = lab[ny // 2, nx // 2]
    if c == 0: return None
    m = lab == c
    if m.sum() < 3: return None
    I = np.where(m, img, 0).clip(0); f = I.sum(); cx = (I * xx).sum() / f; cy = (I * yy).sum() / f
    dx, dy = xx - cx, yy - cy; xx2 = (I * dx * dx).sum() / f; yy2 = (I * dy * dy).sum() / f; xy = (I * dx * dy).sum() / f; T = xx2 + yy2
    if T <= 0: return None
    return np.array([(xx2 - yy2) / T, 2 * xy / T]), np.array([cx, cy]), f
pa_meas, pa_cat, resp = [], [], []
for r in rows:
    try: h = fits.open(r["fn"]); img = h[0].data.astype(float); w = WCS(h[0].header)
    except Exception: continue
    if img.ndim != 2 or img.shape[0] < 40: continue
    img = img - np.median(img); noise = 1.4826 * np.median(np.abs(img[np.abs(img) < 5 * np.std(img)]))
    out = moments(img, noise)
    if out is None: continue
    chi, cen, f = out
    # direction of +x pixel and +y pixel on sky in (east, north)
    ra0, de0 = w.wcs_pix2world([[cen[0], cen[1]]], 0)[0]
    def sky(dxp, dyp):
        ra1, de1 = w.wcs_pix2world([[cen[0] + dxp, cen[1] + dyp]], 0)[0]
        return np.array([(ra1 - ra0) * np.cos(np.radians(de0)), de1 - de0])
    J = np.c_[sky(1, 0), sky(0, 1)]; J /= np.linalg.norm(J[:, 0])       # pixel -> (east, north)
    th_pix = 0.5 * np.arctan2(chi[1], chi[0])                            # major-axis angle in pixel frame
    v = J @ np.array([np.cos(th_pix), np.sin(th_pix)])
    pa_meas.append(np.degrees(np.arctan2(v[1], v[0]))); pa_cat.append(r["pa"])
    # pixel x direction angle on sky (east->north), for reference
    # (2) response test: random string through galaxy
    for dl in (0.15, 0.25):
        phi = rng.uniform(0, np.pi); n = np.array([-np.sin(phi), np.cos(phi)])          # in pixel frame
        u0 = rng.uniform(-0.3, 0.3)                                                      # string offset from galaxy centre (arcsec)
        ny, nx = img.shape; yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
        u = ((xx - cen[0]) * n[0] + (yy - cen[1]) * n[1]) * 0.1 - u0                    # arcsec from string
        sh = np.sign(u) * dl / 2 / 0.1                                                   # pixels
        lens = map_coordinates(img, [yy - sh * n[1], xx - sh * n[0]], order=3, mode="nearest")
        o2 = moments(lens, noise)
        if o2 is None: continue
        psi = 2 * phi + np.pi; dchi = (o2[0][0] - chi[0]) * np.cos(psi) + (o2[0][1] - chi[1]) * np.sin(psi)
        # Gaussian model of same galaxy, map applied PRE-PSF (physical) and POST-PSF (like the real-image test)
        a2 = (r["a"] * 0.1) ** 2; b2 = (r["a"] * (1 - r["e"]) * 0.1) ** 2
        sm, sn = np.sqrt(max(a2 - SPSF ** 2, 4e-4)), np.sqrt(max(b2 - SPSF ** 2, 2e-4)); ang = th_pix - (phi + np.pi / 2)
        c, s = np.cos(ang), np.sin(ang); su = sm**2*c*c + sn**2*s*s; sv = sm**2*s*s + sn**2*c*c; suv = (sm**2 - sn**2)*s*c
        from imgmodel import admom
        g0 = gaussian_filter(render(su, sv, suv, -u0, 0.0), SPSF / PIX)
        gpre = gaussian_filter(render(su, sv, suv, -u0, dl), SPSF / PIX)
        G = (np.arange(160) - 80 + 0.5) * PIX; UU, VV = np.meshgrid(G, G, indexing="ij")
        gpost = map_coordinates(g0, [(UU - np.sign(UU) * dl / 2) / PIX + 79.5, VV / PIX + 79.5], order=3)
        m0 = admom(g0)[0]; mpre = admom(gpre)[0] - m0; mpost = admom(gpost)[0] - m0
        resp.append((dl, dchi, mpre, mpost, r["snr"], r["a"]))
pa_meas = np.array(pa_meas); pa_cat = np.array(pa_cat)
for sign in (1, -1):
    for off in (0, 90):
        dd = np.degrees(np.angle(np.mean(np.exp(2j * np.radians(pa_meas - (sign * pa_cat + off)))))) / 2
        coh = np.abs(np.mean(np.exp(2j * np.radians(pa_meas - (sign * pa_cat + off)))))
        print("PA hypothesis: sky_angle(E->N) = %+d*PA_cat + %d : coherence %.2f, residual offset %.1f deg" % (sign, off, coh, dd))
R = np.array(resp); print("response pairs", len(R))
for dl in (0.15, 0.25):
    k = R[R[:, 0] == dl]
    print("delta %.2f: real-image dchi mean %.3f +- %.3f | gauss pre-PSF %.3f | gauss post-PSF %.3f | real/gauss_post %.2f | gauss_pre/post %.2f" %
          (dl, k[:, 1].mean(), k[:, 1].std() / np.sqrt(len(k)), k[:, 2].mean(), k[:, 3].mean(), k[:, 1].mean() / k[:, 3].mean(), k[:, 2].mean() / k[:, 3].mean()))
