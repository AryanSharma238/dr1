"""Python wrapper for the C blind scan. Shapes converted to the gnomonic (east,north) frame: angle_xy = 90 - PA_cat."""
import numpy as np, ctypes, time
from field import load_field
lib = ctypes.CDLL("/home/claude/strings/csrc/libhough.so")
D = ctypes.POINTER(ctypes.c_double); I = ctypes.POINTER(ctypes.c_int)
CELL = 600.0
def prep(name, dirs, demean=True):
    d, _ = load_field(name, dirs, lean=True)
    a = d.semimajor_axis.values * 0.1; e = d.ellipticity.values.clip(0, 0.95); ok = a > 0
    d = d[ok].reset_index(drop=True); a = a[ok]; e = e[ok]
    chi = (1 - (1 - e) ** 2) / (1 + (1 - e) ** 2); ang = np.radians(90.0 - d.position_angle.values)
    c1 = chi * np.cos(2 * ang); c2 = chi * np.sin(2 * ang)
    if demean:   # remove coherent instrumental (PSF-residual) shape patterns: subtract mean shape in CELL x CELL arcsec cells
        x0, y0 = d.x.values, d.y.values; key = np.floor(x0 / CELL).astype(np.int64) * 1000003 + np.floor(y0 / CELL).astype(np.int64)
        _, inv = np.unique(key, return_inverse=True); inv = inv.ravel(); n = np.bincount(inv)
        c1 = c1 - (np.bincount(inv, c1) / n)[inv]; c2 = c2 - (np.bincount(inv, c2) / n)[inv]
    sig = float(np.sqrt(np.mean(c1 ** 2)))
    return d, np.ascontiguousarray(d.x.values, float), np.ascontiguousarray(d.y.values, float), np.ascontiguousarray(c1), np.ascontiguousarray(c2), sig
def scan(x, y, c1, c2, sig, t0, t1, dt, wt=None, var=None, b=0.1, W=(2, 3, 5), nmin=60, ntop=400, nh=200, hmin=-10.0, hstep=0.1):
    return _scan(x, y, c1, c2, sig, t0, t1, dt, wt, var, b, W, nmin, ntop, nh, hmin, hstep)
def _scan(x, y, c1, c2, sig, t0, t1, dt, wt, var, b, W, nmin, ntop, nh, hmin, hstep):
    if wt is None: wt = np.ones(len(x)); var = np.full(len(x), sig ** 2)
    wt = np.ascontiguousarray(wt, float); var = np.ascontiguousarray(var, float)
    nt = int(np.ceil((t1 - t0) / dt)); R = np.max(np.hypot(x, y)) + 5; rmin = -R; nb = int(2 * R / b) + 1
    bs, br, bw = np.zeros(nt), np.zeros(nt), np.zeros(nt, np.int32); top = np.zeros(ntop * 4); hist = np.zeros(len(W) * nh)
    Wa = np.array(W, np.int32)
    lib.scan(ctypes.c_int(len(x)), x.ctypes.data_as(D), y.ctypes.data_as(D), c1.ctypes.data_as(D), c2.ctypes.data_as(D), wt.ctypes.data_as(D), var.ctypes.data_as(D),
             ctypes.c_double(t0), ctypes.c_double(dt), ctypes.c_int(nt), ctypes.c_double(b), ctypes.c_double(rmin), ctypes.c_int(nb),
             ctypes.c_int(len(W)), Wa.ctypes.data_as(I), ctypes.c_double(sig), ctypes.c_int(nmin),
             bs.ctypes.data_as(D), br.ctypes.data_as(D), bw.ctypes.data_as(I), ctypes.c_int(ntop), top.ctypes.data_as(D),
             ctypes.c_int(nh), ctypes.c_double(hmin), ctypes.c_double(hstep), hist.ctypes.data_as(D))
    return dict(theta=t0 + np.arange(nt) * dt, S=bs, rho=br, w=bw, top=top.reshape(-1, 4), hist=hist.reshape(len(W), nh))
if __name__ == "__main__":
    d, x, y, c1, c2, sig = prep("EDF-F", ("cat",)); print("N", len(x), "sig %.3f" % sig, "Rmax %.0f" % np.max(np.hypot(x, y)), flush=True)
    t = time.time(); r = scan(x, y, c1, c2, sig, 0.3, 0.3 + 2000 * 1.5e-5, 1.5e-5); dtm = time.time() - t
    print("2000 angles in %.1fs -> full 180deg at dt=1.5e-5 would take %.0f min" % (dtm, dtm / 2000 * np.pi / 1.5e-5 / 60), "max S %.2f" % r["S"].max())
