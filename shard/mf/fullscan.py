import numpy as np, sys, pickle, time
from blind import prep, scan
from weights import weights
name, dirs, tag = sys.argv[1], tuple(sys.argv[2].split(",")), sys.argv[3]; mode = sys.argv[4] if len(sys.argv) > 4 else "data"
d, x, y, c1, c2, sig = prep(name, dirs); w, var, resp = weights(d, c1, c2)
if mode == "scramble":   # rotate every shape by a random angle: destroys any coherent signal, keeps positions + noise
    rng = np.random.default_rng(int(sys.argv[5])); ph = rng.uniform(0, np.pi, len(c1)) * 2
    c1, c2 = c1 * np.cos(ph) - c2 * np.sin(ph), c1 * np.sin(ph) + c2 * np.cos(ph)
dt = float(sys.argv[6]) if len(sys.argv) > 6 else 1.5e-5
tag = tag + "_dm"
t = time.time(); out = scan(x, y, c1, c2, sig, 0.0, np.pi, dt, wt=w, var=var)
out.update(field=name, tag=tag, mode=mode, N=len(x), minutes=(time.time() - t) / 60)
pickle.dump(out, open(f"scan_{name}_{tag}_{mode}{sys.argv[5] if mode=='scramble' else ''}.pkl", "wb"))
print(name, tag, mode, "N", len(x), "max S %.2f" % out["S"].max(), "top5", np.round(out["top"][:5], 4).tolist(), "%.0f min" % out["minutes"], flush=True)
