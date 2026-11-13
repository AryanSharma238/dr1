import json, glob, os, subprocess, numpy as np
os.chdir("/home/claude/strings"); done = set(open("vetted.txt").read().split()) if os.path.exists("vetted.txt") else {"P0000", "P0001"}
for f in sorted(glob.glob("dr1/P*/summary.json")):
    s = json.load(open(f)); tag = s["tag"]
    if tag in done: continue
    msg = f"{tag} max_S {s['max_S']:.2f}"
    cand = [l for l in s["lines"] if l[0] >= 6.8][:5]   # was: only the single strongest line per patch
    for S, th, rho, w in cand:
            ra0, dec0 = s["ra0"], s["dec0"]
            n = np.array([-np.sin(th), np.cos(th)]); t = np.array([np.cos(th), np.sin(th)])
            def inv(p):
                X, Y = np.radians(p / 3600); r0, d0 = np.radians(ra0), np.radians(dec0); rr = np.hypot(X, Y); c = np.arctan(rr)
                return np.degrees(r0 + np.arctan2(X * np.sin(c), rr * np.cos(d0) * np.cos(c) - Y * np.sin(d0) * np.sin(c))) % 360, np.degrees(np.arcsin(np.cos(c) * np.sin(d0) + Y * np.sin(c) * np.cos(d0) / rr))
            r1, d1 = inv(rho * n); r2, d2 = inv(rho * n + 10 * t); pa = np.degrees(np.arctan2((r2 - r1) * np.cos(np.radians(d1)), d2 - d1)) % 180
            out = subprocess.run(["python3", "extend.py", "dr1.mer_catalogue_wide_survey", "tilecenters_dr1.csv", f"{r1:.6f}", f"{d1:.6f}", f"{pa:.4f}", "8.0", "0.35", f"dr1_{tag}"],
                                 capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS="1")).stdout
            tot = [l for l in out.splitlines() if "total S" in l]; msg += f"\n   line S {S:.2f} RA {r1:.4f} Dec {d1:.4f} PA {pa:.2f} | " + (tot[0] if tot else "extend failed")

    print(msg); open("vetted.txt", "a").write(tag + "\n"); open("dr1_results.txt", "a").write(msg + "\n")
