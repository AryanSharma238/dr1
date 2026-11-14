"""Summarise twin-pair search over all patches.  usage: python twin_collect.py DIR  (DIR holds .../P*/twin.json, e.g. unzipped artifacts)
A patch is a candidate if its top bin count >= 14 AND exceeds that patch's scrambled-control max by >= 4."""
import json, glob, sys, os
rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "**", "twin.json"), recursive=True)):
    d = json.load(open(f)); tag = os.path.basename(os.path.dirname(f))
    rows.append((tag, d["max"], d["scramble_max"], d["npairs"], d["top"][0]["theta_deg"], d["top"][0]["rho"]))
print(f"{len(rows)} patches"); cands = [r for r in rows if r[1] >= 14 and r[1] >= r[2]+4]
print("max count over patches:", max((r[1] for r in rows), default=0), " max scrambled:", max((r[2] for r in rows), default=0))
print("CANDIDATES:" if cands else "no candidates")
for r in cands: print("  %s  count %.0f  scramble %.0f  theta %.1f deg  rho %.0f arcsec" % (r[0], r[1], r[2], r[4], r[5]))
