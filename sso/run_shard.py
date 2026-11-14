"""One GitHub runner: for each assigned Euclid DR1 observation, download its 4 long VIS exposure catalogs
(anonymous, schema dr1), keep only needed columns, search for slow movers, test completeness with injected fakes."""
import os, sys, json, time, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from astropy.io import fits
from astroquery.esa.euclid import Euclid
from scipy.spatial import cKDTree
import slow
COLS=["ALPHAWIN_J2000","DELTAWIN_J2000","ERRAWIN_WORLD","FLUX_AUTO","FLUXERR_AUTO","SNR_WIN","FLAGS","ELONGATION","FWHM_IMAGE","SPREAD_MODEL","MAG_AUTO"]
NS, K = int(sys.argv[1]), int(sys.argv[2])
F=pd.read_csv("all_files.csv"); obs=sorted(F.observation_id.unique()); mine=[o for i,o in enumerate(obs) if i%NS==K]
os.makedirs("cat",exist_ok=True); os.makedirs("tmp",exist_ok=True); os.makedirs("out",exist_ok=True)
def fetch(r):
    out=f"cat/{r.file_name[:-5]}.npz"
    if os.path.exists(out): return True
    tf=f"tmp/{r.file_name}"
    for k in range(3):
        try:
            Euclid.get_product(file_name=r.file_name,schema="dr1",output_file=tf)
            if os.path.exists(tf) and os.path.getsize(tf)>0: break
        except Exception as e: print("retry",r.file_name,e,flush=True); time.sleep(10)
    if not os.path.exists(tf): return False
    try:
        h=fits.open(tf,memmap=True); parts=[np.c_[[np.asarray(x.data[c],float) for c in COLS]].T for x in h if x.name=="LDAC_OBJECTS" and x.data is not None and len(x.data)]
        h.close(); np.savez_compressed(out,A=np.concatenate(parts),obs=r.observation_id,t=str(r.obs_time)); return True
    finally:
        if os.path.exists(tf): os.remove(tf)
summ=[]
for o in mine:
    t0=time.time(); ok=all(fetch(r) for r in F[F.observation_id==o].itertuples())
    row=dict(obs=int(o),downloaded=ok)
    if ok:
        try:
            D,_,h,n=slow.run(int(o)); c=D[(D.sig>8)&(D.chi2<12)].copy(); c["obs"]=o
            c.to_csv(f"out/movers_{o}.csv",index=False)
            Dj,inj,_,_=slow.run(int(o),inject=dict(n=60,rates=[0.2,0.3,0.5,0.7,1.0]),seed=int(o))
            I=np.array([(x,y) for x,y,v,s in inj]); cj=Dj[(Dj.sig>8)&(Dj.chi2<12)]
            rec=float((cKDTree(cj[["x","y"]].values).query(I)[0]<1).mean()) if len(cj) else 0.0
            row.update(epochs=len(h),sources=int(n),movers=int(len(c)),inj_recovery=rec,span_h=float(h[-1]))
        except Exception as e: row["error"]=str(e)
    row["minutes"]=round((time.time()-t0)/60,1); summ.append(row); print(row,flush=True)
    for f in [f for f in os.listdir("cat") if f"-{int(o):06d}-" in f]: os.remove("cat/"+f)
pd.DataFrame(summ).to_csv(f"out/summary_{K}.csv",index=False)
