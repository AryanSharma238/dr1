import glob,sys,numpy as np,pandas as pd
from scipy.spatial import cKDTree
from astropy.time import Time
C=dict(ra=0,dec=1,err=2,flux=3,ferr=4,snr=5,flags=6,elong=7,fwhm=8,spread=9,mag=10)
def load(obs):
    fs=sorted(glob.glob(f"cat/EUC_VIS_SWL-CAT-{obs:06d}-0?-1-*.npz")); E=[]
    for f in fs:
        z=np.load(f); A=z["A"]; t=Time(str(z["t"])).mjd+280/86400
        E.append((t,A))
    return E
def proj(A,ra0,de0):
    r,d,r0,d0=map(np.radians,(A[:,0],A[:,1],ra0,de0)); c=np.sin(d0)*np.sin(d)+np.cos(d0)*np.cos(d)*np.cos(r-r0)
    return np.c_[np.degrees(np.cos(d)*np.sin(r-r0)/c)*3600, np.degrees((np.cos(d0)*np.sin(d)-np.sin(d0)*np.cos(d)*np.cos(r-r0))/c)*3600]
def affine_to(ref,X):
    # fit X -> ref with bright matches
    i=cKDTree(ref).query(X,distance_upper_bound=0.3)[1]; ok=i<len(ref)
    M=np.c_[X[ok],np.ones(ok.sum())]; P,*_=np.linalg.lstsq(M,ref[i[ok]],rcond=None)
    return np.c_[X,np.ones(len(X))]@P, ok.sum()
def run(obs,inject=None,seed=0,snrmin=8,R=1.2):
    E=load(obs); ra0=np.median(E[0][1][:,0]); de0=np.median(E[0][1][:,1]); T=[];P=[];S=[]
    for t,A in E:
        q=(A[:,C["flags"]]==0)&(A[:,C["snr"]]>snrmin)&(A[:,C["spread"]]>-0.002)&(A[:,C["spread"]]<0.01)&(A[:,C["elong"]]<1.6)&(A[:,C["fwhm"]]>1.3)
        A=A[q]; X=proj(A,ra0,de0); T.append(t); P.append(X); S.append(A)
    T=np.array(T); hrs=(T-T[0])*24
    for k in range(1,len(P)): P[k],n=affine_to(P[0],P[k])
    inj=[]
    if inject:
        rng=np.random.default_rng(seed)
        for _ in range(inject["n"]):
            x0,y0=rng.uniform(-1500,1500,2); v=rng.choice(inject["rates"]); ang=rng.uniform(0,2*np.pi); snr=rng.uniform(snrmin,60); sig=0.16/snr*1.0
            inj.append((x0,y0,v,snr))
            for k in range(len(P)):
                p=np.array([[x0+v*hrs[k]*np.cos(ang),y0+v*hrs[k]*np.sin(ang)]])+rng.normal(0,sig,(1,2))
                P[k]=np.r_[P[k],p]; a=S[k][:1].copy(); a[0,C["snr"]]=snr; a[0,C["err"]]=sig/3600; S[k]=np.r_[S[k],a]
    trees=[cKDTree(p) for p in P]; out=[]
    X0=P[0]
    idx=[trees[k].query(X0,distance_upper_bound=R)[1] for k in range(1,len(P))]
    ok=np.all([ix<len(P[k+1]) for k,ix in enumerate(idx)],axis=0)
    for j in np.where(ok)[0]:
        pts=np.array([X0[j]]+[P[k+1][idx[k][j]] for k in range(len(P)-1)])
        errs=np.array([S[0][j,C["err"]]]+[S[k+1][idx[k][j],C["err"]] for k in range(len(P)-1)])*3600
        sig=np.sqrt(errs**2+0.01**2)   # 10 mas floor for frame-to-frame systematics
        w=1/sig**2; A=np.c_[np.ones(len(hrs)),hrs]
        res=[];vel=[];ev=[]
        for d in range(2):
            Wm=A.T@(A*w[:,None]); cov=np.linalg.inv(Wm); b=cov@(A.T@(w*pts[:,d])); vel.append(b[1]); ev.append(np.sqrt(cov[1,1])); res.append(((pts[:,d]-A@b)**2*w).sum())
        v=np.hypot(*vel); sv=np.hypot(vel[0]*ev[0],vel[1]*ev[1])/max(v,1e-9)
        out.append((j,X0[j,0],X0[j,1],v,v/sv,sum(res),S[0][j,C["mag"]],S[0][j,C["snr"]],S[0][j,0],S[0][j,1],vel[0],vel[1],T[0]))
    D=pd.DataFrame(out,columns=["j","x","y","rate","sig","chi2","mag","snr","ra","dec","vx","vy","mjd"])
    return D,inj,hrs,len(P[0])
if __name__=="__main__":
    obs=int(sys.argv[1]); D,_,hrs,n=run(obs)
    print("obs",obs,"epochs(h)",np.round(hrs,2),"sources",n,"matched in all",len(D))
    print("motion significance quantiles",np.percentile(D.sig,[50,99,99.9]).round(2))
    c=D[(D.sig>8)&(D.chi2<12)].sort_values("sig",ascending=False); print(len(c),"significant movers"); print(c.head(15).round(3).to_string())
