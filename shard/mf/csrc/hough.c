// Blind line scan of the merged-twin shape statistic.
// For each angle theta (string direction), project galaxies onto normal n=(-sin,cos), bin rho at width b,
// accumulate sum c1, sum c2, count; split-direction component cp = -(c1 cos2t + c2 sin2t).
// Sliding window of half-width W bins -> S = sum cp / (sig*sqrt(N)). Keep best per angle, a global top list, and a tail histogram.
#include <stdlib.h>
#include <math.h>
#include <string.h>
typedef struct { double key; int idx; } KI;
static int cmpk(const void*a,const void*b){ double d=((KI*)a)->key-((KI*)b)->key; return (d>0)-(d<0); }
void scan(int N, const double*x, const double*y, const double*c1, const double*c2, const double*wt, const double*var,
          double t0, double dt, int nt, double b, double rmin, int nb, int nw, const int*W, double sig, int nmin,
          double*best_S, double*best_rho, int*best_w,           // per angle
          int ntop, double*top,                                   // ntop x 4 (S, theta, rho, w) - simple replacement of min
          int nh, double hmin, double hstep, double*hist)         // histogram of S over all (theta, rho, w) cells (subsampled every bin)
{
  double *A=calloc(nb,8),*B=calloc(nb,8),*V=calloc(nb,8); int *C=calloc(nb,4);
  KI *ki=malloc(sizeof(KI)*N); int *ord=malloc(sizeof(int)*N);
  double *xs=malloc(8*N),*ys=malloc(8*N),*a1=malloc(8*N),*a2=malloc(8*N),*vv=malloc(8*N);
  int block=(int)(0.5/ (dt*180.0/M_PI)); if(block<1) block=1;   // re-sort every 0.5 deg
  for(int k=0;k<ntop;k++){top[4*k]=-1e9;}
  for(int it=0; it<nt; it++){
    double t=t0+it*dt, s=sin(t), c=cos(t);
    if(it%block==0){
      for(int i=0;i<N;i++){ ki[i].key=-x[i]*s+y[i]*c; ki[i].idx=i; }
      qsort(ki,N,sizeof(KI),cmpk);
      for(int i=0;i<N;i++){ int j=ki[i].idx; xs[i]=x[j]; ys[i]=y[j]; a1[i]=wt[j]*c1[j]; a2[i]=wt[j]*c2[j]; vv[i]=wt[j]*wt[j]*var[j]; }
    }
    memset(A,0,8*nb); memset(B,0,8*nb); memset(V,0,8*nb); memset(C,0,4*nb);
    for(int i=0;i<N;i++){
      double r=-xs[i]*s+ys[i]*c; int k=(int)((r-rmin)/b);
      if(k<0||k>=nb) continue; A[k]+=a1[i]; B[k]+=a2[i]; V[k]+=vv[i]; C[k]++;
    }
    double c2t=cos(2*t), s2t=sin(2*t);
    best_S[it]=-1e9;
    for(int iw=0; iw<nw; iw++){
      int w=W[iw]; double sa=0,sb=0,sv=0; long sc=0;
      for(int k=0;k<2*w+1 && k<nb;k++){ sa+=A[k]; sb+=B[k]; sv+=V[k]; sc+=C[k]; }
      for(int k=w; k<nb-w-1; k++){
        if(sc>=nmin){
          double S=-(sa*c2t+sb*s2t)/sqrt(sv>1e-30?sv:1e-30);
          if(S>best_S[it]){best_S[it]=S; best_rho[it]=rmin+(k+0.5)*b; best_w[it]=w;}
          int h=(int)((S-hmin)/hstep); if(h>=0&&h<nh) hist[iw*nh+h]+=1;
          if(S>top[4*(ntop-1)]){            // insert into sorted top list (descending)
            int p=ntop-1; while(p>0 && top[4*(p-1)]<S){ memcpy(&top[4*p],&top[4*(p-1)],32); p--; }
            top[4*p]=S; top[4*p+1]=t; top[4*p+2]=rmin+(k+0.5)*b; top[4*p+3]=w;
          }
        }
        sa+=A[k+w+1]-A[k-w]; sb+=B[k+w+1]-B[k-w]; sv+=V[k+w+1]-V[k-w]; sc+=C[k+w+1]-C[k-w];
      }
    }
  }
  free(A);free(B);free(V);free(C);free(vv);free(ki);free(ord);free(xs);free(ys);free(a1);free(a2);
}
