// Exact re-implementation of hough.c scan(): same maths, same accumulation order, but processes up to 4 angles per pass over the galaxy
// arrays (memory traffic /4). Results are identical to hough.c (verified by compare_fast.py).
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
typedef struct { double key; int idx; } KI;
static int cmpk(const void*a,const void*b){ double d=((KI*)a)->key-((KI*)b)->key; return (d>0)-(d<0); }
#define BT 4
typedef struct { double a,b,v; int c; int pad; } Bin;
void scan(int N, const double*x, const double*y, const double*c1, const double*c2, const double*wt, const double*var,
          double t0, double dt, int nt, double b, double rmin, int nb, int nw, const int*W, double sig, int nmin,
          double*best_S, double*best_rho, int*best_w, int ntop, double*top, int nh, double hmin, double hstep, double*hist)
{
  Bin *G[BT];
  for(int j=0;j<BT;j++){ G[j]=calloc(nb,sizeof(Bin)); }
  KI *ki=malloc(sizeof(KI)*N);
  double *xs=malloc(8*N),*ys=malloc(8*N),*a1=malloc(8*N),*a2=malloc(8*N),*vv=malloc(8*N);
  int block=(int)(0.5/ (dt*180.0/M_PI)); if(block<1) block=1;
  const int nohist = getenv("HOUGH_NOHIST")!=NULL;   // set => skip the (unused by dr1pipe) S histogram and prune windows that cannot beat best/top; S, rho, w, top identical
  for(int k=0;k<ntop;k++){top[4*k]=-1e9;}
  for(int it=0; it<nt; ){
    if(it%block==0){
      double s=sin(t0+it*dt), c=cos(t0+it*dt);
      for(int i=0;i<N;i++){ ki[i].key=-x[i]*s+y[i]*c; ki[i].idx=i; }
      qsort(ki,N,sizeof(KI),cmpk);
      for(int i=0;i<N;i++){ int j=ki[i].idx; xs[i]=x[j]; ys[i]=y[j]; a1[i]=wt[j]*c1[j]; a2[i]=wt[j]*c2[j]; vv[i]=wt[j]*wt[j]*var[j]; }
    }
    int nxt=(it/block+1)*block, nbat=BT; if(nbat>nxt-it) nbat=nxt-it; if(nbat>nt-it) nbat=nt-it;
    double sj[BT],cj[BT];
    for(int j=0;j<nbat;j++){ double t=t0+(it+j)*dt; sj[j]=sin(t); cj[j]=cos(t);
      memset(G[j],0,sizeof(Bin)*nb); }
    if(nbat==BT){
      Bin *Gp[BT]={G[0],G[1],G[2],G[3]}; enum{CH=1024}; int kk[BT][CH];
      for(int i0=0;i0<N;i0+=CH){
        int n=N-i0<CH?N-i0:CH;
        for(int j=0;j<BT;j++){ const double sjj=sj[j],cjj=cj[j]; const double *X=xs+i0,*Y=ys+i0; int *kj=kk[j];
          for(int i=0;i<n;i++) kj[i]=(int)((-X[i]*sjj+Y[i]*cjj-rmin)/b); }   // vectorisable, same IEEE ops as the original
        for(int i=0;i<n;i++){ double p1=a1[i0+i],p2=a2[i0+i],pv=vv[i0+i];
          for(int j=0;j<BT;j++){ int k=kk[j][i]; if(k>=0&&k<nb){ Bin*q=&Gp[j][k]; q->a+=p1; q->b+=p2; q->v+=pv; q->c++; } } }
      }
    } else {
      for(int i=0;i<N;i++){ double X=xs[i],Y=ys[i],p1=a1[i],p2=a2[i],pv=vv[i];
        for(int j=0;j<nbat;j++){ int k=(int)((-X*sj[j]+Y*cj[j]-rmin)/b); if(k<0||k>=nb) continue; Bin*q=&G[j][k]; q->a+=p1;q->b+=p2;q->v+=pv;q->c++; } }
    }
    for(int j=0;j<nbat;j++){
      int q=it+j; double t=t0+q*dt; Bin *Gj=G[j];
      double c2t=cos(2*t), s2t=sin(2*t);
      best_S[q]=-1e9;
      for(int iw=0; iw<nw; iw++){
        int w=W[iw]; double sa=0,sb=0,sv=0; long sc=0;
        for(int k=0;k<2*w+1 && k<nb;k++){ sa+=Gj[k].a; sb+=Gj[k].b; sv+=Gj[k].v; sc+=Gj[k].c; }
        for(int k=w; k<nb-w-1; k++){
          int doit=(sc>=nmin);
          if(doit && nohist){ double Tm=best_S[q]<top[4*(ntop-1)]?best_S[q]:top[4*(ntop-1)];
            if(Tm>0){ double num=-(sa*c2t+sb*s2t), vs=sv>1e-30?sv:1e-30; if(num<=0 || num*num<=Tm*Tm*vs) doit=0; } }
          if(doit){
            double S=-(sa*c2t+sb*s2t)/sqrt(sv>1e-30?sv:1e-30);
            if(S>best_S[q]){best_S[q]=S; best_rho[q]=rmin+(k+0.5)*b; best_w[q]=w;}
            if(!nohist){ int h=(int)((S-hmin)/hstep); if(h>=0&&h<nh) hist[iw*nh+h]+=1; }
            if(S>top[4*(ntop-1)]){
              int p=ntop-1; while(p>0 && top[4*(p-1)]<S){ memcpy(&top[4*p],&top[4*(p-1)],32); p--; }
              top[4*p]=S; top[4*p+1]=t; top[4*p+2]=rmin+(k+0.5)*b; top[4*p+3]=w;
            }
          }
          sa+=Gj[k+w+1].a-Gj[k-w].a; sb+=Gj[k+w+1].b-Gj[k-w].b; sv+=Gj[k+w+1].v-Gj[k-w].v; sc+=Gj[k+w+1].c-Gj[k-w].c;
        }
      }
    }
    it+=nbat;
  }
  for(int j=0;j<BT;j++){ free(G[j]); }
  free(ki);free(xs);free(ys);free(a1);free(a2);free(vv);
}
