// Exact sampled projected capsule union on an infinite affine periodic yarn.
// Compilation products live in build-support. No constitutive model is used.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include <thread>
#include <atomic>

static bool image(const double *E, const double *a, const double *b,
                  double ax, double ay, double bx, double by,
                  double rx, double ry, double Lx, double Ly, double R,
                  int N, int m, int n, std::vector<unsigned char>& mask,
                  int &hits, double *stats) {
    double sx=m*Lx, sy=n*Ly;
    double minx=std::min(ax,bx)+sx-rx, maxx=std::max(ax,bx)+sx+rx;
    double miny=std::min(ay,by)+sy-ry, maxy=std::max(ay,by)+sy+ry;
    int x0=std::max(0,(int)std::ceil(minx*N/Lx-.5-1e-9));
    int x1=std::min(N-1,(int)std::floor(maxx*N/Lx-.5+1e-9));
    int y0=std::max(0,(int)std::ceil(miny*N/Ly-.5-1e-9));
    int y1=std::min(N-1,(int)std::floor(maxy*N/Ly-.5+1e-9));
    if(x0>x1 || y0>y1) return hits==N*N;
    stats[0]+=1;
    double A0=a[0]+E[0]*sx+E[1]*sy, A1=a[1]+E[2]*sx+E[3]*sy;
    double D0=b[0]-a[0], D1=b[1]-a[1], dd=D0*D0+D1*D1;
    for(int y=y0;y<=y1;y++) for(int x=x0;x<=x1;x++) {
        int index=y*N+x;
        if(mask[index]) continue;
        double qx=(x+.5)*Lx/N, qy=(y+.5)*Ly/N;
        double u=E[0]*qx+E[1]*qy-A0, v=E[2]*qx+E[3]*qy-A1;
        double t=dd>0?std::clamp((u*D0+v*D1)/dd,0.0,1.0):0.0;
        double dx=u-t*D0, dy=v-t*D1;
        stats[1]+=1;
        if(dx*dx+dy*dy<=R*R*(1+1e-12)) { mask[index]=1; hits++; }
    }
    return hits==N*N;
}

static int face(const double *E, const double *H, const double *q,
                int K, double Lx, double Ly, double R, int N,
                int guard, std::vector<unsigned char>& mask, double *stats) {
    std::fill(mask.begin(),mask.end(),0); int hits=0;
    double det=E[0]*E[3]-E[1]*E[2];
    if(!std::isfinite(det)||std::abs(det)<1e-15) return -1;
    double I[4]={E[3]/det,-E[1]/det,-E[2]/det,E[0]/det};
    double rx=R*std::hypot(I[0],I[1])*(1+1e-12);
    double ry=R*std::hypot(I[2],I[3])*(1+1e-12);
    std::vector<double> ends((K+1)*4);
    for(int k=0;k<=K;k++) {
        double x=q[k*3], y=q[k*3+1], z=q[k*3+2];
        double a0=E[0]*x+E[1]*y+H[0]*z, a1=E[2]*x+E[3]*y+H[1]*z;
        ends[k*4]=a0;ends[k*4+1]=a1;
        ends[k*4+2]=I[0]*a0+I[1]*a1;
        ends[k*4+3]=I[2]*a0+I[3]*a1;
    }
    // Near grazing, finite central images often already prove C=1. Evaluate
    // those first; once every sample hit, omitted images cannot change union.
    for(int k=0;k<K;k++) {
        const double *a=&ends[k*4], *b=&ends[(k+1)*4];
        int cm=(int)std::llround((Lx/2-(a[2]+b[2])/2)/Lx);
        int cn=(int)std::llround((Ly/2-(a[3]+b[3])/2)/Ly);
        for(int dm=-1;dm<=1;dm++) for(int dn=-1;dn<=1;dn++)
            if(image(E,a,b,a[2],a[3],b[2],b[3],rx,ry,Lx,Ly,R,N,cm+dm,cn+dn,mask,hits,stats)) {
                stats[2]=1; return hits;
            }
    }
    for(int k=0;k<K;k++) {
        const double *a=&ends[k*4], *b=&ends[(k+1)*4];
        double minx=std::min(a[2],b[2])-rx, maxx=std::max(a[2],b[2])+rx;
        double miny=std::min(a[3],b[3])-ry, maxy=std::max(a[3],b[3])+ry;
        int m0=(int)std::ceil(-maxx/Lx-1e-10)-guard, m1=(int)std::floor((Lx-minx)/Lx+1e-10)+guard;
        int n0=(int)std::ceil(-maxy/Ly-1e-10)-guard, n1=(int)std::floor((Ly-miny)/Ly+1e-10)+guard;
        if((double)(m1-m0+1)*(n1-n0+1)>2000000) return -2;
        for(int m=m0;m<=m1;m++) for(int n=n0;n<=n1;n++)
            if(image(E,a,b,a[2],a[3],b[2],b[3],rx,ry,Lx,Ly,R,N,m,n,mask,hits,stats)) {
                stats[2]=1;return hits;
            }
    }
    return hits;
}

extern "C" __declspec(dllexport) int stocking_periodic_coverage(
    int M,const double *E,const double *H,const double *q,int K,
    double Lx,double Ly,double R,int N,int guard,int workers,
    int *counts,double *stats,unsigned char *single_mask,unsigned char *packed_masks) {
    if(M<1||K<16||N<4||N>1024||Lx<=0||Ly<=0||R<=0||guard<0) return 0;
    std::atomic<int> next{0};
    auto run=[&]() { std::vector<unsigned char> mask(N*N);
        for(;;) {int i=next.fetch_add(1); if(i>=M)break;
            counts[i]=face(E+i*4,H+i*2,q,K,Lx,Ly,R,N,guard,mask,stats+i*3);
            if(M==1&&single_mask)std::copy(mask.begin(),mask.end(),single_mask);
            if(packed_masks) {
                int bytes=(N*N+7)/8;unsigned char *out=packed_masks+(int64_t)i*bytes;
                std::fill(out,out+bytes,0);
                for(int j=0;j<N*N;j++) if(mask[j])out[j/8]|=(unsigned char)(1u<<(j%8));
            }
        }
    };
    int n=std::min(M,std::max(1,workers));std::vector<std::thread> threads;
    for(int i=1;i<n;i++)threads.emplace_back(run);run();
    for(auto& thread:threads)thread.join();return 1;
}
