// Perspective-correct camera raster of the authored periodic phase masks.
// The micro mask is already a geometric capsule union. This kernel selects
// the nearest macro surface and filters its binary phase in camera pixels.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <thread>
#include <vector>

extern "C" __declspec(dllexport) int stocking_camera_coverage(
    int F, const double *v, const double *q, const int *faces,
    const int *socks, const unsigned char *mapped,
    const unsigned char *left, const unsigned char *right,
    int face_count, int grid, double Lx, double Ly,
    int W, int H, int S, int workers,
    unsigned short *hits, unsigned short *mapped_counts,
    unsigned short *surface_counts, double *stats) {
    if(F<1 || W<1 || H<1 || S<1 || S>64 || grid<4 || grid>1024 ||
       grid*grid%8 || Lx<=0 || Ly<=0 || face_count<1) return 0;
    const int SW=W*S, SH=H*S, tile_size=32*S;
    const int NX=(SW+tile_size-1)/tile_size, NY=(SH+tile_size-1)/tile_size;
    const int bytes=grid*grid/8;
    std::vector<std::vector<int>> bins(NX*NY);
    std::vector<int> boxes(F*4);
    std::vector<double> denominators(F);
    for(int f=0;f<F;f++) {
        const double *a=v+f*12;
        double minx=std::min({a[0],a[4],a[8]})*S;
        double maxx=std::max({a[0],a[4],a[8]})*S;
        double miny=std::min({a[1],a[5],a[9]})*S;
        double maxy=std::max({a[1],a[5],a[9]})*S;
        int x0=std::max(0,(int)std::ceil(minx-.5-1e-9));
        int x1=std::min(SW-1,(int)std::floor(maxx-.5+1e-9));
        int y0=std::max(0,(int)std::ceil(miny-.5-1e-9));
        int y1=std::min(SH-1,(int)std::floor(maxy-.5+1e-9));
        int *box=boxes.data()+f*4;box[0]=x0;box[1]=x1;box[2]=y0;box[3]=y1;
        double d=(a[5]-a[9])*(a[0]-a[8])+(a[8]-a[4])*(a[1]-a[9]);
        denominators[f]=d;
        if(x0>x1 || y0>y1 || !std::isfinite(d) || std::abs(d)<1e-20) continue;
        for(int ty=y0/tile_size;ty<=y1/tile_size;ty++)
            for(int tx=x0/tile_size;tx<=x1/tile_size;tx++)bins[ty*NX+tx].push_back(f);
    }
    std::fill(hits,hits+W*H,0);std::fill(mapped_counts,mapped_counts+W*H,0);
    std::fill(surface_counts,surface_counts+W*H,0);
    std::atomic<int> next{0};
    std::vector<double> thread_stats(std::max(1,workers)*3,0);
    auto run=[&](int worker) {
        std::vector<double> depth(tile_size*tile_size);
        std::vector<unsigned char> hit(tile_size*tile_size), map(tile_size*tile_size), surface(tile_size*tile_size);
        for(;;) {
            int tile=next.fetch_add(1);if(tile>=NX*NY)break;
            int X0=(tile%NX)*tile_size, Y0=(tile/NX)*tile_size;
            int TW=std::min(tile_size,SW-X0),TH=std::min(tile_size,SH-Y0);
            std::fill(depth.begin(),depth.end(),std::numeric_limits<double>::infinity());
            std::fill(hit.begin(),hit.end(),0);std::fill(map.begin(),map.end(),0);std::fill(surface.begin(),surface.end(),0);
            for(int f:bins[tile]) {
                const double *a=v+f*12, *fq=q+f*6;
                const int *box=boxes.data()+f*4;
                int x0=std::max(X0,box[0]), x1=std::min(X0+TW-1,box[1]);
                int y0=std::max(Y0,box[2]), y1=std::min(Y0+TH-1,box[3]);
                double d=denominators[f];
                bool fm=mapped[f] && faces[f]>=0 && faces[f]<face_count && (socks[f]==0 || socks[f]==1);
                const unsigned char *mask=fm?((socks[f]==0?left:right)+(size_t)faces[f]*bytes):nullptr;
                for(int y=y0;y<=y1;y++)for(int x=x0;x<=x1;x++) {
                    thread_stats[worker*3]++;
                    double px=(x+.5)/S, py=(y+.5)/S;
                    double b0=((a[5]-a[9])*(px-a[8])+(a[8]-a[4])*(py-a[9]))/d;
                    double b1=((a[9]-a[1])*(px-a[8])+(a[0]-a[8])*(py-a[9]))/d;
                    double b2=1-b0-b1;
                    if(std::min({b0,b1,b2})<-1e-10)continue;
                    int i=(y-Y0)*tile_size+x-X0;
                    double z=b0*a[2]+b1*a[6]+b2*a[10];
                    if(z<-1-1e-9 || z>1+1e-9 || z>=depth[i])continue;
                    depth[i]=z;surface[i]=1;map[i]=fm;hit[i]=0;
                    thread_stats[worker*3+1]++;
                    if(!fm)continue;
                    double p0=b0*a[3],p1=b1*a[7],p2=b2*a[11],den=p0+p1+p2;
                    double qx=(p0*fq[0]+p1*fq[2]+p2*fq[4])/den;
                    double qy=(p0*fq[1]+p1*fq[3]+p2*fq[5])/den;
                    double fx=qx/Lx,fy=qy/Ly;fx-=std::floor(fx);fy-=std::floor(fy);
                    int ix=std::clamp((int)std::floor(fx*grid),0,grid-1);
                    int iy=std::clamp((int)std::floor(fy*grid),0,grid-1);
                    int bit=iy*grid+ix;hit[i]=(mask[bit/8]>>(bit%8))&1;
                    thread_stats[worker*3+2]++;
                }
            }
            for(int y=0;y<TH;y++)for(int x=0;x<TW;x++) {
                int i=y*tile_size+x, pixel=((Y0+y)/S)*W+(X0+x)/S;
                hits[pixel]+=hit[i];mapped_counts[pixel]+=map[i];surface_counts[pixel]+=surface[i];
            }
        }
    };
    workers=std::min(std::max(1,workers),NX*NY);
    std::vector<std::thread> threads;for(int i=1;i<workers;i++)threads.emplace_back(run,i);run(0);
    for(auto& t:threads)t.join();
    for(int k=0;k<3;k++){stats[k]=0;for(int i=0;i<workers;i++)stats[k]+=thread_stats[i*3+k];}
    stats[3]=NX*NY;return 1;
}
