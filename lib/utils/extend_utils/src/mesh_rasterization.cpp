#include <algorithm>
using namespace std;

inline bool same_side(
    float x0, float y0,
    float x1, float y1,
    float tx0, float ty0,
    float tx1, float ty1
)
{
    float dx=x1-x0;
    float dy=y1-y0;
    float nx=-dy;
    float ny=dx;

    float dx0=tx0-x0;
    float dy0=ty0-y0;

    float dx1=tx1-x0;
    float dy1=ty1-y0;

    float val0=dx0*nx+dy0*ny;
    float val1=dx1*nx+dy1*ny;
    return val0*val1>=0;
}

inline bool inside_triangle(
    float x0, float y0,
    float x1, float y1,
    float x2, float y2,
    float x,  float y
)
{
    return same_side(x0,y0,x1,y1,x2,y2,x,y)&&
           same_side(x1,y1,x2,y2,x0,y0,x,y)&&
           same_side(x2,y2,x0,y0,x1,y1,x,y);
}

#ifdef __cplusplus
extern "C" {
#endif

void mesh_binary_rasterization(
    float* traingles,       // [tn,3,2]
    unsigned char* mask,    // [h,w]
    int tn,
    int h,
    int w
)
{
    for(int ti=0;ti<tn;ti++)
    {
        float x0=traingles[ti*3*2+0*2+0],y0=traingles[ti*3*2+0*2+1];
        float x1=traingles[ti*3*2+1*2+0],y1=traingles[ti*3*2+1*2+1];
        float x2=traingles[ti*3*2+2*2+0],y2=traingles[ti*3*2+2*2+1];

        float minx=min({x0,x1,x2}); minx=max(0.f,minx);
        float maxx=max({x0,x1,x2}); maxx=min(float(w-2),maxx);
        float miny=min({y0,y1,y2}); miny=max(0.f,miny);
        float maxy=max({y0,y1,y2}); maxy=min(float(h-2),maxy);

        int begx=int(minx),endx=int(maxx+1.f);
        int begy=int(miny),endy=int(maxy+1.f);
        for(int yi=begy;yi<=endy;yi++)
        for(int xi=begx;xi<=endx;xi++)
        {
            if(mask[yi*w+xi]) continue;
            if(inside_triangle(x0,y0,x1,y1,x2,y2,float(xi),float(yi))) mask[yi*w+xi]=1;
        }
    }
}

#ifdef __cplusplus
}
#endif