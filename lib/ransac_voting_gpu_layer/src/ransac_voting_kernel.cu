#include "cuda_common.h"
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
// #include <ATen/Error.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>


__global__
void generate_hypothesis_kernel(
    float* direct,     // [tn,vn,2]
    float* coords,     // [tn,2]
    int* idxs,         // [hn,vn,2]
    float* hypo_pts,   // [hn,vn,2]
    int tn,
    int vn,
    int hn
)
{
    int hvi = threadIdx.x + blockIdx.x*blockDim.x;
    if(hvi>=hn*vn) return;

    int hi=hvi/vn;
    int vi=hvi-hi*vn;

    int t0=idxs[hi*vn*2+vi*2];
    int t1=idxs[hi*vn*2+vi*2+1];

    float nx0=direct[t0*vn*2+vi*2+1];
    float ny0=-direct[t0*vn*2+vi*2];
    float cx0=coords[t0*2];
    float cy0=coords[t0*2+1];

    float nx1=direct[t1*vn*2+vi*2+1];
    float ny1=-direct[t1*vn*2+vi*2];
    float cx1=coords[t1*2];
    float cy1=coords[t1*2+1];

    // compute intersection
    if(fabs(nx1*ny0-nx0*ny1)<1e-6) return;
    if(fabs(ny1*nx0-ny0*nx1)<1e-6) return;
    float y=(nx1*(nx0*cx0+ny0*cy0)-nx0*(nx1*cx1+ny1*cy1))/(nx1*ny0-nx0*ny1);
    float x=(ny1*(nx0*cx0+ny0*cy0)-ny0*(nx1*cx1+ny1*cy1))/(ny1*nx0-ny0*nx1);

    hypo_pts[hi*vn*2+vi*2]=x;
    hypo_pts[hi*vn*2+vi*2+1]=y;
}

at::Tensor generate_hypothesis_launcher(
    at::Tensor direct,     // [tn,vn,2]
    at::Tensor coords,     // [tn,2]
    at::Tensor idxs        // [hn,vn,2]
)
{
    int tn=direct.size(0);
    int vn=direct.size(1);
    int hn=idxs.size(0);

    assert(direct.size(2)==2);
    assert(coords.size(0)==tn);
    assert(coords.size(1)==2);
    assert(idxs.size(1)==vn);
    assert(idxs.size(2)==2);

    int bdim0,bdim1,bdim2;
    int tdim0,tdim1,tdim2;

    getGPULayout(hn*vn,1,1,&bdim0,&bdim1,&bdim2,&tdim0,&tdim1,&tdim2);

    dim3 bdim(bdim0,bdim1,bdim2);
    dim3 tdim(tdim0,tdim1,tdim2);

    auto hypo_pts = at::zeros({hn,vn,2}, direct.type());
    generate_hypothesis_kernel<<<bdim,tdim>>>(
        direct.data<float>(),
        coords.data<float>(),
        idxs.data<int>(),
        hypo_pts.data<float>(),
        tn,vn,hn
    );
    gpuErrchk(cudaGetLastError())

    return hypo_pts;
}

__global__
void voting_for_hypothesis_kernel(
    float* direct,     // [tn,vn,2]
    float* coords,     // [tn,2]
    float* hypo_pts,   // [hn,vn,2]
    unsigned char* inliers,     // [hn,vn,tn]
    int tn,
    int vn,
    int hn,
    float inlier_thresh
)
{
    int hi = threadIdx.x + blockIdx.x*blockDim.x;
    int vti = threadIdx.y + blockIdx.y*blockDim.y;
    if(hi>=hn||vti>=vn*tn) return;

    int vi=vti/tn;
    int ti=vti-vi*tn;

    float cx=coords[ti*2];
    float cy=coords[ti*2+1];

    float hx=hypo_pts[hi*vn*2+vi*2];
    float hy=hypo_pts[hi*vn*2+vi*2+1];

    float nx=direct[ti*vn*2+vi*2];
    float ny=direct[ti*vn*2+vi*2+1];

    float dx=hx-cx;
    float dy=hy-cy;

    float norm1=sqrt(nx*nx+ny*ny);
    float norm2=sqrt(dx*dx+dy*dy);
    if(norm1<1e-6||norm2<1e-6) return;

    float angle_dist=(dx*nx+dy*ny)/(norm1*norm2);
    if(angle_dist>inlier_thresh)
        inliers[hi*vn*tn+vi*tn+ti]=1;
}


void voting_for_hypothesis_launcher(
    at::Tensor direct,     // [tn,vn,2]
    at::Tensor coords,     // [tn,2]
    at::Tensor hypo_pts,   // [hn,vn,2]
    at::Tensor inliers,    // [hn,vn,tn]
    float inlier_thresh
)
{
    int tn=direct.size(0);
    int vn=direct.size(1);
    int hn=hypo_pts.size(0);

    assert(direct.size(2)==2);
    assert(coords.size(0)==tn);
    assert(coords.size(1)==2);
    assert(hypo_pts.size(1)==vn);
    assert(hypo_pts.size(2)==2);
    assert(inliers.size(0)==hn);
    assert(inliers.size(1)==vn);
    assert(inliers.size(2)==tn);

    int bdim0,bdim1,bdim2;
    int tdim0,tdim1,tdim2;

    getGPULayout(hn,vn*tn,1,&bdim0,&bdim1,&bdim2,&tdim0,&tdim1,&tdim2);

    dim3 bdim(bdim0,bdim1,bdim2);
    dim3 tdim(tdim0,tdim1,tdim2);


    voting_for_hypothesis_kernel<<<bdim,tdim>>>(
        direct.data<float>(),
        coords.data<float>(),
        hypo_pts.data<float>(),
        inliers.data<unsigned char>(),
        tn,vn,hn,inlier_thresh
    );
    gpuErrchk(cudaGetLastError())
}


__global__
void generate_hypothesis_vanishing_point_kernel(
    float* direct,     // [tn,vn,2]
    float* coords,     // [tn,2]
    int* idxs,         // [hn,vn,2]
    float* hypo_pts,   // [hn,vn,3]
    int tn,
    int vn,
    int hn
)
{
    int hvi = threadIdx.x + blockIdx.x*blockDim.x;
    if(hvi>=hn*vn) return;

    int hi=hvi/vn;
    int vi=hvi-hi*vn;

    int id0=idxs[hi*vn*2+vi*2];
    int id1=idxs[hi*vn*2+vi*2+1];

    // nx0*(z*cx0-x)+ny0*(z*cy0-y)=0
    float dx0=direct[id0*vn*2+vi*2];
    float dy0=direct[id0*vn*2+vi*2+1];
    float cx0=coords[id0*2];
    float cy0=coords[id0*2+1];

    float dx1=direct[id1*vn*2+vi*2];
    float dy1=direct[id1*vn*2+vi*2+1];
    float cx1=coords[id1*2];
    float cy1=coords[id1*2+1];

    float lx0=dy0;
    float ly0=-dx0;
    float lz0=cy0*dx0-cx0*dy0;

    float lx1=dy1;
    float ly1=-dx1;
    float lz1=cy1*dx1-cx1*dy1;

    // z=t0*t2 y=t1*t2 x=t3*t0
    float x=ly0*lz1-lz0*ly1;
    float y=lz0*lx1-lx0*lz1;
    float z=lx0*ly1-ly0*lx1;

    // flip direction  dx0=-ny0 dy0=nx0
    float val_x0=dx0*(x-z*cx0);
    float val_x1=dx1*(x-z*cx1);
    float val_y0=dy0*(y-z*cy0);
    float val_y1=dy1*(y-z*cy1);

    if(val_x0<0&&val_x1<0&&val_y0<0&&val_y1<0)
    { z=-z;x=-x;y=-y; }
    // if not consistent, which means two rays don't intersect
    if(val_x0*val_x1<0||val_y0*val_y1<0)
    { x=0.f;y=0.f;z=0.f; }

    hypo_pts[hi*vn*3+vi*3]=x;
    hypo_pts[hi*vn*3+vi*3+1]=y;
    hypo_pts[hi*vn*3+vi*3+2]=z;
}

at::Tensor generate_hypothesis_vanishing_point_launcher(
    at::Tensor direct,     // [tn,vn,2]
    at::Tensor coords,     // [tn,2]
    at::Tensor idxs        // [hn,vn,2]
)
{
    int tn=direct.size(0);
    int vn=direct.size(1);
    int hn=idxs.size(0);

    assert(direct.size(2)==2);
    assert(coords.size(0)==tn);
    assert(coords.size(1)==2);
    assert(idxs.size(1)==vn);
    assert(idxs.size(2)==2);

    int bdim0,bdim1,bdim2;
    int tdim0,tdim1,tdim2;

    getGPULayout(hn*vn,1,1,&bdim0,&bdim1,&bdim2,&tdim0,&tdim1,&tdim2);

    dim3 bdim(bdim0,bdim1,bdim2);
    dim3 tdim(tdim0,tdim1,tdim2);

    auto hypo_pts = at::zeros({hn,vn,3}, direct.type());
    generate_hypothesis_vanishing_point_kernel<<<bdim,tdim>>>(
        direct.data<float>(),
        coords.data<float>(),
        idxs.data<int>(),
        hypo_pts.data<float>(),
        tn,vn,hn
    );
    gpuErrchk(cudaGetLastError())

    return hypo_pts;
}

__global__
void voting_for_hypothesis_vanishing_point_kernel(
    float* direct,              // [tn,vn,2]
    float* coords,              // [tn,2]
    float* hypo_pts,            // [hn,vn,3]
    unsigned char* inliers,     // [hn,vn,tn]
    int tn,
    int vn,
    int hn,
    float inlier_thresh
)
{
    int hi = threadIdx.x + blockIdx.x*blockDim.x;
    int vti = threadIdx.y + blockIdx.y*blockDim.y;
    if(hi>=hn||vti>=vn*tn) return;

    int vi=vti/tn;
    int ti=vti-vi*tn;

    float cx=coords[ti*2];
    float cy=coords[ti*2+1];

    float hx=hypo_pts[hi*vn*3+vi*3];
    float hy=hypo_pts[hi*vn*3+vi*3+1];
    float hz=hypo_pts[hi*vn*3+vi*3+2];

    float direct_x=direct[ti*vn*2+vi*2];
    float direct_y=direct[ti*vn*2+vi*2+1];

    float diff_x=hx-cx*hz;
    float diff_y=hy-cy*hz;

    float norm1=sqrt(direct_x*direct_x+direct_y*direct_y);
    float norm2=sqrt(diff_x*diff_x+diff_y*diff_y);
    if(norm1<1e-6||norm2<1e-6) return;

    float angle_dist=(direct_x*diff_x+direct_y*diff_y)/(norm1*norm2);
    float val_x=diff_x*direct_x;
    float val_y=diff_y*direct_y;
    if(val_x<0||val_y<0) return; // the direction is wrong
    if(fabs(angle_dist)>inlier_thresh)
        inliers[hi*vn*tn+vi*tn+ti]=1;
}


void voting_for_hypothesis_vanishing_point_launcher(
    at::Tensor direct,     // [tn,vn,2]
    at::Tensor coords,     // [tn,2]
    at::Tensor hypo_pts,   // [hn,vn,3]
    at::Tensor inliers,    // [hn,vn,tn]
    float inlier_thresh
)
{
    int tn=direct.size(0);
    int vn=direct.size(1);
    int hn=hypo_pts.size(0);

    assert(direct.size(2)==2);
    assert(coords.size(0)==tn);
    assert(coords.size(1)==2);
    assert(hypo_pts.size(1)==vn);
    assert(hypo_pts.size(2)==3);
    assert(inliers.size(0)==hn);
    assert(inliers.size(1)==vn);
    assert(inliers.size(2)==tn);

    int bdim0,bdim1,bdim2;
    int tdim0,tdim1,tdim2;

    getGPULayout(hn,vn*tn,1,&bdim0,&bdim1,&bdim2,&tdim0,&tdim1,&tdim2);

    dim3 bdim(bdim0,bdim1,bdim2);
    dim3 tdim(tdim0,tdim1,tdim2);


    voting_for_hypothesis_vanishing_point_kernel<<<bdim,tdim>>>(
        direct.data<float>(),
        coords.data<float>(),
        hypo_pts.data<float>(),
        inliers.data<unsigned char>(),
        tn,vn,hn,inlier_thresh
    );
    gpuErrchk(cudaGetLastError())
}