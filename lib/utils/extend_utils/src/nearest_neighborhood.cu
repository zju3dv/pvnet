#include <float.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

int infTwoExp(int val)
{
    int inf=1;
    while(val>inf) inf<<=1;
    return inf;
}

void getGPULayout(
        int dim0,int dim1,int dim2,
        int* bdim0,int* bdim1,int* bdim2,
        int* tdim0,int* tdim1,int* tdim2
)
{
    (*tdim2)=64;
    if(dim2<(*tdim2)) (*tdim2)=infTwoExp(dim2);
    (*bdim2)=dim2/(*tdim2);
    if(dim2%(*tdim2)>0) (*bdim2)++;

    (*tdim1)=1024/(*tdim2);
    if(dim1<(*tdim1)) (*tdim1)=infTwoExp(dim1);
    (*bdim1)=dim1/(*tdim1);
    if(dim1%(*tdim1)>0) (*bdim1)++;

    (*tdim0)=1024/((*tdim1)*(*tdim2));
    if(dim0<(*tdim0)) (*tdim0)=infTwoExp(dim0);
    (*bdim0)=dim0/(*tdim0);
    if(dim0%(*tdim0)>0) (*bdim0)++;
}

__global__
void findNearestPoint3DIdxKernel(
    float* ref_pts,   // [b,pn1,3]
    float* que_pts,   // [b,pn2,3]
    int* idxs,        // [b,pn2]
    int b,
    int pn1,
    int pn2,
    int exclude_self
)
{
    int bi = threadIdx.x + blockIdx.x*blockDim.x;
    int p2i = threadIdx.y + blockIdx.y*blockDim.y;
    if(p2i>=pn2||bi>=b) return;

    float x2=que_pts[bi*pn2*3+p2i*3];
    float y2=que_pts[bi*pn2*3+p2i*3+1];
    float z2=que_pts[bi*pn2*3+p2i*3+2];
    float min_dist=FLT_MAX;
    int min_idx=0;
    for(int p1i=0;p1i<pn1;p1i++)
    {
        if(exclude_self&&p1i==p2i) continue;
        float x1=ref_pts[bi*pn1*3+p1i*3];
        float y1=ref_pts[bi*pn1*3+p1i*3+1];
        float z1=ref_pts[bi*pn1*3+p1i*3+2];

        float dist=(x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)+(z1-z2)*(z1-z2);
        if(dist<min_dist)
        {
            min_dist=dist;
            min_idx=p1i;
        }
    }
    idxs[bi*pn2+p2i]=min_idx;
}
__global__
void findNearestPoint2DIdxKernel(
    float* ref_pts,   // [b,pn1,2]
    float* que_pts,   // [b,pn2,2]
    int* idxs,        // [b,pn2]
    int b,
    int pn1,
    int pn2,
    int exclude_self
)
{
    int bi = threadIdx.x + blockIdx.x*blockDim.x;
    int p2i = threadIdx.y + blockIdx.y*blockDim.y;
    if(p2i>=pn2||bi>=b) return;

    float x2=que_pts[bi*pn2*2+p2i*2];
    float y2=que_pts[bi*pn2*2+p2i*2+1];
    float min_dist=FLT_MAX;
    int min_idx=0;
    for(int p1i=0;p1i<pn1;p1i++)
    {
        if(exclude_self&&p1i==p2i) continue;
        float x1=ref_pts[bi*pn1*2+p1i*2];
        float y1=ref_pts[bi*pn1*2+p1i*2+1];

        float dist=(x1-x2)*(x1-x2)+(y1-y2)*(y1-y2);
        if(dist<min_dist)
        {
            min_dist=dist;
            min_idx=p1i;
        }
    }
    idxs[bi*pn2+p2i]=min_idx;
}

#ifdef __cplusplus
extern "C" {
#endif

void findNearestPointIdxLauncher(
    float* ref_pts,   // [b,pn1,dim]
    float* que_pts,   // [b,pn2,dim]
    int* idxs,        // [b,pn2]
    int b,
    int pn1,
    int pn2,
    int dim,
    int exclude_self
)
{
    float* ref_pts_dev,* que_pts_dev;
    int* idxs_dev;
    gpuErrchk(cudaMalloc(&ref_pts_dev,b*pn1*sizeof(float)*dim))
    gpuErrchk(cudaMalloc(&que_pts_dev,b*pn2*sizeof(float)*dim))
    gpuErrchk(cudaMalloc(&idxs_dev,b*pn2*sizeof(int)))

    gpuErrchk(cudaMemcpy(ref_pts_dev,ref_pts,b*pn1*sizeof(float)*dim,cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(que_pts_dev,que_pts,b*pn2*sizeof(float)*dim,cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(idxs_dev,idxs,b*pn2*sizeof(int),cudaMemcpyHostToDevice))

    int bdim0,bdim1,bdim2;
    int tdim0,tdim1,tdim2;

    getGPULayout(b,pn2,1,&bdim0,&bdim1,&bdim2,&tdim0,&tdim1,&tdim2);

    dim3 bdim(bdim0,bdim1,bdim2);
    dim3 tdim(tdim0,tdim1,tdim2);

    if(dim==3)
        findNearestPoint3DIdxKernel<<<bdim,tdim>>>(ref_pts_dev,que_pts_dev,idxs_dev,b,pn1,pn2,exclude_self);
    else
        findNearestPoint2DIdxKernel<<<bdim,tdim>>>(ref_pts_dev,que_pts_dev,idxs_dev,b,pn1,pn2,exclude_self);
    gpuErrchk(cudaGetLastError())

    gpuErrchk(cudaMemcpy(idxs,idxs_dev,b*pn2*sizeof(int),cudaMemcpyDeviceToHost))
    gpuErrchk(cudaFree(ref_pts_dev))
    gpuErrchk(cudaFree(que_pts_dev))
    gpuErrchk(cudaFree(idxs_dev))

}

#ifdef __cplusplus
}
#endif
