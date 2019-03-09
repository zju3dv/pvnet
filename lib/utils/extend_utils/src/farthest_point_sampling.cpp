#include<vector>
#include<float.h>
#include<time.h>
#include<stdlib.h>
#include<iostream>
#include<cmath>
#include<cstring>
#include <assert.h>
#include <functional>
#include <random>

using namespace std;

struct Vec3 {
    Vec3() { x = y = z = 0; }
    Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) { }
    float x,y,z;
    Vec3 operator-(const Vec3& b) const {return {this->x-b.x,this->y-b.y,this->z-b.z};}
    Vec3 operator*(float val) const {return {this->x*val,this->y*val,this->z*val};}
    Vec3 operator/(float val) const {return (*this)*(1.f/val);}
    float operator*(const Vec3& b) {return this->x*b.x+this->y*b.y+this->z*b.z;}
    Vec3 operator-() {return {-this->x,-this->y,-this->z};}
    Vec3 operator+(const Vec3& b) { return {this->x+b.x,this->y+b.y,this->z+b.z}; }
    void operator+=(const Vec3& b) {this->x+=b.x;this->y+=b.y;this->z+=b.z;}
    float squared_norm() {return this->x*this->x+this->y*this->y+this->z*this->z;}
};
struct Vec2 {
    Vec2() { x = y = 0; }
    Vec2(float x_, float y_) : x(x_), y(y_) { }
    float x,y;
    Vec2 operator-(const Vec2& b) const {return {this->x-b.x,this->y-b.y};}
    Vec2 operator*(float val) const {return {this->x*val,this->y*val};}
    Vec2 operator/(float val) const {return (*this)*(1.f/val);}
    float operator*(const Vec2& b) {return this->x*b.x+this->y*b.y;}
    Vec2 operator-() {return {-this->x,-this->y};}
    void operator+=(const Vec2& b) {this->x+=b.x;this->y+=b.y;}
    float squared_norm() {return this->x*this->x+this->y*this->y;}
};

template<typename VecType>
inline void update_min_dist(
        const std::vector<VecType> &input_pts,
        const std::vector<bool> &mask,
        std::vector<float> &min_dist,
        int cur_idx
)
{
    for(int i=0;i<input_pts.size();i++)
    {
        if(mask[i]) continue;
        float dist=(input_pts[i]-input_pts[cur_idx]).squared_norm();
        if(dist<min_dist[i]) min_dist[i]=dist;
    }
}

inline int find_max_dist_idx(
        const std::vector<float> &min_dist,
        const std::vector<bool> &mask
)
{
    int max_idx=0;
    float max_d=0.f;
    for(int i=0;i<min_dist.size();i++)
    {
        if(mask[i]) continue;
        if(min_dist[i]>max_d)
        {
            max_idx=i;
            max_d=min_dist[i];
        }
    }
    return max_idx;
}


template<typename VecType>
void sample_farthest_points(
        const std::vector<VecType> &input_pts,
        std::vector<int> &retained_idxs,
        int sample_num
)
{
    std::vector<bool> mask(input_pts.size(),false);
    std::vector<float> min_dist(input_pts.size(),FLT_MAX);

    retained_idxs.resize(sample_num);

    // complexity o(KN) K is the sample number, N is the original number
    // 0. sample a random point
    // 1. add selected idx
    // 2. update min_dist by selected idx
    // 3. select the furthest idx and back to 1. until K points are selected.
    srand(time(0));
    int cur_idx=rand()%input_pts.size();
    for(int i=0;i<sample_num;i++)
    {
        mask[cur_idx]=true;
        retained_idxs[i]=cur_idx;
        if(i<sample_num-1)
        {
            update_min_dist(input_pts, mask, min_dist, cur_idx);
            cur_idx = find_max_dist_idx(min_dist, mask);
        }
    }
}

Vec3 max_vec3(const Vec3& v0, const Vec3& v1)
{
    float x=max(v0.x,v1.x);
    float y=max(v0.y,v1.y);
    float z=max(v0.z,v1.z);
    return Vec3(x,y,z);
}
Vec3 min_vec3(const Vec3& v0, const Vec3& v1)
{
    float x=min(v0.x,v1.x);
    float y=min(v0.y,v1.y);
    float z=min(v0.z,v1.z);
    return Vec3(x,y,z);
}

void sample_farthest_points_init_center(
        const std::vector<Vec3> &input_pts,
        std::vector<int> &retained_idxs,
        int sample_num
)
{
    std::vector<bool> mask(input_pts.size(),false);
    std::vector<float> min_dist(input_pts.size(),FLT_MAX);

    // get center
    Vec3 max_coords(-FLT_MAX,-FLT_MAX,-FLT_MAX),min_coords(FLT_MAX,FLT_MAX,FLT_MAX);
    for(int i=0;i<input_pts.size();i++)
    {
        max_coords=max_vec3(max_coords,input_pts[i]);
        min_coords=min_vec3(min_coords,input_pts[i]);
    }
    Vec3 center=(max_coords+min_coords)/2.f;
    // initialize center
    for(int i=0;i<min_dist.size();i++)
        min_dist[i]=min((input_pts[i]-center).squared_norm(),min_dist[i]);
    retained_idxs.resize(sample_num);

    // complexity o(KN) K is the sample number, N is the original number
    // 0. sample a random point
    // 1. add selected idx
    // 2. update min_dist by selected idx
    // 3. select the furthest idx and back to 1. until K points are selected.
    int cur_idx = find_max_dist_idx(min_dist, mask);
    for(int i=0;i<sample_num;i++)
    {
        mask[cur_idx]=true;
        retained_idxs[i]=cur_idx;
        if(i<sample_num-1)
        {
            update_min_dist(input_pts, mask, min_dist, cur_idx);
            cur_idx = find_max_dist_idx(min_dist, mask);
        }
    }
}

#ifdef __cplusplus
extern "C" {
#endif

void farthest_point_sampling(
    float* pts,     // [pn,3]
    int* idxs,      // [sn]
    int pn,
    int sn
)
{
    std::vector<Vec3> vec_pts(pn);
    for(int pi=0;pi<pn;pi++)
    {
        vec_pts[pi].x=pts[pi*3];
        vec_pts[pi].y=pts[pi*3+1];
        vec_pts[pi].z=pts[pi*3+2];
    }
    std::vector<int> vec_idxs;
    sample_farthest_points(vec_pts,vec_idxs,sn);

    for(int si=0;si<sn;si++) idxs[si]=vec_idxs[si];
}

void farthest_point_sampling_init_center(
    float* pts,     // [pn,3]
    int* idxs,      // [sn]
    int pn,
    int sn
)
{
    std::vector<Vec3> vec_pts(pn);
    for(int pi=0;pi<pn;pi++)
    {
        vec_pts[pi].x=pts[pi*3];
        vec_pts[pi].y=pts[pi*3+1];
        vec_pts[pi].z=pts[pi*3+2];
    }
    std::vector<int> vec_idxs;
    sample_farthest_points_init_center(vec_pts,vec_idxs,sn);

    for(int si=0;si<sn;si++) idxs[si]=vec_idxs[si];
}

#ifdef __cplusplus
}
#endif