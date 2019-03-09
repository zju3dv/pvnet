void mesh_binary_rasterization(
    float* traingles,       // [tn,3,2]
    unsigned char* mask,    // [h,w]
    int tn,
    int h,
    int w
);

void farthest_point_sampling(
    float* pts,     // [pn,3]
    int* idxs,      // [sn]
    int pn,
    int sn
);

void farthest_point_sampling_init_center(
    float* pts,     // [pn,3]
    int* idxs,      // [sn]
    int pn,
    int sn
);

void uncertainty_pnp(
    double* pts2d,  // pn,2
    double* pts3d,  // pn,3
    double* wgt2d,  // pn,3 wxx,wxy,wyy
    double* K,      // 3,3
    double* init_rt,// 6
    double* result_rt,// 6
    int pn
);

void findNearestPointIdxLauncher(
    float* ref_pts,   // [b,pn1,dim]
    float* que_pts,   // [b,pn2,dim]
    int* idxs,        // [b,pn2]
    int b,
    int pn1,
    int pn2,
    int dim,
    int exclude_self
);

//void render_depth_cffi(
//    float* RT,
//    float* K,
//    float* vert,
//    int* face,
//    float* buffer,
//    int fn, int h, int w,
//    int init
//);
//
//void render_rgb_cffi(
//    float* RT,
//    float* K,
//    float* vert,
//    float* colors,
//    int* face,
//    char* buffer,
//    int fn, int h, int w,
//    int init
//);