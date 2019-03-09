import sys
sys.path.append('.')
from lib.utils.extend_utils._extend_utils import lib, ffi
import numpy as np
import cv2

def mesh_binary_rasterization(triangles_2d,h,w):
    tn,_,_=triangles_2d.shape
    assert(triangles_2d.shape[1]==3)
    assert(triangles_2d.shape[2]==2)

    mask=np.ascontiguousarray(np.zeros([h,w],np.uint8))
    triangles_2d=np.ascontiguousarray(triangles_2d,np.float32)

    mask_ptr=ffi.cast('unsigned char *',mask.ctypes.data)
    triangles_2d_ptr=ffi.cast('float *',triangles_2d.ctypes.data)

    lib.mesh_binary_rasterization(triangles_2d_ptr,mask_ptr,tn,h,w)

    return mask

def farthest_point_sampling(pts,sn,init_center=False):
    pn,_=pts.shape
    assert(pts.shape[1]==3)

    pts=np.ascontiguousarray(pts,np.float32)
    idxs=np.ascontiguousarray(np.zeros([sn],np.int32))

    pts_ptr=ffi.cast('float*',pts.ctypes.data)
    idxs_ptr=ffi.cast('int*',idxs.ctypes.data)

    if init_center:
        lib.farthest_point_sampling_init_center(pts_ptr, idxs_ptr, pn, sn)
    else:
        lib.farthest_point_sampling(pts_ptr,idxs_ptr,pn,sn)

    return pts[idxs]

def find_nearest_point_idx(ref_pts,que_pts):
    '''
    for every point in que_pts, find the nearest point in ref_pts
    :param ref_pts:  pn1,3 or 2
    :param que_pts:  pn2,3 or 2
    :return:  idxs pn2
    '''
    assert(ref_pts.shape[1]==que_pts.shape[1] and 1<que_pts.shape[1]<=3)
    pn1=ref_pts.shape[0]
    pn2=que_pts.shape[0]
    dim=ref_pts.shape[1]

    ref_pts=np.ascontiguousarray(ref_pts[None,:,:],np.float32)
    que_pts=np.ascontiguousarray(que_pts[None,:,:],np.float32)
    idxs=np.zeros([1,pn2],np.int32)

    ref_pts_ptr=ffi.cast('float *',ref_pts.ctypes.data)
    que_pts_ptr=ffi.cast('float *',que_pts.ctypes.data)
    idxs_ptr=ffi.cast('int *',idxs.ctypes.data)
    lib.findNearestPointIdxLauncher(ref_pts_ptr,que_pts_ptr,idxs_ptr,1,pn1,pn2,dim,0)

    return idxs[0]


def uncertainty_pnp(points_2d, weights_2d, points_3d, camera_matrix):
    '''
    :param points_2d:           [pn,2]
    :param weights_2d:          [pn,3] wxx,wxy,wyy
    :param points_3d:           [pn,3]
    :param camera_matrix:       [3,3]
    :return:
    '''
    pn=points_2d.shape[0]
    assert(points_3d.shape[0]==pn and pn>=4)

    try:
        dist_coeffs = uncertainty_pnp.dist_coeffs
    except:
        dist_coeffs = np.zeros(shape=[8, 1], dtype=np.float64)

    points_3d=points_3d.astype(np.float64)
    points_2d=points_2d.astype(np.float64)
    weights_2d=weights_2d.astype(np.float64)
    camera_matrix=camera_matrix.astype(np.float64)

    idxs=np.argsort(weights_2d[:,0]+weights_2d[:,1])[-4:]

    _, R_exp, t=cv2.solvePnP(np.expand_dims(points_3d[idxs,:],0),
                             np.expand_dims(points_2d[idxs,:],0),
                             camera_matrix,dist_coeffs,None,None,False,flags=cv2.SOLVEPNP_P3P)

    if pn==4:
        # no other points
        R, _ = cv2.Rodrigues(R_exp)
        Rt=np.concatenate([R, t], axis=-1)
        return Rt

    points_2d=np.ascontiguousarray(points_2d,np.float64)
    points_3d=np.ascontiguousarray(points_3d,np.float64)
    weights_2d=np.ascontiguousarray(weights_2d,np.float64)
    camera_matrix=np.ascontiguousarray(camera_matrix,np.float64)
    init_rt=np.ascontiguousarray(np.concatenate([R_exp,t],0),np.float64)

    points_2d_ptr=ffi.cast('double*',points_2d.ctypes.data)
    points_3d_ptr=ffi.cast('double*',points_3d.ctypes.data)
    weights_3d_ptr=ffi.cast('double*',weights_2d.ctypes.data)
    camera_matrix_ptr=ffi.cast('double*',camera_matrix.ctypes.data)
    init_rt_ptr=ffi.cast('double*',init_rt.ctypes.data)
    result_rt=np.empty([6],np.float64)
    result_rt_ptr=ffi.cast('double*',result_rt.ctypes.data)

    lib.uncertainty_pnp(points_2d_ptr,points_3d_ptr,weights_3d_ptr,camera_matrix_ptr,init_rt_ptr,result_rt_ptr,pn)

    R, _ = cv2.Rodrigues(result_rt[:3])
    Rt = np.concatenate([R, result_rt[3:,None]], axis=-1)
    return Rt

def uncertainty_pnp_v2(points_2d, covars, points_3d, camera_matrix, type='single'):
    '''
    :param points_2d:           [pn,2]
    :param covars:              [pn,2,2]
    :param points_3d:           [pn,3]
    :param camera_matrix:       [3,3]
    :return:
    '''
    pn=points_2d.shape[0]
    assert(points_3d.shape[0]==pn and pn>=4 and covars.shape[0]==pn)

    points_3d=points_3d.astype(np.float64)
    points_2d=points_2d.astype(np.float64)
    camera_matrix=camera_matrix.astype(np.float64)

    weights_2d=[]
    for pi in range(pn):
        weight=0.0
        if covars[pi,0,0]<1e-5:
            weights_2d.append(weight)
        else:
            weight=np.max(np.linalg.eigvals(covars[pi]))
            weights_2d.append(1.0/weight)
    weights_2d=np.asarray(weights_2d,np.float64)

    try:
        dist_coeffs = uncertainty_pnp.dist_coeffs
    except:
        dist_coeffs = np.zeros(shape=[8, 1], dtype=np.float64)

    idxs=np.argsort(weights_2d)[-4:]
    _, R_exp, t=cv2.solvePnP(np.expand_dims(points_3d[idxs,:],0),
                             np.expand_dims(points_2d[idxs,:],0),
                             camera_matrix,dist_coeffs,None,None,False,flags=cv2.SOLVEPNP_P3P)

    if pn==4:
        # no other points
        R, _ = cv2.Rodrigues(R_exp)
        Rt=np.concatenate([R, t], axis=-1)
        return Rt

    points_2d=np.ascontiguousarray(points_2d,np.float64)
    points_3d=np.ascontiguousarray(points_3d,np.float64)
    weights_2d=weights_2d[:,None]
    weights_2d=np.concatenate([weights_2d,np.zeros([pn,1]),weights_2d],1)
    weights_2d=np.ascontiguousarray(weights_2d,np.float64)
    camera_matrix=np.ascontiguousarray(camera_matrix,np.float64)
    init_rt=np.ascontiguousarray(np.concatenate([R_exp,t],0),np.float64)

    points_2d_ptr=ffi.cast('double*',points_2d.ctypes.data)
    points_3d_ptr=ffi.cast('double*',points_3d.ctypes.data)
    weights_3d_ptr=ffi.cast('double*',weights_2d.ctypes.data)
    camera_matrix_ptr=ffi.cast('double*',camera_matrix.ctypes.data)
    init_rt_ptr=ffi.cast('double*',init_rt.ctypes.data)
    result_rt=np.empty([6],np.float64)
    result_rt_ptr=ffi.cast('double*',result_rt.ctypes.data)

    lib.uncertainty_pnp(points_2d_ptr,points_3d_ptr,weights_3d_ptr,camera_matrix_ptr,init_rt_ptr,result_rt_ptr,pn)

    R, _ = cv2.Rodrigues(result_rt[:3])
    Rt = np.concatenate([R, result_rt[3:,None]], axis=-1)
    return Rt



def post_refinement(mask,pose,K,pts):
    '''
    find_edge: mask->edge cv2.findContours
    find_silhouette_pts: vertices + triangle + pose + K
                        ->render depth map, find silhouette pixel and project back
    find_nearest_pairs: projected pixel + contour pixel
    optimize:
    :param mask: [h,w]
    :param pose: [3,4]
    :param K:    [3,3]
    :param pts:  [pn,3]
    :return:
    '''
    pass

def render_mesh_depth(RT,K,vert,face,h,w,init):
    fn=face.shape[0]

    RT=np.ascontiguousarray(RT,np.float32)
    K=np.ascontiguousarray(K,np.float32)
    vert=np.ascontiguousarray(vert,np.float32)
    face=np.ascontiguousarray(face,np.int32)
    img=np.ascontiguousarray(np.zeros([h,w],np.float32))

    RT_ptr=ffi.cast('float *',RT.ctypes.data)
    K_ptr=ffi.cast('float *',K.ctypes.data)
    vert_ptr=ffi.cast('float *',vert.ctypes.data)
    face_ptr=ffi.cast('int *',face.ctypes.data)
    img_ptr=ffi.cast('float *',img.ctypes.data)
    init_val=1 if init else 0
    lib.render_depth_cffi(RT_ptr,K_ptr,vert_ptr,face_ptr,img_ptr,fn,h,w,init_val)

    return img

def render_mesh_rgb(RT,K,vert,colors,face,h,w,init):
    fn=face.shape[0]

    RT=np.ascontiguousarray(RT,np.float32)
    K=np.ascontiguousarray(K,np.float32)
    vert=np.ascontiguousarray(vert,np.float32)
    colors=np.ascontiguousarray(colors,np.float32)
    face=np.ascontiguousarray(face,np.int32)
    img=np.ascontiguousarray(np.zeros([h,w,4],np.uint8))

    RT_ptr=ffi.cast('float *',RT.ctypes.data)
    K_ptr=ffi.cast('float *',K.ctypes.data)
    vert_ptr=ffi.cast('float *',vert.ctypes.data)
    colors_ptr=ffi.cast('float *',colors.ctypes.data)
    face_ptr=ffi.cast('int *',face.ctypes.data)
    img_ptr=ffi.cast('char *',img.ctypes.data)
    init_val=1 if init else 0
    lib.render_rgb_cffi(RT_ptr,K_ptr,vert_ptr,colors_ptr,face_ptr,img_ptr,fn,h,w,init_val)

    return img

if __name__=="__main__":
    import sys
    sys.path.append('.')
    from lib.utils.data_utils import LineModImageDB,LineModModelDB,Projector
    from lib.datasets.linemod_dataset import LineModDatasetRealAug,ImageSizeBatchSampler,VotingType
    from lib.ransac_voting_gpu_layer.ransac_voting_gpu import ransac_voting_layer
    from torch.utils.data import RandomSampler,DataLoader
    from lib.utils.draw_utils import pts_to_img_pts
    from lib.utils.evaluation_utils import pnp
    import random

    image_db = LineModImageDB('duck', has_ro_set=False, has_ra_set=False, has_plane_set=False, has_render_set=False,
                              has_ms_set=False,has_fuse_set=False)
    random.shuffle(image_db.real_set)
    dataset = LineModDatasetRealAug(image_db.real_set[:5], data_prefix=image_db.linemod_dir,
                                    vote_type=VotingType.Extreme, augment=False)
    sampler = RandomSampler(dataset)
    batch_sampler = ImageSizeBatchSampler(sampler, 5, False)
    loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=8)
    modeldb=LineModModelDB()
    camera_matrix=Projector().intrinsic_matrix['linemod'].astype(np.float32)
    for i, data in enumerate(loader):
        rgb, mask, vertex, vertex_weight, pose, gt_corners = data
        pts2d=gt_corners[0].numpy()[:,:2].astype(np.float32)

        pts3d=modeldb.get_extreme_3d('duck')
        pts3d=np.concatenate([pts3d,modeldb.get_centers_3d('duck')[None,:]],0).astype(np.float32)
        wgt2d=np.zeros([pts2d.shape[0],3]).astype(np.float32)
        wgt2d[:,(0,2)]=1.0

        for k in range(pts2d.shape[0]):
            if np.random.random()<0.5:
                scale = np.random.uniform(1, 8)
            else:
                scale = np.random.uniform(32, 48)
            pts2d[k]+=np.random.normal(0,scale,2)
            wgt2d[k,(0,2)]=1/scale
        wgt2d/=wgt2d.max()

        pose_pred=uncertainty_pnp(pts2d,wgt2d,pts3d,camera_matrix)
        pose_pred2=pnp(pts3d,pts2d,camera_matrix)

        pts2d1,_=pts_to_img_pts(pts3d,pose_pred[:,:3],pose_pred[:,3],camera_matrix)
        pts2d2,_=pts_to_img_pts(pts3d,pose_pred2[:,:3],pose_pred2[:,3],camera_matrix)

        residual1=np.mean(np.abs(pts2d1-pts2d))
        residual2=np.mean(np.abs(pts2d2-pts2d))

        print(residual1,residual2)
        pose=pose.numpy()
        print(np.mean(np.abs(pose-pose_pred)))
        print(np.mean(np.abs(pose-pose_pred2)))
