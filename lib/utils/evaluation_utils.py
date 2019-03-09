import time

import scipy
import sys
sys.path.append('.')

from lib.utils.config import cfg
from lib.utils.data_utils import LineModModelDB, Projector
from plyfile import PlyData
import numpy as np
import cv2
import os
import uuid

from lib.datasets.linemod_dataset import VotingType
from lib.utils.extend_utils.extend_utils import uncertainty_pnp, find_nearest_point_idx, uncertainty_pnp_v2


def pnp(points_3d, points_2d, camera_matrix,method=cv2.SOLVEPNP_ITERATIVE):
    try:
        dist_coeffs = pnp.dist_coeffs
    except:
        dist_coeffs = np.zeros(shape=[8, 1], dtype='float64')

    assert points_3d.shape[0] == points_2d.shape[0], 'points 3D and points 2D must have same number of vertices'
    if method==cv2.SOLVEPNP_EPNP:
        points_3d=np.expand_dims(points_3d, 0)
        points_2d=np.expand_dims(points_2d, 0)

    points_2d = np.ascontiguousarray(points_2d.astype(np.float64))
    points_3d = np.ascontiguousarray(points_3d.astype(np.float64))
    camera_matrix = camera_matrix.astype(np.float64)
    _, R_exp, t = cv2.solvePnP(points_3d,
                               points_2d,
                               camera_matrix,
                               dist_coeffs,
                               flags=method)
                              # , None, None, False, cv2.SOLVEPNP_UPNP)

    # R_exp, t, _ = cv2.solvePnPRansac(points_3D,
    #                            points_2D,
    #                            cameraMatrix,
    #                            distCoeffs,
    #                            reprojectionError=12.0)

    R, _ = cv2.Rodrigues(R_exp)
    # trans_3d=np.matmul(points_3d,R.transpose())+t.transpose()
    # if np.max(trans_3d[:,2]<0):
    #     R=-R
    #     t=-t

    return np.concatenate([R, t], axis=-1)

def find_nearest_point_distance(pts1,pts2):
    '''

    :param pts1:  pn1,2 or 3
    :param pts2:  pn2,2 or 3
    :return:
    '''
    idxs=find_nearest_point_idx(pts1,pts2)
    return np.linalg.norm(pts1[idxs]-pts2,2,1)

class Evaluator(object):
    def __init__(self):
        self.linemod_db = LineModModelDB()
        self.projector=Projector()
        self.projection_2d_recorder = []
        self.add_recorder = []
        self.cm_degree_5_recorder = []
        self.proj_mean_diffs=[]
        self.add_dists=[]
        self.uncertainty_pnp_cost=[]

    def projection_2d(self, pose_pred, pose_targets, model, K, threshold=5):
        model_2d_pred = self.projector.project_K(model, pose_pred, K)
        model_2d_targets = self.projector.project_K(model, pose_targets, K)
        proj_mean_diff=np.mean(np.linalg.norm(model_2d_pred - model_2d_targets, axis=-1))

        self.proj_mean_diffs.append(proj_mean_diff)
        self.projection_2d_recorder.append(proj_mean_diff < threshold)

    def projection_2d_sym(self, pose_pred, pose_targets, model, K, threshold=5):
        model_2d_pred = self.projector.project_K(model, pose_pred, K)
        model_2d_targets = self.projector.project_K(model, pose_targets, K)
        proj_mean_diff=np.mean(find_nearest_point_distance(model_2d_pred,model_2d_targets))

        self.proj_mean_diffs.append(proj_mean_diff)
        self.projection_2d_recorder.append(proj_mean_diff < threshold)

    def add_metric(self, pose_pred, pose_targets, model, diameter, percentage=0.1):
        """ ADD metric
        1. compute the average of the 3d distances between the transformed vertices
        2. pose_pred is considered correct if the distance is less than 10% of the object's diameter
        """
        diameter = diameter * percentage
        model_pred = np.dot(model, pose_pred[:, :3].T) + pose_pred[:, 3]
        model_targets = np.dot(model, pose_targets[:, :3].T) + pose_targets[:, 3]

        # from skimage.io import imsave
        # id=uuid.uuid1()
        # write_points('{}_pred.txt'.format(id),model_pred)
        # write_points('{}_targ.txt'.format(id),model_targets)
        #
        # img_pts_pred=pts_to_img_pts(model_pred,np.identity(3),np.zeros(3),self.projector.intrinsic_matrix['blender'])[0]
        # img_pts_pred=img_pts_to_pts_img(img_pts_pred,480,640).flatten()
        # img=np.zeros([480*640,3],np.uint8)
        # img_pts_targ=pts_to_img_pts(model_targets,np.identity(3),np.zeros(3),self.projector.intrinsic_matrix['blender'])[0]
        # img_pts_targ=img_pts_to_pts_img(img_pts_targ,480,640).flatten()
        # img[img_pts_pred>0]+=np.asarray([255,0,0],np.uint8)
        # img[img_pts_targ>0]+=np.asarray([0,255,0],np.uint8)
        # img=img.reshape([480,640,3])
        # imsave('{}.png'.format(id),img)

        mean_dist=np.mean(np.linalg.norm(model_pred - model_targets, axis=-1))
        self.add_recorder.append(mean_dist < diameter)
        self.add_dists.append(mean_dist)

    def add_metric_sym(self, pose_pred, pose_targets, model, diameter, percentage=0.1):
        """ ADD metric
        1. compute the average of the 3d distances between the transformed vertices
        2. pose_pred is considered correct if the distance is less than 10% of the object's diameter
        """
        diameter = diameter * percentage
        model_pred = np.dot(model, pose_pred[:, :3].T) + pose_pred[:, 3]
        model_targets = np.dot(model, pose_targets[:, :3].T) + pose_targets[:, 3]

        mean_dist=np.mean(find_nearest_point_distance(model_pred,model_targets))
        self.add_recorder.append(mean_dist < diameter)
        self.add_dists.append(mean_dist)

    def cm_degree_5_metric(self, pose_pred, pose_targets):
        """ 5 cm 5 degree metric
        1. pose_pred is considered correct if the translation and rotation errors are below 5 cm and 5 degree respectively
        """
        translation_distance = np.linalg.norm(pose_pred[:, 3] - pose_targets[:, 3]) * 100
        rotation_diff = np.dot(pose_pred[:, :3], pose_targets[:, :3].T)
        trace = np.trace(rotation_diff)
        trace = trace if trace <= 3 else 3
        angular_distance = np.rad2deg(np.arccos((trace - 1.) / 2.))
        self.cm_degree_5_recorder.append(translation_distance < 5 and angular_distance < 5)

    def evaluate(self, points_2d, pose_targets, class_type, intri_type='blender', vote_type=VotingType.BB8, intri_matrix=None):
        points_3d = VotingType.get_pts_3d(vote_type, class_type)

        if intri_type=='use_intrinsic' and intri_matrix is not None:
            K=intri_matrix
        else:
            K=self.projector.intrinsic_matrix[intri_type]

        pose_pred = pnp(points_3d, points_2d, K)
        model = self.linemod_db.get_ply_model(class_type)
        diameter = self.linemod_db.get_diameter(class_type)

        if class_type in ['eggbox','glue']:
            self.add_metric_sym(pose_pred, pose_targets, model, diameter)
        else:
            self.add_metric(pose_pred, pose_targets, model, diameter)

        self.projection_2d(pose_pred, pose_targets, model, K)
        self.cm_degree_5_metric(pose_pred, pose_targets)

        return pose_pred

    def evaluate_uncertainty(self, mean_pts2d, covar, pose_targets, class_type,
                             intri_type='blender', vote_type=VotingType.BB8,intri_matrix=None):
        points_3d=VotingType.get_pts_3d(vote_type,class_type)

        begin=time.time()
        # full
        cov_invs = []
        for vi in range(covar.shape[0]):
            if covar[vi,0,0]<1e-6 or np.sum(np.isnan(covar)[vi])>0:
                cov_invs.append(np.zeros([2,2]).astype(np.float32))
                continue

            cov_inv = np.linalg.inv(scipy.linalg.sqrtm(covar[vi]))
            cov_invs.append(cov_inv)
        cov_invs = np.asarray(cov_invs)  # pn,2,2
        weights = cov_invs.reshape([-1, 4])
        weights = weights[:, (0, 1, 3)]

        if intri_type=='use_intrinsic' and intri_matrix is not None:
            K=intri_matrix
        else:
            K=self.projector.intrinsic_matrix[intri_type]

        pose_pred = uncertainty_pnp(mean_pts2d, weights, points_3d, K)
        model = self.linemod_db.get_ply_model(class_type)
        diameter = self.linemod_db.get_diameter(class_type)
        self.uncertainty_pnp_cost.append(time.time()-begin)

        if class_type in ['eggbox','glue']:
            self.add_metric_sym(pose_pred, pose_targets, model, diameter)
        else:
            self.add_metric(pose_pred, pose_targets, model, diameter)

        self.projection_2d(pose_pred, pose_targets, model, K)
        self.cm_degree_5_metric(pose_pred, pose_targets)

        return pose_pred

    def evaluate_uncertainty_v2(self, mean_pts2d, covar, pose_targets, class_type,
                             intri_type='blender', vote_type=VotingType.BB8):
        points_3d = VotingType.get_pts_3d(vote_type, class_type)

        pose_pred = uncertainty_pnp_v2(mean_pts2d, covar, points_3d, self.projector.intrinsic_matrix[intri_type])
        model = self.linemod_db.get_ply_model(class_type)
        diameter = self.linemod_db.get_diameter(class_type)

        if class_type in ['eggbox','glue']:
            self.projection_2d_sym(pose_pred, pose_targets, model, self.projector.intrinsic_matrix[intri_type])
            self.add_metric_sym(pose_pred, pose_targets, model, diameter)
        else:
            self.projection_2d(pose_pred, pose_targets, model, self.projector.intrinsic_matrix[intri_type])
            self.add_metric(pose_pred, pose_targets, model, diameter)
        self.cm_degree_5_metric(pose_pred, pose_targets)

    def average_precision(self,verbose=True):
        np.save('tmp.npy',np.asarray(self.proj_mean_diffs))
        if verbose:
            print('2d projections metric: {}'.format(np.mean(self.projection_2d_recorder)))
            print('ADD metric: {}'.format(np.mean(self.add_recorder)))
            print('5 cm 5 degree metric: {}'.format(np.mean(self.cm_degree_5_recorder)))

        return np.mean(self.projection_2d_recorder),np.mean(self.add_recorder),np.mean(self.cm_degree_5_recorder)

