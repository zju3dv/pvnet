import cv2

import lmdb
import numpy as np
import os

from PIL import Image
from plyfile import PlyData

from lib.utils.config import cfg
from transforms3d.euler import mat2euler

import pickle


def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def save_pickle(data, pkl_path):
    os.system('mkdir -p {}'.format(os.path.dirname(pkl_path)))
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)


def read_pose(rot_path, tra_path):
    rot = np.loadtxt(rot_path, skiprows=1)
    tra = np.loadtxt(tra_path, skiprows=1) / 100.
    return np.concatenate([rot, np.reshape(tra, newshape=[3, 1])], axis=-1)


class ModelAligner(object):
    rotation_transform = np.array([[1., 0., 0.],
                                   [0., -1., 0.],
                                   [0., 0., -1.]])
    translation_transforms = {
        # 'cat': np.array([-0.00577495, -0.01259045, -0.04062323])
    }
    intrinsic_matrix = {
        'linemod': np.array([[572.4114, 0., 325.2611],
                              [0., 573.57043, 242.04899],
                              [0., 0., 1.]]),
        # 'blender': np.array([[280.0, 0.0, 128.0],
        #                      [0.0, 280.0, 128.0],
        #                      [0.0, 0.0, 1.0]]),
        'blender': np.array([[700.,    0.,  320.],
                             [0.,  700.,  240.],
                             [0.,    0.,    1.]])
    }

    def __init__(self, class_type='cat'):
        self.class_type = class_type
        self.blender_model_path = os.path.join(cfg.LINEMOD,'{}/{}.ply'.format(class_type, class_type))
        self.orig_model_path = os.path.join(cfg.LINEMOD_ORIG,'{}/mesh.ply'.format(class_type))
        self.orig_old_model_path = os.path.join(cfg.LINEMOD_ORIG,'{}/OLDmesh.ply'.format(class_type))
        self.transform_dat_path = os.path.join(cfg.LINEMOD_ORIG,'{}/transform.dat'.format(class_type))

        self.R_p2w,self.t_p2w,self.s_p2w=self.setup_p2w_transform()

    @staticmethod
    def setup_p2w_transform():
        transform1 = np.array([[0.161513626575, -0.827108919621, 0.538334608078, -0.245206743479],
                               [-0.986692547798, -0.124983474612, 0.104004733264, -0.050683632493],
                               [-0.018740313128, -0.547968924046, -0.836288750172, 0.387638419867]])
        transform2 = np.array([[0.976471602917, 0.201606079936, -0.076541729271, -0.000718327821],
                               [-0.196746662259, 0.978194475174, 0.066531419754, 0.000077120210],
                               [0.088285841048, -0.049906700850, 0.994844079018, -0.001409600372]])

        R1 = transform1[:, :3]
        t1 = transform1[:, 3]
        R2 = transform2[:, :3]
        t2 = transform2[:, 3]

        # printer system to world system
        t_p2w = np.dot(R2, t1) + t2
        R_p2w = np.dot(R2, R1)
        s_p2w = 0.85
        return R_p2w,t_p2w,s_p2w

    def pose_p2w(self,RT):
        t,R=RT[:,3],RT[:,:3]
        R_w2c=np.dot(R, self.R_p2w.T)
        t_w2c=-np.dot(R_w2c,self.t_p2w)+self.s_p2w*t
        return np.concatenate([R_w2c,t_w2c[:,None]],1)

    @staticmethod
    def load_ply_model(model_path):
        ply = PlyData.read(model_path)
        data = ply.elements[0].data
        x = data['x']
        y = data['y']
        z = data['z']
        return np.stack([x, y, z], axis=-1)

    def read_transform_dat(self):
        transform_dat = np.loadtxt(self.transform_dat_path, skiprows=1)[:, 1]
        transform_dat = np.reshape(transform_dat, newshape=[3, 4])
        return transform_dat

    def load_orig_model(self):
        if os.path.exists(self.orig_model_path):
            return self.load_ply_model(self.orig_model_path) / 1000.
        else:
            transform = self.read_transform_dat()
            old_model = self.load_ply_model(self.orig_old_model_path) / 1000.
            old_model = np.dot(old_model, transform[:, :3].T) + transform[:, 3]
            return old_model

    def get_translation_transform(self):
        if self.class_type in self.translation_transforms:
            return self.translation_transforms[self.class_type]

        blender_model = self.load_ply_model(self.blender_model_path)
        orig_model = self.load_orig_model()
        blender_model = np.dot(blender_model, self.rotation_transform.T)
        translation_transform = np.mean(orig_model, axis=0) - np.mean(blender_model, axis=0)
        self.translation_transforms[self.class_type] = translation_transform

        return translation_transform

    def align_model(self):
        blender_model = self.load_ply_model(self.blender_model_path)
        orig_model = self.load_orig_model()
        blender_model = np.dot(blender_model, self.rotation_transform.T)
        blender_model += (np.mean(orig_model, axis=0) - np.mean(blender_model, axis=0))
        np.savetxt(os.path.join(cfg.DATA_DIR, 'blender_model.txt'), blender_model)
        np.savetxt(os.path.join(cfg.DATA_DIR, 'orig_model.txt'), orig_model)

    def project_model(self, model, pose, camera_type):
        camera_points_3d = np.dot(model, pose[:, :3].T) + pose[:, 3]
        camera_points_3d = np.dot(camera_points_3d, self.intrinsic_matrix[camera_type].T)
        return camera_points_3d[:, :2] / camera_points_3d[:, 2:]

    def validate(self, idx):
        model = self.load_ply_model(self.blender_model_path)
        pose = read_pickle('/home/pengsida/Datasets/LINEMOD/renders/{}/{}_RT.pkl'.format(self.class_type, idx))['RT']
        model_2d = self.project_model(model, pose, 'blender')
        img = np.array(Image.open('/home/pengsida/Datasets/LINEMOD/renders/{}/{}.jpg'.format(self.class_type, idx)))

        import matplotlib.pyplot as plt
        plt.imshow(img)
        plt.plot(model_2d[:, 0], model_2d[:, 1], 'r.')
        plt.show()


class PoseTransformer(object):
    rotation_transform = np.array([[1., 0., 0.],
                                   [0., -1., 0.],
                                   [0., 0., -1.]])
    translation_transforms = {}
    class_type_to_number = {
        'ape': '001',
        'can': '004',
        'cat': '005',
        'driller': '006',
        'duck': '007',
        'eggbox': '008',
        'glue': '009',
        'holepuncher': '010'
    }
    blender_models={}

    def __init__(self, class_type):
        self.class_type = class_type
        self.blender_model_path = os.path.join(cfg.LINEMOD,'{}/{}.ply'.format(class_type, class_type))
        self.orig_model_path = os.path.join(cfg.LINEMOD_ORIG,'{}/mesh.ply'.format(class_type))
        self.xyz_pattern = os.path.join(cfg.OCCLUSION_LINEMOD,'models/{}/{}.xyz')
        self.model_aligner = ModelAligner(class_type)

    def orig_pose_to_blender_pose(self, pose):
        rot, tra = pose[:, :3], pose[:, 3]
        tra = tra + np.dot(rot, self.model_aligner.get_translation_transform())
        rot = np.dot(rot, self.rotation_transform)
        return np.concatenate([rot, np.reshape(tra, newshape=[3, 1])], axis=-1)

    @staticmethod
    def blender_pose_to_blender_euler(pose):
        euler = [r / np.pi * 180 for r in mat2euler(pose, axes='szxz')]
        euler[0] = -(euler[0] + 90) % 360
        euler[1] = euler[1] - 90
        return np.array(euler)

    def orig_pose_to_blender_euler(self, pose):
        blender_pose = self.orig_pose_to_blender_pose(pose)
        return self.blender_pose_to_blender_euler(blender_pose)

    @staticmethod
    def load_ply_model(model_path):
        ply = PlyData.read(model_path)
        data = ply.elements[0].data
        x = data['x']
        y = data['y']
        z = data['z']
        return np.stack([x, y, z], axis=-1)

    def get_blender_model(self):
        if self.class_type in self.blender_models:
            return self.blender_models[self.class_type]

        blender_model = self.load_ply_model(self.blender_model_path.format(self.class_type, self.class_type))
        self.blender_models[self.class_type] = blender_model

        return blender_model

    def get_translation_transform(self):
        if self.class_type in self.translation_transforms:
            return self.translation_transforms[self.class_type]

        model = self.get_blender_model()
        xyz = np.loadtxt(self.xyz_pattern.format(
            self.class_type.title(), self.class_type_to_number[self.class_type]))
        rotation = np.array([[0., 0., 1.],
                             [1., 0., 0.],
                             [0., 1., 0.]])
        xyz = np.dot(xyz, rotation.T)
        translation_transform = np.mean(xyz, axis=0) - np.mean(model, axis=0)
        self.translation_transforms[self.class_type] = translation_transform

        return translation_transform

    def occlusion_pose_to_blender_pose(self, pose):
        rot, tra = pose[:, :3], pose[:, 3]
        rotation = np.array([[0., 1., 0.],
                             [0., 0., 1.],
                             [1., 0., 0.]])
        rot = np.dot(rot, rotation)

        tra[1:] *= -1
        translation_transform = np.dot(rot, self.get_translation_transform())
        rot[1:] *= -1
        translation_transform[1:] *= -1
        tra += translation_transform
        pose = np.concatenate([rot, np.reshape(tra, newshape=[3, 1])], axis=-1)

        return pose


class Projector(object):
    intrinsic_matrix = {
        'linemod': np.array([[572.4114, 0., 325.2611],
                              [0., 573.57043, 242.04899],
                              [0., 0., 1.]]),
        'blender': np.array([[700.,    0.,  320.],
                             [0.,  700.,  240.],
                             [0.,    0.,    1.]]),
        'pascal': np.asarray([[-3000.0, 0.0, 0.0],
                              [0.0, 3000.0, 0.0],
                              [0.0,    0.0, 1.0]])
    }

    def project(self,pts_3d,RT,K_type):
        pts_2d=np.matmul(pts_3d,RT[:,:3].T)+RT[:,3:].T
        pts_2d=np.matmul(pts_2d,self.intrinsic_matrix[K_type].T)
        pts_2d=pts_2d[:,:2]/pts_2d[:,2:]
        return pts_2d

    def project_h(self,pts_3dh,RT,K_type):
        '''

        :param pts_3dh: [n,4]
        :param RT:      [3,4]
        :param K_type:
        :return: [n,3]
        '''
        K=self.intrinsic_matrix[K_type]
        return np.matmul(np.matmul(pts_3dh,RT.transpose()),K.transpose())

    def project_pascal(self,pts_3d,RT,principle):
        '''

        :param pts_3d:    [n,3]
        :param principle: [2,2]
        :return:
        '''
        K=self.intrinsic_matrix['pascal'].copy()
        K[:2,2]=principle
        cam_3d=np.matmul(pts_3d,RT[:,:3].T)+RT[:,3:].T
        cam_3d[np.abs(cam_3d[:,2])<1e-5,2]=1e-5 # revise depth
        pts_2d=np.matmul(cam_3d,K.T)
        pts_2d=pts_2d[:,:2]/pts_2d[:,2:]
        return pts_2d, cam_3d

    def project_pascal_h(self, pts_3dh,RT,principle):
        K=self.intrinsic_matrix['pascal'].copy()
        K[:2,2]=principle
        return np.matmul(np.matmul(pts_3dh,RT.transpose()),K.transpose())

    @staticmethod
    def project_K(pts_3d,RT,K):
        pts_2d=np.matmul(pts_3d,RT[:,:3].T)+RT[:,3:].T
        pts_2d=np.matmul(pts_2d,K.T)
        pts_2d=pts_2d[:,:2]/pts_2d[:,2:]
        return pts_2d


def randomly_read_background():
    cache_dir = "./data/cache"
    if os.path.exists(os.path.join(cache_dir,'background_info.pkl')):
        fns=read_pickle(os.path.join(cache_dir,'background_info.pkl'))
    else:
        fns=glob(os.path.join(background_dir,'*.jpg'))+glob(os.path.join(background_dir,'*.png'))
        save_pickle(fns,os.path.join(cache_dir,'background_info.pkl'))
    return imread(fns[np.random.randint(0,len(fns))])



def vertex_layer_reshape(vertex):
    b,vn,h,w=vertex.shape
    vertex=vertex.permute(0,2,3,1)
    vn//=2
    vertex=vertex.view(b,h,w,vn,2)
    return vertex

def mask_depth_to_point_cloud(mask,depth,K):
    ys, xs=np.nonzero(mask)
    dpts=depth[ys,xs]
    xs,ys=np.asarray(xs,np.float32),np.asarray(ys,np.float32)
    xys=np.concatenate([xs[:,None],ys[:,None]],1)
    xys*=dpts[:,None]
    xyds=np.concatenate([xys,dpts[:,None]],1)
    pts=np.matmul(xyds,np.linalg.inv(K).transpose())
    return pts

def mask_depth_to_pts(mask,depth,K,output_2d=False):
    hs,ws=np.nonzero(mask)
    pts_2d=np.asarray([ws,hs],np.float32).transpose()
    depth=depth[hs,ws]
    pts=np.asarray([ws,hs,depth],np.float32).transpose()
    pts[:,:2]*=pts[:,2:]
    if output_2d:
        return np.dot(pts,np.linalg.inv(K).transpose()), pts_2d
    else:
        return np.dot(pts,np.linalg.inv(K).transpose())
