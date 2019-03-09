import random
import time

import cv2
import sys


sys.path.append('.')
sys.path.append('..')

import numpy as np
import glob
import os
# from skimage.io import imread
from plyfile import PlyData
from PIL import Image, ImageFile
from lib.utils.config import cfg
from lib.utils.extend_utils.extend_utils import farthest_point_sampling
from lib.utils.base_utils import read_pickle, save_pickle, Projector, PoseTransformer, read_pose, ModelAligner
from scipy.misc import imread,imsave
from lib.utils.draw_utils import write_points, pts_to_img_pts, img_pts_to_pts_img


def read_rgb_np(rgb_path):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    img = Image.open(rgb_path).convert('RGB')
    img = np.array(img,np.uint8)
    return img


def read_mask_np(mask_path):
    mask = Image.open(mask_path)
    mask_seg = np.array(mask).astype(np.int32)
    return mask_seg


class LineModModelDB(object):
    '''
    LineModModelDB is used for managing the mesh of each model
    '''
    corners_3d = {}
    models = {}
    diameters = {}
    centers_3d = {}
    farthest_3d = {'8': {}, '4': {}, '12': {}, '16': {}, '20': {}}
    small_bbox_corners={}

    def __init__(self):
        self.ply_pattern = os.path.join(cfg.LINEMOD, '{}/{}.ply')
        self.diameter_pattern = os.path.join(cfg.LINEMOD_ORIG,'{}/distance.txt')
        self.farthest_pattern = os.path.join(cfg.LINEMOD,'{}/farthest{}.txt')

    def get_corners_3d(self, class_type):
        if class_type in self.corners_3d:
            return self.corners_3d[class_type]

        corner_pth=os.path.join(cfg.LINEMOD, class_type, 'corners.txt')
        if os.path.exists(corner_pth):
            self.corners_3d[class_type]=np.loadtxt(corner_pth)
            return self.corners_3d[class_type]

        ply_path = self.ply_pattern.format(class_type, class_type)
        ply = PlyData.read(ply_path)
        data = ply.elements[0].data

        x = data['x']
        min_x, max_x = np.min(x), np.max(x)
        y = data['y']
        min_y, max_y = np.min(y), np.max(y)
        z = data['z']
        min_z, max_z = np.min(z), np.max(z)
        corners_3d = np.array([
            [min_x, min_y, min_z],
            [min_x, min_y, max_z],
            [min_x, max_y, min_z],
            [min_x, max_y, max_z],
            [max_x, min_y, min_z],
            [max_x, min_y, max_z],
            [max_x, max_y, min_z],
            [max_x, max_y, max_z],
        ])
        self.corners_3d[class_type] = corners_3d
        np.savetxt(corner_pth,corners_3d)

        return corners_3d

    def get_small_bbox(self, class_type):
        if class_type in self.small_bbox_corners:
            return self.small_bbox_corners[class_type]

        corners=self.get_corners_3d(class_type)
        center=np.mean(corners,0)
        small_bbox_corners=(corners-center[None,:])*2.0/3.0+center[None,:]
        self.small_bbox_corners[class_type]=small_bbox_corners

        return small_bbox_corners

    def get_ply_model(self, class_type):
        if class_type in self.models:
            return self.models[class_type]

        ply = PlyData.read(self.ply_pattern.format(class_type, class_type))
        data = ply.elements[0].data
        x = data['x']
        y = data['y']
        z = data['z']
        model = np.stack([x, y, z], axis=-1)
        self.models[class_type] = model
        return model

    def get_diameter(self, class_type):
        if class_type in self.diameters:
            return self.diameters[class_type]

        diameter_path = self.diameter_pattern.format(class_type)
        diameter = np.loadtxt(diameter_path) / 100.
        self.diameters[class_type] = diameter
        return diameter

    def get_centers_3d(self, class_type):
        if class_type in self.centers_3d:
            return self.centers_3d[class_type]

        c3d=self.get_corners_3d(class_type)
        self.centers_3d[class_type]=(np.max(c3d,0)+np.min(c3d,0))/2
        return self.centers_3d[class_type]

    def get_farthest_3d(self,class_type,num=8):
        if class_type in self.farthest_3d['{}'.format(num)]:
            return self.farthest_3d['{}'.format(num)][class_type]

        if num==8:
            farthest_path = self.farthest_pattern.format(class_type,'')
        else:
            farthest_path = self.farthest_pattern.format(class_type,num)
        farthest_pts = np.loadtxt(farthest_path)
        self.farthest_3d['{}'.format(num)][class_type] = farthest_pts
        return farthest_pts

    @staticmethod
    def compute_farthest_surface_point_3d():
        for cls in cfg.linemod_cls_names:
            pts=np.loadtxt(os.path.join(cfg.LINEMOD, cls,'dense_pts.txt'))[:,:3]
            spts=farthest_point_sampling(pts,8,True)
            write_points(os.path.join(cfg.LINEMOD, cls, 'farthest.txt'.format(cls)),spts)

    @staticmethod
    def compute_farthest_surface_point_3d_num(num):
        for cls in cfg.linemod_cls_names:
            pts=np.loadtxt(os.path.join(cfg.LINEMOD, cls,'dense_pts.txt'))[:,:3]
            spts=farthest_point_sampling(pts,num,True)
            write_points(os.path.join(cfg.LINEMOD, cls, 'farthest{}.txt'.format(num)),spts)

    def get_ply_mesh(self,class_type):
        ply = PlyData.read(self.ply_pattern.format(class_type, class_type))
        vert = np.asarray([ply['vertex'].data['x'],ply['vertex'].data['y'],ply['vertex'].data['z']]).transpose()
        vert_id = [id for id in ply['face'].data['vertex_indices']]
        vert_id = np.asarray(vert_id,np.int64)

        return vert, vert_id

class LineModImageDB(object):
    '''

    rgb_pth relative path to cfg.LINEMOD
    dpt_pth relative path to cfg.LINEMOD
    RT np.float32 [3,4]
    cls_typ 'cat' ...
    rnd_typ 'real' or 'render'
    corner  np.float32 [8,2]
    '''
    def __init__(self, cls_name, render_num=10000, fuse_num=10000, ms_num=10000,
                 has_render_set=True, has_fuse_set=True):
        self.cls_name=cls_name

        # some dirs for processing
        os.path.join(cfg.LINEMOD,'posedb','{}_render.pkl'.format(cls_name))
        self.linemod_dir=cfg.LINEMOD
        self.render_dir='renders/{}'.format(cls_name)
        self.rgb_dir='{}/JPEGImages'.format(cls_name)
        self.mask_dir='{}/mask'.format(cls_name)
        self.rt_dir=os.path.join(cfg.DATA_DIR,'LINEMOD_ORIG',cls_name,'data')
        self.render_num=render_num

        self.test_fn='{}/test.txt'.format(cls_name)
        self.train_fn='{}/train.txt'.format(cls_name)
        self.val_fn='{}/val.txt'.format(cls_name)

        if has_render_set:
            self.render_pkl=os.path.join(self.linemod_dir,'posedb','{}_render.pkl'.format(cls_name))
            # prepare dataset
            if os.path.exists(self.render_pkl):
                # read cached
                self.render_set=read_pickle(self.render_pkl)
            else:
                # process render set
                self.render_set=self.collect_render_set_info(self.render_pkl,self.render_dir)
        else:
            self.render_set=[]

        self.real_pkl=os.path.join(self.linemod_dir,'posedb','{}_real.pkl'.format(cls_name))
        if os.path.exists(self.real_pkl):
            # read cached
            self.real_set=read_pickle(self.real_pkl)
        else:
            # process real set
            self.real_set=self.collect_real_set_info()

        # prepare train test split
        self.train_real_set=[]
        self.test_real_set=[]
        self.val_real_set=[]
        self.collect_train_val_test_info()

        self.fuse_set=[]
        self.fuse_dir='fuse'
        self.fuse_num=fuse_num
        self.cls_idx=cfg.linemod_cls_names.index(cls_name)

        if has_fuse_set:
            self.fuse_pkl=os.path.join(cfg.LINEMOD,'posedb','{}_fuse.pkl'.format(cls_name))
            # prepare dataset
            if os.path.exists(self.fuse_pkl):
                # read cached
                self.fuse_set=read_pickle(self.fuse_pkl)
            else:
                # process render set
                self.fuse_set=self.collect_fuse_info()
        else:
            self.fuse_set=[]

    def collect_render_set_info(self,pkl_file,render_dir,format='jpg'):
        database=[]
        projector=Projector()
        modeldb=LineModModelDB()
        for k in range(self.render_num):
            data={}
            data['rgb_pth']=os.path.join(render_dir,'{}.{}'.format(k,format))
            data['dpt_pth']=os.path.join(render_dir,'{}_depth.png'.format(k))
            data['RT']=read_pickle(os.path.join(self.linemod_dir,render_dir,'{}_RT.pkl'.format(k)))['RT']
            data['cls_typ']=self.cls_name
            data['rnd_typ']='render'
            data['corners']=projector.project(modeldb.get_corners_3d(self.cls_name),data['RT'],'blender')
            data['farthest']=projector.project(modeldb.get_farthest_3d(self.cls_name),data['RT'],'blender')
            data['center']=projector.project(modeldb.get_centers_3d(self.cls_name)[None,:],data['RT'],'blender')
            for num in [4,12,16,20]:
                data['farthest{}'.format(num)]=projector.project(modeldb.get_farthest_3d(self.cls_name,num),data['RT'],'blender')
            data['small_bbox'] = projector.project(modeldb.get_small_bbox(self.cls_name), data['RT'], 'blender')
            axis_direct=np.concatenate([np.identity(3), np.zeros([3, 1])], 1).astype(np.float32)
            data['van_pts']=projector.project_h(axis_direct, data['RT'], 'blender')
            database.append(data)

        save_pickle(database,pkl_file)
        return database

    def collect_real_set_info(self):
        database=[]
        projector=Projector()
        modeldb=LineModModelDB()
        img_num=len(os.listdir(os.path.join(self.linemod_dir,self.rgb_dir)))
        for k in range(img_num):
            data={}
            data['rgb_pth']=os.path.join(self.rgb_dir, '{:06}.jpg'.format(k))
            data['dpt_pth']=os.path.join(self.mask_dir, '{:04}.png'.format(k))
            pose=read_pose(os.path.join(self.rt_dir, 'rot{}.rot'.format(k)),
                           os.path.join(self.rt_dir, 'tra{}.tra'.format(k)))
            pose_transformer = PoseTransformer(class_type=self.cls_name)
            data['RT'] = pose_transformer.orig_pose_to_blender_pose(pose).astype(np.float32)
            data['cls_typ']=self.cls_name
            data['rnd_typ']='real'
            data['corners']=projector.project(modeldb.get_corners_3d(self.cls_name),data['RT'],'linemod')
            data['farthest']=projector.project(modeldb.get_farthest_3d(self.cls_name),data['RT'],'linemod')
            for num in [4,12,16,20]:
                data['farthest{}'.format(num)]=projector.project(modeldb.get_farthest_3d(self.cls_name,num),data['RT'],'linemod')
            data['center']=projector.project(modeldb.get_centers_3d(self.cls_name)[None, :],data['RT'],'linemod')
            data['small_bbox'] = projector.project(modeldb.get_small_bbox(self.cls_name), data['RT'], 'linemod')
            axis_direct=np.concatenate([np.identity(3), np.zeros([3, 1])], 1).astype(np.float32)
            data['van_pts']=projector.project_h(axis_direct, data['RT'], 'linemod')
            database.append(data)

        save_pickle(database,self.real_pkl)
        return database

    def collect_train_val_test_info(self):
        with open(os.path.join(self.linemod_dir,self.test_fn),'r') as f:
            test_fns=[line.strip().split('/')[-1] for line in f.readlines()]

        with open(os.path.join(self.linemod_dir,self.train_fn),'r') as f:
            train_fns=[line.strip().split('/')[-1] for line in f.readlines()]

        with open(os.path.join(self.linemod_dir,self.val_fn),'r') as f:
            val_fns=[line.strip().split('/')[-1] for line in f.readlines()]

        for data in self.real_set:
            if data['rgb_pth'].split('/')[-1] in test_fns:
                if data['rgb_pth'].split('/')[-1] in val_fns:
                    self.val_real_set.append(data)
                else:
                    self.test_real_set.append(data)

            if data['rgb_pth'].split('/')[-1] in train_fns:
                self.train_real_set.append(data)

    def collect_fuse_info(self):
        database=[]
        modeldb=LineModModelDB()
        projector=Projector()
        for k in range(self.fuse_num):
            data=dict()
            data['rgb_pth']=os.path.join(self.fuse_dir, '{}_rgb.jpg'.format(k))
            data['dpt_pth']=os.path.join(self.fuse_dir, '{}_mask.png'.format(k))

            # if too few foreground pts then continue
            mask=imread(os.path.join(self.linemod_dir,data['dpt_pth']))
            if np.sum(mask==(cfg.linemod_cls_names.index(self.cls_name)+1))<400: continue

            data['cls_typ']=self.cls_name
            data['rnd_typ']='fuse'
            begins,poses=read_pickle(os.path.join(self.linemod_dir,self.fuse_dir,'{}_info.pkl'.format(k)))
            data['RT'] = poses[self.cls_idx]
            K=projector.intrinsic_matrix['linemod'].copy()
            K[0,2]+=begins[self.cls_idx,1]
            K[1,2]+=begins[self.cls_idx,0]
            data['K']=K
            data['corners']=projector.project_K(modeldb.get_corners_3d(self.cls_name),data['RT'],K)
            data['center']=projector.project_K(modeldb.get_centers_3d(self.cls_name),data['RT'],K)
            data['farthest']=projector.project_K(modeldb.get_farthest_3d(self.cls_name),data['RT'],K)
            for num in [4,12,16,20]:
                data['farthest{}'.format(num)]=projector.project_K(modeldb.get_farthest_3d(self.cls_name,num),data['RT'],K)
            data['small_bbox'] = projector.project_K(modeldb.get_small_bbox(self.cls_name), data['RT'], K)
            database.append(data)

        save_pickle(database,self.fuse_pkl)
        return database

    def collect_ms_info(self):
        database=[]
        projector=Projector()
        model_db=LineModModelDB()
        for k in range(self.ms_num):
            data=dict()
            data['rgb_pth']=os.path.join(self.ms_dir, '{}.jpg'.format(k))
            data['dpt_pth']=os.path.join(self.ms_dir, '{}_{}_mask.png'.format(k,self.cls_name))

            # if too few foreground pts then continue
            mask=imread(os.path.join(self.linemod_dir,data['dpt_pth']))
            if np.sum(mask)<5: continue

            data['RT'] = read_pickle(os.path.join(self.linemod_dir, self.ms_dir, '{}_{}_RT.pkl'.format(self.cls_name,k)))['RT']
            data['cls_typ']=self.cls_name
            data['rnd_typ']='render_multi'
            data['corners']=projector.project(model_db.get_corners_3d(self.cls_name),data['RT'],'blender')
            data['farthest']=projector.project(model_db.get_farthest_3d(self.cls_name),data['RT'],'blender')
            for num in [4,12,16,20]:
                data['farthest{}'.format(num)]=projector.project(modeldb.get_farthest_3d(self.cls_name,num),data['RT'],'blender')
            data['center']=projector.project(model_db.get_centers_3d(self.cls_name)[None,:],data['RT'],'blender')
            data['small_bbox'] = projector.project(modeldb.get_small_bbox(self.cls_name), data['RT'], 'blender')
            axis_direct=np.concatenate([np.identity(3), np.zeros([3, 1])], 1).astype(np.float32)
            data['van_pts']=projector.project_h(axis_direct, data['RT'], 'blender')
            database.append(data)

        save_pickle(database,self.ms_pkl)
        return database

    def collect_printer_info(self):
        pdb=PrinterModelDB(self.cls_name)
        database=[]
        modeldb=LineModModelDB()
        for k in range(pdb.image_num):
            data={}
            data['rgb_pth']=pdb.image_pattern.format(k+1)
            data['dpt_pth']=pdb.mask_pattern.format(k+1)
            data['RT']=pdb.aligned_poses[k]
            data['K']=pdb.K[self.cls_name]
            data['cls_typ']=self.cls_name
            data['rnd_typ']='printer'
            data['corners']=Projector.project_K(modeldb.get_corners_3d(self.cls_name),data['RT'],pdb.K[self.cls_name])
            data['farthest']=Projector.project_K(modeldb.get_farthest_3d(self.cls_name),data['RT'],pdb.K[self.cls_name])
            for num in [4,12,16,20]:
                data['farthest{}'.format(num)]=Projector.project_K(modeldb.get_farthest_3d(self.cls_name,num),data['RT'],pdb.K[self.cls_name])
            data['center']=Projector.project_K(modeldb.get_centers_3d(self.cls_name)[None, :],data['RT'],pdb.K[self.cls_name])
            database.append(data)

        save_pickle(database,self.printer_pkl)
        return database

    @staticmethod
    def split_val_set():
        image_dbs=[LineModImageDB(cls,has_ms_set=False,has_fuse_set=False) for cls in cfg.linemod_cls_names]
        for db in image_dbs:
            random.shuffle(db.test_real_set)
            with open(os.path.join(db.linemod_dir,db.cls_name,'val.txt'),'w') as f:
                for k in range(len(db.test_real_set)//2):
                    f.write('LINEMOD/'+db.test_real_set[k]['rgb_pth']+'\n')

    @staticmethod
    def crop_instance(rgb,msk,size=256):
        h,w,_=rgb.shape
        hs,ws=np.nonzero(msk)
        hmin,hmax=np.min(hs),np.max(hs)
        wmin,wmax=np.min(ws),np.max(ws)
        hlen=hmax-hmin
        wlen=wmax-wmin

        type=np.random.random()
        if type<=0.4:
            # truncate x
            truncated_ratio=np.random.uniform(0.4,0.6)
            x_pt=int(wlen*truncated_ratio)+wmin
            if np.random.random()<0.5:
                wbeg = x_pt - size
                wend = x_pt
            else:
                wbeg = x_pt
                wend = x_pt + size

            if hlen<size:
                hbeg = np.random.randint(hmax-size,hmin)
                hend = hbeg + size
            else:
                hbeg = np.random.randint(hmax-size,hmin+size)
                hend = hbeg + size
        elif 0.4<type<=0.8:
            # truncate y
            truncated_ratio=np.random.uniform(0.4,0.6)
            y_pt=int(hlen*truncated_ratio)+hmin
            if np.random.random()<0.5:
                hbeg = y_pt - size
                hend = y_pt
            else:
                hbeg = y_pt
                hend = y_pt + size

            if wlen<size:
                wbeg = np.random.randint(wmax-size,wmin)
                wend = wbeg + size
            else:
                wbeg = np.random.randint(wmax-size,wmin+size)
                wend = wbeg + size
        else:
            # truncate x and y
            truncated_ratio=np.random.uniform(0.4,0.6)
            y_pt=int(hlen*truncated_ratio)+hmin
            if np.random.random()<0.5:
                hbeg = y_pt - size
                hend = y_pt
            else:
                hbeg = y_pt
                hend = y_pt + size

            truncated_ratio=np.random.uniform(0.4,0.6)
            x_pt=int(wlen*truncated_ratio)+wmin
            if np.random.random()<0.5:
                wbeg = x_pt - size
                wend = x_pt
            else:
                wbeg = x_pt
                wend = x_pt + size

        hpad_bfr=0 if hbeg>=0 else 0-hbeg
        hpad_aft=0 if hend<=h else hend-h
        wpad_bfr=0 if wbeg>=0 else 0-wbeg
        wpad_aft=0 if wend<=w else wend-w

        hbeg=hbeg if hbeg>=0 else 0
        hend=hend if hend<=h else h
        wbeg=wbeg if wbeg>=0 else 0
        wend=wend if wend<=w else w

        rgb_new=np.pad(rgb[hbeg:hend,wbeg:wend],((hpad_bfr,hpad_aft),(wpad_bfr,wpad_aft),(0,0)),mode='constant')
        msk_new=np.pad(msk[hbeg:hend,wbeg:wend],((hpad_bfr,hpad_aft),(wpad_bfr,wpad_aft)),mode='constant')

        return rgb_new,msk_new,-hbeg+hpad_bfr,-wbeg+wpad_bfr

    @staticmethod
    def make_truncated_linemod_dataset():
        for cls_name in cfg.linemod_cls_names:
            print(cls_name)
            linemod_dir=cfg.LINEMOD
            rgb_dir='{}/JPEGImages'.format(cls_name)
            mask_dir='{}/mask'.format(cls_name)
            rt_dir=os.path.join(cfg.DATA_DIR,'LINEMOD_ORIG',cls_name,'data')

            if not os.path.exists(os.path.join(linemod_dir,'truncated',cls_name)):
                os.mkdir(os.path.join(linemod_dir,'truncated',cls_name))

            projector=Projector()
            img_num=len(os.listdir(os.path.join(linemod_dir,rgb_dir)))
            print(img_num)
            for k in range(img_num):
                rgb=imread(os.path.join(linemod_dir, rgb_dir, '{:06}.jpg'.format(k)))
                msk=imread(os.path.join(linemod_dir, mask_dir, '{:04}.png'.format(k)))
                msk=(np.sum(msk,2)>0).astype(np.uint8)

                before=np.sum(msk)
                count=0
                while True:
                    rgb_new,msk_new,hbeg,wbeg=LineModImageDB.crop_instance(rgb,msk,256)
                    after=np.sum(msk_new)
                    count+=1
                    if after/before>=0.2 or count>50:
                        rgb,msk=rgb_new, msk_new
                        break

                imsave(os.path.join(linemod_dir,'truncated',cls_name,'{:06}_rgb.jpg'.format(k)),rgb)
                imsave(os.path.join(linemod_dir,'truncated',cls_name,'{:04}_msk.png'.format(k)),msk)

                pose=read_pose(os.path.join(rt_dir, 'rot{}.rot'.format(k)),
                               os.path.join(rt_dir, 'tra{}.tra'.format(k)))
                pose_transformer = PoseTransformer(class_type=cls_name)
                pose = pose_transformer.orig_pose_to_blender_pose(pose).astype(np.float32)

                K=projector.intrinsic_matrix['linemod'].copy()
                K[0,2]+=wbeg
                K[1,2]+=hbeg

                save_pickle([pose,K],os.path.join(linemod_dir,'truncated',cls_name,'{:06}_info.pkl'.format(k)))
                if k%500==0: print(k)

class SpecialDuckDataset(object):
    def __init__(self):
        pass

    @staticmethod
    def get_dataset(num=10):
        dataset=[]
        projector=Projector()
        modeldb=LineModModelDB()
        for k in range(num):
            data={}
            data['rgb_pth']='special/duck/{}.jpg'.format(k)
            data['dpt_pth']='special/duck/{}_depth.png'.format(k)
            data['RT']=read_pickle(os.path.join(cfg.LINEMOD,'special/duck/{}_RT.pkl'.format(k)))['RT']
            data['center']=projector.project(modeldb.get_centers_3d('duck'),data['RT'],'blender')
            data['rnd_typ']='render'
            dataset.append(data)
        return dataset

class OcclusionLineModImageDB(object):
    def __init__(self,cls_name):
        self.cls_name=cls_name

        # some dirs for processing
        self.linemod_dir=cfg.OCCLUSION_LINEMOD
        self.rgb_dir='RGB-D/rgb_noseg'
        self.mask_dir='masks/{}'.format(cls_name)
        self.rt_dir=os.path.join(self.linemod_dir,'poses/{}{}'.format(cls_name[0].upper(),cls_name[1:]))

        self.real_pkl=os.path.join(self.linemod_dir,'posedb','{}_real.pkl'.format(cls_name))
        if os.path.exists(self.real_pkl):
            # read cached
            self.real_set=read_pickle(self.real_pkl)
        else:
            # process real set
            self.real_set=self.collect_real_set_info()

        self.test_real_set=[]
        self.train_real_set=[]
        self.get_train_test_split()

    def get_train_test_split(self):
        test_fns=[]
        with open(os.path.join(cfg.LINEMOD,self.cls_name,'test_occlusion.txt'),'r') as f:
            for line in f.readlines():
                test_id=int(line.strip().split('/')[-1].split('.')[0])
                test_fns.append('color_{:05}.png'.format(test_id))

        # print(len(self.real_set),len(test_fns))
        for data in self.real_set:
            fn=data['rgb_pth'].split('/')[-1]
            if fn in test_fns:
                self.test_real_set.append(data)
            else:
                self.train_real_set.append(data)

    def collect_real_set_info(self):
        database=[]
        projector=Projector()
        modeldb=LineModModelDB()

        transformer=PoseTransformer(class_type=self.cls_name)

        img_num=len(os.listdir(os.path.join(self.linemod_dir,self.rgb_dir)))
        print(img_num)
        for k in range(img_num):
            data={}
            data['rgb_pth']=os.path.join(self.rgb_dir,'color_{:05}.png'.format(k))
            data['dpt_pth']=os.path.join(self.mask_dir,'{}.png'.format(k))

            pose=self.read_pose(os.path.join(self.rt_dir,'info_{:05}.txt'.format(k)))
            if len(pose)==0:
                # os.system('cp {} ./{:05}.png'.format(os.path.join(cfg.OCCLUSION_LINEMOD,data['rgb_pth']),k))
                continue
            data['RT']=transformer.occlusion_pose_to_blender_pose(pose)
            data['cls_typ']=self.cls_name
            data['rnd_typ']='real'
            data['corners']=projector.project(modeldb.get_corners_3d(self.cls_name),data['RT'],'linemod')
            data['farthest']=projector.project(modeldb.get_farthest_3d(self.cls_name),data['RT'],'linemod')
            for num in [4,12,16,20]:
                data['farthest{}'.format(num)]=projector.project(modeldb.get_farthest_3d(self.cls_name,num),data['RT'],'linemod')
            data['center']=projector.project(modeldb.get_centers_3d(self.cls_name)[None,:],data['RT'],'linemod')
            data['small_bbox'] = projector.project(modeldb.get_small_bbox(self.cls_name), data['RT'], 'linemod')
            axis_direct=np.concatenate([np.identity(3), np.zeros([3, 1])], 1).astype(np.float32)
            data['van_pts']=projector.project_h(axis_direct, data['RT'], 'blender')
            database.append(data)

        save_pickle(database,self.real_pkl)
        return database

    def get_test_val_split(self):
        with open(os.path.join(self.linemod_dir,'{}_val.txt'.format(self.cls_name)),'r') as f:
            val_fns=[line.strip() for line in f.readlines()]

        for data in self.real_set:
            if data['rgb_pth'] in val_fns:
                self.val_real_set.append(data)
            else:
                self.test_real_set.append(data)

    @staticmethod
    def read_pose(pose_path):
        with open(pose_path) as pose_info:
            lines = [line[:-1] for line in pose_info.readlines()]
            if 'rotation:' not in lines:
                return np.array([])
            row = lines.index('rotation:') + 1
            rotation = np.loadtxt(lines[row:row + 3])
            translation = np.loadtxt(lines[row + 4:row + 5])
        return np.concatenate([rotation, np.reshape(translation, newshape=[3, 1])], axis=-1)

    @staticmethod
    def split_val_set():
        image_dbs=[OcclusionLineModImageDB(cls) for cls in cfg.occ_linemod_cls_names]
        for db in image_dbs:
            random.shuffle(db.real_set)
            with open(os.path.join(db.linemod_dir,'{}_val.txt'.format(db.cls_name)),'w') as f:
                for k in range(len(db.real_set)//2):
                    f.write(db.real_set[k]['rgb_pth']+'\n')

class TruncatedLineModImageDB(object):
    def __init__(self,cls_name):
        self.cls_name=cls_name

        # some dirs for processing
        self.linemod_dir=cfg.LINEMOD

        self.pkl=os.path.join(self.linemod_dir,'posedb','{}_truncated.pkl'.format(cls_name))
        if os.path.exists(self.pkl):
            # read cached
            self.set=read_pickle(self.pkl)
        else:
            # process real set
            self.set=self.collect_truncated_set_info()

    def collect_truncated_set_info(self):
        database=[]
        projector=Projector()
        modeldb=LineModModelDB()

        img_num=len(os.listdir(os.path.join(self.linemod_dir,self.cls_name,'JPEGImages')))
        for k in range(img_num):
            data={}
            data['rgb_pth']=os.path.join('truncated',self.cls_name,'{:06}_rgb.jpg'.format(k))
            data['dpt_pth']=os.path.join('truncated',self.cls_name,'{:04}_msk.png'.format(k))

            pose,K=read_pickle(os.path.join(self.linemod_dir,'truncated',self.cls_name,'{:06}_info.pkl'.format(k)))
            data['RT']=pose
            data['cls_typ']=self.cls_name
            data['rnd_typ']='truncated'
            data['corners']=projector.project_K(modeldb.get_corners_3d(self.cls_name),data['RT'],K)
            data['farthest']=projector.project_K(modeldb.get_farthest_3d(self.cls_name),data['RT'],K)
            for num in [4,12,16,20]:
                data['farthest{}'.format(num)]=projector.project_K(modeldb.get_farthest_3d(self.cls_name,num),data['RT'],K)
            data['small_bbox'] = projector.project_K(modeldb.get_small_bbox(self.cls_name), data['RT'], K)
            data['center']=projector.project_K(modeldb.get_centers_3d(self.cls_name)[None,:],data['RT'],K)
            # axis_direct=np.concatenate([np.identity(3), np.zeros([3, 1])], 1).astype(np.float32)
            # data['van_pts']=projector.project_h(axis_direct, data['RT'], K)
            data['K']=K
            database.append(data)

        save_pickle(database,self.pkl)
        return database

class OcclusionLineModDB(LineModModelDB):
    class_type_to_number = {
        'ape': '001',
        'can': '004',
        'cat': '005',
        'driller': '006',
        'duck': '007',
        'eggbox': '008',
        'glue': '009',
        'holepuncher': '010',
        'benchvise': '011'
    }
    translation_transforms = {}
    blender_models = {}

    def __init__(self):
        super(OcclusionLineModDB, self).__init__()
        from lib.utils.render_utils import OpenGLRenderer
        self.class_type = None
        self.xyz_pattern = os.path.join(cfg.OCCLUSION_LINEMOD,'models/{}/{}.xyz')
        self.rgb_pattern = os.path.join(cfg.OCCLUSION_LINEMOD,'RGB-D/rgb_noseg/color_{:05}.png')
        self.pose_pattern = os.path.join(cfg.OCCLUSION_LINEMOD,'poses/{}/info_{:05}.txt')
        self.rgb_dir_path = os.path.join(cfg.OCCLUSION_LINEMOD,'RGB-D/rgb_noseg')
        self.mask_dir_pattern = os.path.join(cfg.OCCLUSION_LINEMOD,'masks/{}')
        self.mask_pattern = os.path.join(self.mask_dir_pattern, '{}.png')
        self.opengl_renderer = OpenGLRenderer()

    @staticmethod
    def load_ply_model(model_path):
        ply = PlyData.read(model_path)
        data = ply.elements[0].data
        x = data['x']
        y = data['y']
        z = data['z']
        return np.stack([x, y, z], axis=-1)

    @staticmethod
    def read_pose(pose_path):
        with open(pose_path) as pose_info:
            lines = [line[:-1] for line in pose_info.readlines()]
            if 'rotation:' not in lines:
                return np.array([])
            row = lines.index('rotation:') + 1
            rotation = np.loadtxt(lines[row:row + 3])
            translation = np.loadtxt(lines[row + 4:row + 5])
        return np.concatenate([rotation, np.reshape(translation, newshape=[3, 1])], axis=-1)

    def get_blender_model(self):
        if self.class_type in self.blender_models:
            return self.blender_models[self.class_type]

        blender_model = self.load_ply_model(self.ply_pattern.format(self.class_type, self.class_type))
        self.blender_models[self.class_type] = blender_model

        return blender_model

    def get_translation_transform(self):
        if self.class_type in self.translation_transforms:
            return self.translation_transforms[self.class_type]

        model = self.get_blender_model()
        xyz = np.loadtxt(self.xyz_pattern.format(self.class_type.title(), self.class_type_to_number[self.class_type]))
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

    @staticmethod
    def read_benchvise_pose(index):
        orig_pose_dir_path = os.path.join(cfg.LINEMOD_ORIG, 'benchvise/data')
        pose=read_pose(os.path.join(orig_pose_dir_path, 'rot{}.rot'.format(index)),
                       os.path.join(orig_pose_dir_path, 'tra{}.tra'.format(index)))
        pose_transformer = PoseTransformer(class_type='benchvise')
        return pose_transformer.orig_pose_to_blender_pose(pose).astype(np.float32)

    def read_blender_pose(self, index):
        if self.class_type == 'benchvise':
            return self.read_benchvise_pose(index)
        pose_path = self.pose_pattern.format(self.class_type.title(), index)
        pose = self.read_pose(pose_path)
        if len(pose) == 0:
            return np.array([])
        return self.occlusion_pose_to_blender_pose(pose)

    def get_mask_of_all_objects(self, index):
        """ get the mask of all objects
        1. initialize both mask map and depth map
        2. update the mask map and depth map for each object by order
            2.1 compute the col_row and depth of objects
            2.2 for each pixel, if object's depth is shallower than the corresponding one in the depth map, then replace the label in the mask map
        """
        mask_map = np.zeros(shape=[480, 640], dtype=np.uint8)
        depth_map = 10 * np.ones(shape=[480, 640], dtype=np.float32)

        def update(class_type):
            self.class_type = class_type
            pose = self.read_blender_pose(index)
            if len(pose) == 0:
                return
            depth = self.opengl_renderer.render(class_type, pose, camera_type='linemod')
            col_row = np.argwhere(depth != 0)[:, [1, 0]]
            depth = depth[col_row[:, 1], col_row[:, 0]]
            pixel_depth = depth_map[col_row[:, 1], col_row[:, 0]]
            inds = (depth < pixel_depth)
            selected_col_row = col_row[inds]
            selected_depth = depth[inds]

            selected_col_row = 640 * selected_col_row[:, 1] + selected_col_row[:, 0]
            inds = np.lexsort([selected_depth, selected_col_row])
            selected_col_row = selected_col_row[inds]
            selected_depth = selected_depth[inds]
            selected_col_row, inds = np.unique(selected_col_row, return_index=True)
            selected_depth = selected_depth[inds]
            selected_row = selected_col_row // 640
            selected_col = selected_col_row % 640

            mask_map[selected_row, selected_col] = int(self.class_type_to_number[class_type])
            depth_map[selected_row, selected_col] = selected_depth

        for class_type in self.class_type_to_number.keys():
            update(class_type)

        return mask_map

    def get_mask(self, index):
        """ get the mask for each object
        1. get the mask of all objects
        2. separate each object's mask from the mask map
        """
        mask_path = self.mask_pattern.format('all_objects', index)
        mask_map = self.get_mask_of_all_objects(index)
        if os.path.exists(mask_path):
            mask_map = np.array(Image.open(mask_path))
        else:
            mask_map = self.get_mask_of_all_objects(index)
            Image.fromarray(mask_map).save(mask_path)

        for class_type, class_type_num in self.class_type_to_number.items():
            mask_path = self.mask_pattern.format(class_type, index)
            class_type_num = int(class_type_num)
            mask = (mask_map == class_type_num).astype(np.uint8)
            Image.fromarray(mask).save(mask_path)

    def get_masks(self):
        """ get masks for each object in images
        1. mkdir for each category
        2. get masks for each image
        """
        mask_dir_path = self.mask_dir_pattern.format('all_objects')
        os.system('mkdir -p {}'.format(mask_dir_path))
        for class_type in self.class_type_to_number.keys():
            mask_dir_path = self.mask_dir_pattern.format(class_type)
            os.system('mkdir -p {}'.format(mask_dir_path))

        num_masks = len(os.listdir(self.rgb_dir_path))
        for i in range(num_masks):
            self.get_mask(i)

class OcclusionLineModDBSyn(OcclusionLineModDB):
    def __init__(self):
        super(OcclusionLineModDBSyn, self).__init__()
        self.pose_pattern = os.path.join(cfg.LINEMOD, 'renders/all_objects/{}_{}_RT.pkl')
        self.mask_dir_pattern = os.path.join(cfg.LINEMOD, 'renders/all_objects')
        self.mask_pattern = os.path.join(self.mask_dir_pattern, '{}_{}_mask.png')

    def read_blender_pose(self, index):
        pose_path = self.pose_pattern.format(self.class_type, index)
        return read_pickle(pose_path)['RT']

    def get_mask(self, index):
        """ get the mask for each object
        1. get the mask of all objects
        2. separate each object's mask from the mask map
        """
        mask_path = self.mask_pattern.format(index, 'all_objects')
        if os.path.exists(mask_path):
            mask_map = np.array(Image.open(mask_path))
        else:
            mask_map = self.get_mask_of_all_objects(index)
            Image.fromarray(mask_map).save(mask_path)

        for class_type, class_type_num in self.class_type_to_number.items():
            mask_path = self.mask_pattern.format(index, class_type)
            class_type_num = int(class_type_num)
            mask = (mask_map == class_type_num).astype(np.uint8)
            Image.fromarray(mask).save(mask_path)

    def get_masks(self):
        """ get masks for each object in images
        1. mkdir for each category
        2. get masks for each image
        """
        mask_dir_path = self.mask_dir_pattern
        os.system('mkdir -p {}'.format(mask_dir_path))
        for class_type in self.class_type_to_number.keys():
            mask_dir_path = self.mask_dir_pattern.format(class_type)
            os.system('mkdir -p {}'.format(mask_dir_path))

        num_masks = len(glob.glob(os.path.join(mask_dir_path, '*_depth.png')))
        for i in range(num_masks):
            self.get_mask(i)
            print('{} done'.format(i))

class YCBDB(object):
    def __init__(self, class_type):
        self.class_type = class_type
        self.data_dir_path = os.path.join(cfg.YCB, 'data')
        self.rgb_pattern = os.path.join(self.data_dir_path, '{:04}/{:06}-color.png')
        self.projector = Projector()

    def validate_pose(self):
        rgb_path = '/home/pengsida/Datasets/YCB/renders/{}/0.jpg'.format(self.class_type)
        pose_path = '/home/pengsida/Datasets/YCB/renders/{}/0_RT.pkl'.format(self.class_type)
        model_path = '/home/pengsida/Datasets/YCB/models/{}/points.xyz'.format(self.class_type)

        img = np.array(Image.open(rgb_path))
        pose = read_pickle(pose_path)['RT']
        model_3d = np.loadtxt(model_path)
        model_2d = self.projector.project(model_3d, pose, 'blender')
        import matplotlib.pyplot as plt
        plt.imshow(img)
        plt.plot(model_2d[:, 0], model_2d[:, 1], 'r.')
        plt.show()

    @staticmethod
    def get_proper_crop_size():
        mask_paths = glob.glob('/home/pengsida/Datasets/YCB/renders/003_cracker_box/*_depth.png')
        widths = []
        heights = []

        for mask_path in mask_paths:
            mask = np.array(Image.open(mask_path))
            row_col = np.argwhere(mask == 1)
            min_row, max_row = np.min(row_col[:, 0]), np.max(row_col[:, 0])
            min_col, max_col = np.min(row_col[:, 1]), np.max(row_col[:, 1])
            width = max_col - min_col
            height = max_row - min_row
            widths.append(width)
            heights.append(height)

        widths = np.array(widths)
        heights = np.array(heights)
        inds = np.lexsort([heights, widths])
        print('min width: {}, max width: {}'.format(np.min(widths), np.max(widths)))
        print('min height: {}, max height: {}'.format(np.min(heights), np.max(heights)))
        print('min size: {}, {}'.format(heights[inds[0]], widths[inds[0]]))
        print('max size: {}, {}'.format(heights[inds[-1]], widths[inds[-1]]))


class PrinterModelDB(object):
    K={
        'cat':np.asarray([[551.701,0.0,325.622],[0.0,552.732,239.996],[0.0,0.0,1.0]],np.float32)
    }
    def __init__(self,cls_name='cat'):
        self.root_dir=os.path.join(cfg.DATA_DIR, '{}_print'.format(cls_name))
        self.image_dir=os.path.join(self.root_dir,'images')
        self.image_pattern=os.path.join(self.image_dir,'{:06}_color.png')
        self.mask_pattern=os.path.join(self.image_dir,'{:06}_mask.png')

        self.printer_model_pts= np.loadtxt(os.path.join(self.root_dir, 'point_cloud.txt'))[:, :3]
        self.model_pts=LineModModelDB().get_ply_model(cls_name)
        self.poses=self.parse_poses()
        self.image_num=len(self.poses)
        self.aligned_poses=self.align_poses()

    # parse pose
    def parse_poses(self):
        RTs=[]
        with open(os.path.join(self.root_dir, 'new_camera_poses_ascii.txt'), 'r') as f:
            lines=f.readlines()
            pose_num=len(lines)//5
            for k in range(pose_num):
                cur_lines=[line.replace('\n',' ') for line in lines[k*5+1:k*5+4]]
                RT=[]
                for line in cur_lines:
                    for item in line.strip().split(' '):
                        if len(item)>0:
                            RT.append(float(item))
                RT=np.asarray(RT).reshape([3, 4])
                R=RT[:,:3].transpose()
                T=-np.dot(R,RT[:,3])
                RT=np.concatenate([R,T[:,None]],1)
                RTs.append(RT)

        return RTs

    def validate_original_poses(self):
        for k in range(0,self.image_num,20):

            rgb=imread(self.image_pattern.format(k+1))
            img_pts=Projector.project_K(self.printer_model_pts.copy(), self.poses[k], self.K['cat'])
            pts_img=img_pts_to_pts_img(img_pts,484,648)
            print(self.poses[k])
            rgb[pts_img>0]//=2
            rgb[pts_img>0]+=np.asarray([127,0,0],np.uint8)

            plt.imshow(rgb)
            plt.show()

    def generate_mask_image(self):
        from lib.utils.draw_utils import img_pts_to_pts_img
        for k in range(0,self.image_num):
            img_pts=Projector.project_K(self.printer_model_pts.copy(), self.poses[k], self.K['cat'])
            pts_img=img_pts_to_pts_img(img_pts,484,648)
            imsave(self.mask_pattern.format(k+1),pts_img.astype(np.uint8))

    def validate_aligned_poses(self):
        aligner=ModelAligner()
        for k in range(0,self.image_num,20):

            rgb=imread(self.image_pattern.format(k+1))
            pose_aligned=aligner.pose_p2w(self.poses[k])
            img_pts=Projector.project_K(self.model_pts.copy(), pose_aligned, self.K['cat'])
            pts_img=img_pts_to_pts_img(img_pts,484,648)
            rgb[pts_img>0]//=2
            rgb[pts_img>0]+=np.asarray([127,0,0],np.uint8)

            plt.imshow(rgb)
            plt.show()

    def align_poses(self):
        aligner=ModelAligner()
        poses=[]
        for k in range(0,self.image_num):
            pose_aligned=aligner.pose_p2w(self.poses[k])
            poses.append(pose_aligned)

        return poses


if __name__=="__main__":
    LineModModelDB.compute_farthest_surface_point_3d()
    LineModModelDB.compute_farthest_surface_point_3d_num(4)
    LineModModelDB.compute_farthest_surface_point_3d_num(12)
    LineModModelDB.compute_farthest_surface_point_3d_num(16)
    LineModModelDB.compute_farthest_surface_point_3d_num(20)

