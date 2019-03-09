import random
import sys
import time

sys.path.append('.')

from lib.datasets.augmentation import blur_image, mask_out_instance, rotate_instance, crop_resize_instance_v2, \
    crop_resize_instance_v1, crop_or_padding_to_fixed_size, flip
from lib.utils.config import cfg
from lib.utils.data_utils import read_pickle, LineModImageDB, read_rgb_np, read_mask_np, OcclusionLineModImageDB, \
    LineModModelDB
import os
import torch
from torch.utils.data import Dataset, Sampler
import torchvision.transforms as transforms
import cv2
import numpy as np
import glob
from PIL import Image
import json

from lib.utils.draw_utils import visualize_bounding_box, visualize_vanishing_points, visualize_points, imagenet_to_uint8


def read_rgb(rgb_path):
    img = Image.open(rgb_path).convert('RGB')
    img = np.array(img)
    return torch.from_numpy(img).float().permute(2, 0, 1)


def read_mask(mask_path):
    mask = Image.open(mask_path).convert('1')
    mask_seg = np.array(mask).astype(np.int32)
    return torch.from_numpy(mask_seg).long()


def read_vertex(vertex_path):
    vertex = read_pickle(vertex_path)
    return torch.from_numpy(vertex).float().permute(2, 0, 1)


def read_pose(pose_path):
    pose = read_pickle(pose_path)['RT']
    return torch.from_numpy(pose).float()


###### read functions ######
def read_corner_np(corner_path):
    return np.loadtxt(corner_path)[1:].reshape([-1,2])[1:9]


##### transformation functions #######
def compute_vertex(mask, center_2d):
    xy = np.argwhere(mask == 1)[:, [1, 0]]
    vertex = center_2d[:, np.newaxis, :] - xy
    norm = np.linalg.norm(vertex, axis=2, keepdims=True)
    norm[norm<1e-3] += 1e-3
    vertex = vertex / norm

    vertex_x = np.tile(mask, reps=[8, 1, 1]).astype(np.float32)
    vertex_x[:, xy[:, 1], xy[:, 0]] = vertex[..., 0]
    vertex_y = np.tile(mask, reps=[8, 1, 1]).astype(np.float32)
    vertex_y[:, xy[:, 1], xy[:, 0]] = vertex[..., 1]
    vertex = np.stack([vertex_x, vertex_y], axis=-1)

    return np.reshape(np.transpose(vertex, axes=[1, 2, 0, 3]), newshape=[mask.shape[0], mask.shape[1], -1])

def compute_vertex_hcoords(mask, hcoords, use_motion=False):
    h,w=mask.shape
    m=hcoords.shape[0]
    xy = np.argwhere(mask == 1)[:, [1, 0]]
    vertex = xy[:, None, :]*hcoords[None, :, 2:]
    vertex = hcoords[None, :, :2] - vertex
    if not use_motion:
        norm = np.linalg.norm(vertex, axis=2, keepdims=True)
        norm[norm<1e-3] += 1e-3
        vertex = vertex / norm

    vertex_out=np.zeros([h,w,m,2],np.float32)
    vertex_out[xy[:, 1], xy[:, 0]] = vertex
    return  np.reshape(vertex_out,[h,w,m*2])

class VotingType:
    BB8=0
    BB8C=1
    BB8S=2
    VanPts=3
    Farthest=5
    Farthest4=6
    Farthest12=7
    Farthest16=8
    Farthest20=9

    @staticmethod
    def get_data_pts_2d(vote_type,data):
        if vote_type==VotingType.BB8:
            cor = data['corners'].copy()  # note the copy here!!!
            hcoords=np.concatenate([cor,np.ones([8,1],np.float32)],1) # [8,3]
        elif vote_type==VotingType.BB8C:
            cor = data['corners'].copy()
            cen = data['center'].copy()
            hcoords = np.concatenate([cor,cen],0)
            hcoords = np.concatenate([hcoords,np.ones([9,1],np.float32)],1)
        elif vote_type==VotingType.BB8S:
            cor = data['small_bbox'].copy()
            cen = data['center'].copy()
            hcoords = np.concatenate([cor,cen],0)
            hcoords = np.concatenate([hcoords,np.ones([9,1],np.float32)],1)
        elif vote_type==VotingType.VanPts:
            cen = data['center'].copy()
            van = data['van_pts'].copy()
            hcoords = np.concatenate([cen,np.ones([1,1],np.float32)],1)
            hcoords = np.concatenate([van,hcoords],0)
        elif vote_type==VotingType.Farthest:
            cen = data['center'].copy()
            far = data['farthest'].copy()
            hcoords = np.concatenate([far,cen],0)
            hcoords = np.concatenate([hcoords,np.ones([hcoords.shape[0],1],np.float32)],1)
        elif vote_type==VotingType.Farthest4:
            cen = data['center'].copy()
            far = data['farthest4'].copy()
            hcoords = np.concatenate([far,cen],0)
            hcoords = np.concatenate([hcoords,np.ones([hcoords.shape[0],1],np.float32)],1)
        elif vote_type==VotingType.Farthest12:
            cen = data['center'].copy()
            far = data['farthest12'].copy()
            hcoords = np.concatenate([far,cen],0)
            hcoords = np.concatenate([hcoords,np.ones([hcoords.shape[0],1],np.float32)],1)
        elif vote_type==VotingType.Farthest16:
            cen = data['center'].copy()
            far = data['farthest16'].copy()
            hcoords = np.concatenate([far,cen],0)
            hcoords = np.concatenate([hcoords,np.ones([hcoords.shape[0],1],np.float32)],1)
        elif vote_type==VotingType.Farthest20:
            cen = data['center'].copy()
            far = data['farthest20'].copy()
            hcoords = np.concatenate([far,cen],0)
            hcoords = np.concatenate([hcoords,np.ones([hcoords.shape[0],1],np.float32)],1)

        return hcoords

    @staticmethod
    def get_pts_3d(vote_type,class_type):
        linemod_db=LineModModelDB()
        if vote_type==VotingType.BB8C:
            points_3d = linemod_db.get_corners_3d(class_type)
            points_3d = np.concatenate([points_3d,linemod_db.get_centers_3d(class_type)[None,:]],0)
        elif vote_type==VotingType.BB8S:
            points_3d = linemod_db.get_small_bbox(class_type)
            points_3d = np.concatenate([points_3d,linemod_db.get_centers_3d(class_type)[None,:]],0)
        elif vote_type==VotingType.Farthest:
            points_3d = linemod_db.get_farthest_3d(class_type)
            points_3d = np.concatenate([points_3d, linemod_db.get_centers_3d(class_type)[None, :]], 0)
        elif vote_type==VotingType.Farthest4:
            points_3d = linemod_db.get_farthest_3d(class_type,4)
            points_3d = np.concatenate([points_3d, linemod_db.get_centers_3d(class_type)[None, :]], 0)
        elif vote_type==VotingType.Farthest12:
            points_3d = linemod_db.get_farthest_3d(class_type,12)
            points_3d = np.concatenate([points_3d, linemod_db.get_centers_3d(class_type)[None, :]], 0)
        elif vote_type==VotingType.Farthest16:
            points_3d = linemod_db.get_farthest_3d(class_type,16)
            points_3d = np.concatenate([points_3d, linemod_db.get_centers_3d(class_type)[None, :]], 0)
        elif vote_type==VotingType.Farthest20:
            points_3d = linemod_db.get_farthest_3d(class_type,20)
            points_3d = np.concatenate([points_3d, linemod_db.get_centers_3d(class_type)[None, :]], 0)
        else: # BB8
            points_3d = linemod_db.get_corners_3d(class_type)

        return points_3d

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'default_linemod_cfg.json'),'r') as f:
    default_aug_cfg=json.load(f)

class LineModDatasetRealAug(Dataset):
    def __init__(self, imagedb, data_prefix=cfg.LINEMOD, vote_type=VotingType.BB8,
                 augment=False, cfg=default_aug_cfg, background_mask_out=False, use_intrinsic=False,
                 use_motion=False):
        self.imagedb=imagedb
        self.augment=augment
        self.background_mask_out=background_mask_out
        self.use_intrinsic=use_intrinsic
        self.use_motion=use_motion
        self.cfg=cfg

        self.img_transforms=transforms.Compose([
            transforms.ColorJitter(self.cfg['brightness'],self.cfg['contrast'],self.cfg['saturation'],self.cfg['hue']),
            transforms.ToTensor(), # if image.dtype is np.uint8, then it will be divided by 255
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.test_img_transforms=transforms.Compose([
            transforms.ToTensor(), # if image.dtype is np.uint8, then it will be divided by 255
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.vote_type=vote_type
        self.data_prefix=data_prefix

    def __getitem__(self, index_tuple):
        index, height, width = index_tuple

        rgb_path = os.path.join(self.data_prefix,self.imagedb[index]['rgb_pth'])
        mask_path = os.path.join(self.data_prefix,self.imagedb[index]['dpt_pth'])

        pose = self.imagedb[index]['RT'].copy()
        rgb = read_rgb_np(rgb_path)
        mask = read_mask_np(mask_path)
        if self.imagedb[index]['rnd_typ']=='real' and len(mask.shape)==3:
            mask=np.sum(mask,2)>0
            mask=np.asarray(mask,np.int32)

        if self.imagedb[index]['rnd_typ']=='fuse':
            mask=np.asarray(mask==(cfg.linemod_cls_names.index(self.imagedb[index]['cls_typ'])+1),np.int32)

        hcoords=VotingType.get_data_pts_2d(self.vote_type,self.imagedb[index])

        if self.use_intrinsic:
            K = torch.tensor(self.imagedb[index]['K'].astype(np.float32))

        if self.augment:
            rgb, mask, hcoords = self.augmentation(rgb, mask, hcoords, height, width)

        ver = compute_vertex_hcoords(mask, hcoords, self.use_motion)
        ver=torch.tensor(ver, dtype=torch.float32).permute(2, 0, 1)
        mask=torch.tensor(np.ascontiguousarray(mask),dtype=torch.int64)
        ver_weight=mask.unsqueeze(0).float()

        if self.augment: # and self.imagedb[index]['rnd_typ']!='real':
            # if not real and do augmentation then jitter color
            if self.cfg['blur'] and np.random.random()<0.5:
                blur_image(rgb,np.random.choice([3,5,7,9]))
            if self.cfg['jitter']:
                rgb=self.img_transforms(Image.fromarray(np.ascontiguousarray(rgb, np.uint8)))
            else:
                rgb=self.test_img_transforms(Image.fromarray(np.ascontiguousarray(rgb, np.uint8)))
            if self.cfg['use_mask_out'] and np.random.random()<0.1:
                rgb *= (mask[None, :, :]).float()
        else:
            rgb=self.test_img_transforms(Image.fromarray(np.ascontiguousarray(rgb, np.uint8)))

        if self.imagedb[index]['rnd_typ']=='fuse' and self.cfg['ignore_fuse_ms_vertex']: ver_weight*=0.0

        pose=torch.tensor(pose.astype(np.float32))
        hcoords=torch.tensor(hcoords.astype(np.float32))
        if self.use_intrinsic:
            return rgb, mask, ver, ver_weight, pose, hcoords, K
        else:
            return rgb, mask, ver, ver_weight, pose, hcoords

    def __len__(self):
        return len(self.imagedb)

    def augmentation(self, img, mask, hcoords, height, width):
        foreground=np.sum(mask)
        # randomly mask out to add occlusion
        if self.cfg['mask'] and np.random.random() < 0.5:
            img, mask = mask_out_instance(img, mask, self.cfg['min_mask'], self.cfg['max_mask'])

        if foreground>0:
            # randomly rotate around the center of the instance
            if self.cfg['rotation']:
                img, mask, hcoords = rotate_instance(img, mask, hcoords, self.cfg['rot_ang_min'], self.cfg['rot_ang_max'])

            # randomly crop and resize
            if self.cfg['crop']:
                if not self.cfg['use_old']:
                    # 1. Under 80% probability, we resize the image, which will ensure the size of instance is [hmin,hmax][wmin,wmax]
                    #    otherwise, keep the image unchanged
                    # 2. crop or padding the image to a fixed size
                    img, mask, hcoords = crop_resize_instance_v2(img, mask, hcoords, height, width, self.cfg['overlap_ratio'],
                                                                 self.cfg['resize_hmin'], self.cfg['resize_hmax'],
                                                                 self.cfg['resize_wmin'], self.cfg['resize_wmax'])
                else:
                    # 1. firstly crop a region which is [scale_min,scale_max]*[height,width], which ensures that
                    #    the area of the intersection between the cropped region and the instance region is at least
                    #    overlap_ratio**2 of instance region.
                    # 2. if the region is larger than original image, then padding 0
                    # 3. then resize the cropped image to [height, width] (bilinear for image, nearest for mask)
                    img, mask, hcoords = crop_resize_instance_v1(img, mask, hcoords, height, width, self.cfg['overlap_ratio'],
                                                                 self.cfg['resize_ratio_min'], self.cfg['resize_ratio_max'])
        else:
            img, mask = crop_or_padding_to_fixed_size(img, mask, height, width)


        # randomly flip
        if self.cfg['flip'] and np.random.random() < 0.5:
            img, mask, hcoords = flip(img, mask, hcoords)

        return img, mask, hcoords


class ImageSizeBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, drop_last, cfg=default_aug_cfg):

        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of torch.utils.data.Sampler, but got sampler={}".format(sampler))
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size should be a positive integeral value, but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got drop_last={}".format(drop_last))

        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.hmin=cfg['hmin']
        self.hmax=cfg['hmax']
        self.wmin=cfg['wmin']
        self.wmax=cfg['wmax']
        self.size_int=cfg['size_int']
        self.hint=(self.hmax-self.hmin)//self.size_int+1
        self.wint=(self.wmax-self.wmin)//self.size_int+1

    def generate_height_width(self):
        hi, wi = np.random.randint(0, self.hint), np.random.randint(0, self.wint)
        h, w = self.hmin + hi * self.size_int, self.wmin + wi * self.size_int
        return h,w

    def __iter__(self):
        batch = []
        h, w = self.generate_height_width()
        for idx in self.sampler:
            batch.append((idx, h, w))
            if len(batch) == self.batch_size:
                h, w = self.generate_height_width()
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            h, w = self.generate_height_width()
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size

