from easydict import EasyDict
import os
import sys
import numpy as np

cfg = EasyDict()

"""
Path settings
"""
cfg.UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
cfg.LIB_DIR = os.path.dirname(cfg.UTILS_DIR)
cfg.ROOT_DIR = os.path.dirname(cfg.LIB_DIR)
cfg.NETS_DIR = os.path.join(cfg.LIB_DIR, 'networks')
cfg.DATASET_DIR = os.path.join(cfg.LIB_DIR, 'datasets')
cfg.DATA_DIR = os.path.join(cfg.ROOT_DIR, 'data')
cfg.MODEL_DIR = os.path.join(cfg.DATA_DIR, 'model')
cfg.REC_DIR = os.path.join(cfg.DATA_DIR, 'record')
cfg.FIGURE_DIR = os.path.join(cfg.ROOT_DIR, 'figure')


def add_path():
    for key, value in cfg.items():
        if 'DIR' in key:
            sys.path.insert(0, value)


add_path()


"""
Data settings
"""
cfg.LINEMOD = os.path.join(cfg.DATA_DIR, 'LINEMOD')
cfg.LINEMOD_ORIG = os.path.join(cfg.DATA_DIR, 'LINEMOD_ORIG')
cfg.OCCLUSION_LINEMOD = os.path.join(cfg.DATA_DIR, 'OCCLUSION_LINEMOD')
cfg.YCB = os.path.join(cfg.DATA_DIR, 'YCB')

"""
Rendering setting
"""
cfg.BLENDER_PATH = '/home/pengsida/Software/blender-2.79a-linux-glibc219-x86_64/blender'
cfg.NUM_SYN = 35000
# cfg.NUM_SYN = 10000
cfg.WIDTH = 640
cfg.HEIGHT = 480
cfg.low_azi = 0
cfg.high_azi = 360
cfg.low_ele = -15
cfg.high_ele = 40
cfg.low_theta = 10
cfg.high_theta = 40
cfg.cam_dist = 0.5
cfg.MIN_DEPTH = 0
cfg.MAX_DEPTH = 2

cfg.render_K=np.array([[700., 0., 320.],
                       [0., 700., 240.],
                       [0., 0., 1.]],np.float32)

cfg.linemod_K=np.array([[572.41140,0.       ,325.26110],
                        [0.       ,573.57043,242.04899],
                        [0.       ,0.       ,1.       ]],np.float32)

cfg.linemod_cls_names=['ape','cam','cat','duck','glue','iron','phone',
                       'benchvise','can','driller','eggbox','holepuncher','lamp']
cfg.occ_linemod_cls_names=['ape','can','cat','driller','duck','eggbox','glue','holepuncher']
cfg.linemod_plane=['can']

cfg.symmetry_linemod_cls_names=['glue','eggbox']


'''
pascal 3d +
'''
cfg.PASCAL = os.path.join(cfg.DATA_DIR, 'PASCAL3D')
cfg.pascal_cls_names=['aeroplane','bicycle','boat','bottle','bus','car',
                      'chair','diningtable','motorbike','sofa','train','tvmonitor']
cfg.pascal_size=128


'''
YCB
'''
cfg.ycb_sym_cls=[21,20,19,16,13] # foam_brick extra_large_clamp large_clamp wood_block bowl
cfg.ycb_class_num=21
