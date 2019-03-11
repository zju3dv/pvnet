import sys

sys.path.append('.')
sys.path.append('..')
from lib.networks.model_repository import *
from lib.utils.arg_utils import args
from lib.utils.net_utils import smooth_l1_loss, load_model, compute_precision_recall
from lib.ransac_voting_gpu_layer.ransac_voting_gpu import ransac_voting_layer_v3
from lib.utils.evaluation_utils import pnp
from lib.utils.draw_utils import imagenet_to_uint8, visualize_bounding_box
from lib.utils.base_utils import Projector
import json

from lib.utils.config import cfg

from torch.nn import DataParallel
from torch import nn, optim
import torch
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

with open(args.cfg_file, 'r') as f:
    train_cfg = json.load(f)
train_cfg['model_name'] = '{}_{}'.format(args.linemod_cls, train_cfg['model_name'])

vote_num = 9


class NetWrapper(nn.Module):
    def __init__(self, net):
        super(NetWrapper, self).__init__()
        self.net = net
        self.criterion = nn.CrossEntropyLoss(reduce=False)

    def forward(self, image, mask, vertex, vertex_weights):
        seg_pred, vertex_pred = self.net(image)
        loss_seg = self.criterion(seg_pred, mask)
        loss_seg = torch.mean(loss_seg.view(loss_seg.shape[0], -1), 1)
        loss_vertex = smooth_l1_loss(vertex_pred, vertex, vertex_weights, reduce=False)
        precision, recall = compute_precision_recall(seg_pred, mask)
        return seg_pred, vertex_pred, loss_seg, loss_vertex, precision, recall


class EvalWrapper(nn.Module):
    def forward(self, seg_pred, vertex_pred, use_argmax=True):
        vertex_pred = vertex_pred.permute(0, 2, 3, 1)
        b, h, w, vn_2 = vertex_pred.shape
        vertex_pred = vertex_pred.view(b, h, w, vn_2 // 2, 2)
        if use_argmax:
            mask = torch.argmax(seg_pred, 1)
        else:
            mask = seg_pred
        return ransac_voting_layer_v3(mask, vertex_pred, 512, inlier_thresh=0.99)


def compute_vertex(mask, points_2d):
    num_keypoints = points_2d.shape[0]
    h, w = mask.shape
    m = points_2d.shape[0]
    xy = np.argwhere(mask == 1)[:, [1, 0]]
    vertex = xy[:, None, :] * np.ones(shape=[1, num_keypoints, 1])
    vertex = points_2d[None, :, :2] - vertex
    norm = np.linalg.norm(vertex, axis=2, keepdims=True)
    norm[norm < 1e-3] += 1e-3
    vertex = vertex / norm

    vertex_out = np.zeros([h, w, m, 2], np.float32)
    vertex_out[xy[:, 1], xy[:, 0]] = vertex
    return np.reshape(vertex_out, [h, w, m * 2])


def read_data():
    import torchvision.transforms as transforms

    demo_dir_path = os.path.join(cfg.DATA_DIR, 'demo')
    rgb = Image.open(os.path.join(demo_dir_path, 'cat.jpg'))
    mask = np.array(Image.open(os.path.join(demo_dir_path, 'cat_mask.png'))).astype(np.int32)[..., 0]
    mask[mask != 0] = 1
    points_3d = np.loadtxt(os.path.join(demo_dir_path, 'cat_points_3d.txt'))
    bb8_3d = np.loadtxt(os.path.join(demo_dir_path, 'cat_bb8_3d.txt'))
    pose = np.load(os.path.join(demo_dir_path, 'cat_pose.npy'))

    projector = Projector()
    points_2d = projector.project(points_3d, pose, 'linemod')
    vertex = compute_vertex(mask, points_2d)

    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    rgb = transformer(rgb)
    vertex = torch.tensor(vertex, dtype=torch.float32).permute(2, 0, 1)
    mask = torch.tensor(np.ascontiguousarray(mask), dtype=torch.int64)
    vertex_weight = mask.unsqueeze(0).float()
    pose = torch.tensor(pose.astype(np.float32))
    points_2d = torch.tensor(points_2d.astype(np.float32))
    data = (rgb, mask, vertex, vertex_weight, pose, points_2d)

    return data, points_3d, bb8_3d


def demo():
    net = Resnet18_8s(ver_dim=vote_num * 2, seg_dim=2)
    net = NetWrapper(net).cuda()
    net = DataParallel(net)

    optimizer = optim.Adam(net.parameters(), lr=train_cfg['lr'])
    model_dir = os.path.join(cfg.MODEL_DIR, "cat_demo")
    load_model(net.module.net, optimizer, model_dir, args.load_epoch)
    data, points_3d, bb8_3d = read_data()
    image, mask, vertex, vertex_weights, pose, corner_target = [d.unsqueeze(0).cuda() for d in data]
    seg_pred, vertex_pred, loss_seg, loss_vertex, precision, recall = net(image, mask, vertex, vertex_weights)

    eval_net = DataParallel(EvalWrapper().cuda())
    corner_pred = eval_net(seg_pred, vertex_pred).cpu().detach().numpy()[0]
    camera_matrix = np.array([[572.4114, 0., 325.2611],
                              [0., 573.57043, 242.04899],
                              [0., 0., 1.]])
    pose_pred = pnp(points_3d, corner_pred, camera_matrix)

    projector = Projector()
    bb8_2d_pred = projector.project(bb8_3d, pose_pred, 'linemod')
    bb8_2d_gt = projector.project(bb8_3d, pose[0].detach().cpu().numpy(), 'linemod')
    image = imagenet_to_uint8(image.detach().cpu().numpy())[0]
    visualize_bounding_box(image[None, ...], bb8_2d_pred[None, None, ...], bb8_2d_gt[None, None, ...])


if __name__ == "__main__":
    demo()
