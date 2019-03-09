from lib.utils.config import cfg
from lib.networks.vgg import vgg16
from lib.utils.draw_utils import visualize_bounding_box
from lib.utils.net_utils import conv, AverageMeter, load_model, save_model, Recorder, smooth_l1_loss, load_net
from lib.utils.arg_utils import args
from lib.utils.evaluation_utils import Evaluator
from lib.datasets.linemod_dataset import LineModDatasetSyn, LineModDatasetReal
from lib.hough_voting_layer.hough_voting import HoughVoting
from lib.hough_voting_gpu_layer.hough_voting_gpu import hough_voting_gpu
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class VGG16Convs(nn.Module):
    def __init__(self):
        super(VGG16Convs, self).__init__()

        self.vgg = vgg16(pretrained=True)
        self.conv4_3 = self.vgg.features[:23]
        self.conv5_3 = self.vgg.features[23:30]

        self.seg_branch1 = conv(num_input=512, num_output=64, kernel_size=1, stride=1, padding=0)
        self.seg_branch2 = conv(num_input=512, num_output=64, kernel_size=1, stride=1, padding=0)
        self.seg = conv(num_input=64, num_output=2, kernel_size=1, stride=1, padding=0, relu=False)

        self.vertex_branch1 = conv(num_input=512, num_output=128, kernel_size=1, stride=1, padding=0, relu=False)
        self.vertex_branch2 = conv(num_input=512, num_output=128, kernel_size=1, stride=1, padding=0, relu=False)
        self.vertex = conv(num_input=128, num_output=16, kernel_size=1, stride=1, padding=0, relu=False)

    def forward(self, im_data):
        conv4_3 = self.conv4_3(im_data)
        conv5_3 = self.conv5_3(conv4_3)

        seg_branch1 = self.seg_branch1(conv4_3)
        seg_branch2 = self.seg_branch2(conv5_3)
        seg_add = seg_branch1 + F.interpolate(seg_branch2, size=seg_branch1.shape[2:], mode='bilinear', align_corners=False)
        seg_score = self.seg(F.interpolate(seg_add, size=im_data.shape[2:], mode='bilinear', align_corners=False))
        seg_pred = F.softmax(seg_score, dim=1)

        vertex_branch1 = self.vertex_branch1(conv4_3)
        vertex_branch2 = self.vertex_branch2(conv5_3)
        vertex_add = vertex_branch1 + F.interpolate(vertex_branch2, size=vertex_branch1.shape[2:], mode='bilinear', align_corners=False)
        vertex_pred = self.vertex(F.interpolate(vertex_add, size=im_data.shape[2:], mode='bilinear', align_corners=False))

        return seg_score, seg_pred, vertex_pred


loss_rec = AverageMeter()
print_interval = 10
rec_interval = 100
recorder = Recorder(args.rec)


def train(net, optimizer, dataloader, device, epoch):
    net.train()
    size = len(dataloader)
    for idx, data in enumerate(dataloader):
        im_data, mask, vertex_targets, vertex_weights = [d.to(device) for d in data]

        seg_score, seg_pred, vertex_pred = net(im_data)
        loss_cls = F.cross_entropy(seg_score, mask)
        loss_vertex = smooth_l1_loss(vertex_pred, vertex_targets, vertex_weights)
        loss = loss_cls + loss_vertex
        loss_rec.update(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % print_interval == 0:
            step = epoch * size + idx
            recorder.rec_loss(loss_rec.avg, step)
            loss_rec.reset()

        if idx % rec_interval == 0:
            batch_size = im_data.shape[0]
            nrow = 5 if batch_size > 5 else batch_size
            recorder.rec_segmentation(seg_pred, num_classes=2, nrow=nrow, step=step)
            recorder.rec_vertex(vertex_pred, vertex_weights, nrow=4, step=step)


def train_vgg16():
    device = torch.device('cuda:0')

    dataset = LineModDatasetReal()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    net = VGG16Convs()
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=2e-5, weight_decay=5e-4)

    model_dir = os.path.join(cfg.MODEL_DIR, 'vgg16')
    epoch = load_model(net, optimizer, model_dir)
    for epoch in range(epoch, epoch + 100):
        train(net, optimizer, dataloader, device, epoch)
        save_model(net, optimizer, epoch, model_dir)


class VGG16ConvsEvaluation(object):
    def __init__(self, dataset=LineModDatasetSyn(mode='test')):
        self.device = torch.device('cuda:0')
        self.dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=4)
        net = VGG16Convs()
        self.net = net.to(self.device)
        self.hough_voting = HoughVoting(num_classes=2)

        load_net(self.net, os.path.join(cfg.MODEL_DIR, 'vgg16'))

    @staticmethod
    def visualize_seg(seg_pred):
        seg_label = torch.argmax(seg_pred, dim=1).detach().cpu().numpy()
        seg_label = np.transpose(np.reshape(seg_label, newshape=[2, 5, 256, 256]), axes=[0, 2, 1, 3])
        seg_label = np.reshape(seg_label, newshape=[2 * 256, 5 * 256])
        plt.imshow(seg_label)
        plt.show()

    @staticmethod
    def visualize_vertex(vertex_pred, seg_pred):
        mask = (torch.argmax(seg_pred, dim=1, keepdim=True) != 0).float()
        vertex_pred = vertex_pred * mask
        vertex_pred = vertex_pred.detach().cpu().numpy()
        batch_size = vertex_pred.shape[1] // 2

        for idx in range(batch_size):
            vertex = vertex_pred[:, 2 * idx: 2 * idx + 2]
            vertex = np.reshape(np.transpose(vertex, axes=[0, 2, 1, 3]), newshape=[10, 256, 512])
            vertex = np.transpose(np.reshape(vertex, newshape=[5, 2, 256, 512]), axes=[0, 2, 1, 3])
            vertex = np.reshape(vertex, newshape=[5 * 256, 2 * 512])
            plt.imshow(vertex)
            plt.show()

    def get_corners(self, mask, vertex_pred):
        """
        :param mask: N x H x W
        :param vertex_pred: N x 2 x H x W
        :return: corners: N x num_objects x num_corners x 2
        """
        labelmap = mask.detach().cpu().numpy()
        vertmap = vertex_pred.permute(0, 2, 3, 1).detach().cpu().numpy()
        corners_size = vertmap.shape[3] // 2

        corners = []
        for idx in range(corners_size):
            corners.append(self.hough_voting(labelmap, vertmap[..., 2 * idx: 2 * idx + 2]))

        return np.transpose(corners, axes=[1, 2, 0, 3])

    @staticmethod
    def get_corners_gpu(mask, vertex_pred):
        """
        :param mask: N x H x W
        :param vertex_pred: N x 2 x H x W
        :return: corners: N x num_objects x num_corners x 2
        """
        labelmap = mask.int()
        vertmap = vertex_pred.permute(0, 2, 3, 1)
        corners_size = vertmap.shape[3] // 2

        corners = []
        for idx in range(corners_size):
            corner = torch.cat(hough_voting_gpu(labelmap, vertmap[..., 2 * idx:2 * idx + 2].contiguous(), 2))
            corners.append(corner)

        return torch.stack(corners, dim=0).permute(1, 0, 2).unsqueeze(1).detach().cpu().numpy()

    @staticmethod
    def visualize_bounding_box(corners_pred, corners_targets, rgb):
        visualize_bounding_box(corners_pred, corners_targets, rgb)

    def run(self):
        self.net.eval()
        evaluator = Evaluator()

        for idx, data in enumerate(self.dataloader):
            im_data, pose_targets = data
            pose_targets = pose_targets.detach().cpu().numpy()
            seg_score, seg_pred, vertex_pred = self.net(im_data.to(self.device))
            corners_pred = self.get_corners_gpu(torch.argmax(seg_pred, dim=1), vertex_pred)
            batch_size = im_data.shape[0]
            for idx in range(0, batch_size):
                evaluator.evaluate(points_2d=corners_pred[idx, 0], pose_targets=pose_targets[idx], class_type='cat')

        evaluator.average_precision()
