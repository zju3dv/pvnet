import sys

from skimage.io import imsave


sys.path.append('.')
sys.path.append('..')
from lib.ransac_voting_gpu_layer.ransac_voting_gpu import ransac_voting_layer_v3, \
    estimate_voting_distribution_with_mean, ransac_voting_layer_v5, ransac_motion_voting
from lib.networks.model_repository import *
from lib.datasets.linemod_dataset import LineModDatasetRealAug, ImageSizeBatchSampler, VotingType
from lib.utils.data_utils import LineModImageDB, OcclusionLineModImageDB, TruncatedLineModImageDB
from lib.utils.arg_utils import args
from lib.utils.draw_utils import visualize_bounding_box, imagenet_to_uint8, visualize_mask, visualize_points, img_pts_to_pts_img
from lib.utils.base_utils import save_pickle
import json

from lib.utils.evaluation_utils import Evaluator
from lib.utils.net_utils import AverageMeter, Recorder, smooth_l1_loss, \
    load_model, save_model, adjust_learning_rate, compute_precision_recall, set_learning_rate
from lib.utils.config import cfg

from torch.nn import DataParallel
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch import nn, optim
import torch
import torch.nn.functional as F
import os
import time
from collections import OrderedDict
import random
import numpy as np

with open(args.cfg_file,'r') as f:
    train_cfg=json.load(f)
train_cfg['model_name']='{}_{}'.format(args.linemod_cls,train_cfg['model_name'])

if train_cfg['vote_type']=='BB8C':
    vote_type=VotingType.BB8C
    vote_num=9
elif train_cfg['vote_type']=='BB8S':
    vote_type=VotingType.BB8S
    vote_num=9
elif train_cfg['vote_type']=='Farthest':
    vote_type=VotingType.Farthest
    vote_num=9
elif train_cfg['vote_type']=='Farthest4':
    vote_type=VotingType.Farthest4
    vote_num=5
elif train_cfg['vote_type']=='Farthest12':
    vote_type=VotingType.Farthest12
    vote_num=13
elif train_cfg['vote_type']=='Farthest16':
    vote_type=VotingType.Farthest16
    vote_num=17
else:
    assert(train_cfg['vote_type']=='BB8')
    vote_type=VotingType.BB8
    vote_num=8


seg_loss_rec = AverageMeter()
ver_loss_rec = AverageMeter()
precision_rec = AverageMeter()
recall_rec = AverageMeter()
recs=[seg_loss_rec,ver_loss_rec,precision_rec,recall_rec]
recs_names=['scalar/seg','scalar/ver','scalar/precision','scalar/recall']

data_time = AverageMeter()
batch_time = AverageMeter()
recorder = Recorder(True,os.path.join(cfg.REC_DIR,train_cfg['model_name']),
                    os.path.join(cfg.REC_DIR,train_cfg['model_name']+'.log'))

# network_time,voting_time,load_time=[],[],[]

# poses_pr=[]
# poses_gt=[]

class NetWrapper(nn.Module):
    def __init__(self,net):
        super(NetWrapper,self).__init__()
        self.net=net
        self.criterion=nn.CrossEntropyLoss(reduce=False)

    def forward(self, image, mask, vertex, vertex_weights):
        seg_pred, vertex_pred = self.net(image)
        loss_seg = self.criterion(seg_pred, mask)
        loss_seg = torch.mean(loss_seg.view(loss_seg.shape[0],-1),1)
        loss_vertex = smooth_l1_loss(vertex_pred, vertex, vertex_weights, reduce=False)
        precision, recall = compute_precision_recall(seg_pred, mask)
        return seg_pred, vertex_pred, loss_seg, loss_vertex, precision, recall


class EvalWrapper(nn.Module):
    def forward(self, seg_pred, vertex_pred, use_argmax=True, use_uncertainty=False):
        vertex_pred=vertex_pred.permute(0,2,3,1)
        b,h,w,vn_2=vertex_pred.shape
        vertex_pred=vertex_pred.view(b,h,w,vn_2//2,2)
        if use_argmax:
            mask=torch.argmax(seg_pred,1)
        else:
            mask=seg_pred
        if use_uncertainty:
            return ransac_voting_layer_v5(mask,vertex_pred,128,inlier_thresh=0.99,max_num=100)
        else:
            return ransac_voting_layer_v3(mask,vertex_pred,128,inlier_thresh=0.99,max_num=100)

class MotionEvalWrapper(nn.Module):
    def forward(self, seg_pred, vertex_pred, use_argmax=True, use_uncertainty=False):
        vertex_pred=vertex_pred.permute(0,2,3,1)
        b,h,w,vn_2=vertex_pred.shape
        vertex_pred=vertex_pred.view(b,h,w,vn_2//2,2)
        if use_argmax:
            mask=torch.argmax(seg_pred,1)
        else:
            mask=seg_pred
        return ransac_motion_voting(mask, vertex_pred)

class UncertaintyEvalWrapper(nn.Module):
    def forward(self, seg_pred, vertex_pred, use_argmax=True):
        vertex_pred=vertex_pred.permute(0,2,3,1)
        b,h,w,vn_2=vertex_pred.shape
        vertex_pred=vertex_pred.view(b,h,w,vn_2//2,2)
        if use_argmax:
            mask=torch.argmax(seg_pred,1)
        else:
            mask=seg_pred
        mean=ransac_voting_layer_v3(mask, vertex_pred, 512, inlier_thresh=0.99)
        mean, var=estimate_voting_distribution_with_mean(mask,vertex_pred,mean)
        return mean, var

def train(net, optimizer, dataloader, epoch):
    for rec in recs: rec.reset()
    data_time.reset()
    batch_time.reset()

    train_begin=time.time()

    net.train()
    size = len(dataloader)
    end=time.time()
    for idx, data in enumerate(dataloader):
        image, mask, vertex, vertex_weights, pose, _ = [d.cuda() for d in data]
        data_time.update(time.time()-end)

        seg_pred, vertex_pred, loss_seg, loss_vertex, precision, recall = net(image, mask, vertex, vertex_weights)
        loss_seg, loss_vertex, precision, recall=[torch.mean(val) for val in (loss_seg, loss_vertex, precision, recall)]
        loss = loss_seg + loss_vertex * train_cfg['vertex_loss_ratio']
        vals=(loss_seg,loss_vertex,precision,recall)
        for rec,val in zip(recs,vals): rec.update(val)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time()-end)
        end=time.time()

        if idx % train_cfg['loss_rec_step'] == 0:
            step = epoch * size + idx
            losses_batch=OrderedDict()
            for name,rec in zip(recs_names,recs): losses_batch['train/'+name]=rec.avg
            recorder.rec_loss_batch(losses_batch,step,epoch)
            for rec in recs: rec.reset()

            data_time.reset()
            batch_time.reset()

        if idx % train_cfg['img_rec_step'] == 0:
            batch_size = image.shape[0]
            nrow = 5 if batch_size > 5 else batch_size
            recorder.rec_segmentation(F.softmax(seg_pred, dim=1), num_classes=2, nrow=nrow, step=step, name='train/image/seg')
            recorder.rec_vertex(vertex_pred, vertex_weights, nrow=4, step=step, name='train/image/ver')

    print('epoch {} training cost {} s'.format(epoch,time.time()-train_begin))

def val(net, dataloader, epoch, val_prefix='val', use_camera_intrinsic=False, use_motion=False):
    for rec in recs: rec.reset()

    test_begin = time.time()
    evaluator = Evaluator()

    eval_net=DataParallel(EvalWrapper().cuda()) if not use_motion else DataParallel(MotionEvalWrapper().cuda())
    uncertain_eval_net=DataParallel(UncertaintyEvalWrapper().cuda())
    net.eval()
    for idx, data in enumerate(dataloader):
        if use_camera_intrinsic:
            image, mask, vertex, vertex_weights, pose, corner_target, Ks = [d.cuda() for d in data]
        else:
            image, mask, vertex, vertex_weights, pose, corner_target = [d.cuda() for d in data]

        with torch.no_grad():
            seg_pred, vertex_pred, loss_seg, loss_vertex, precision, recall = net(image, mask, vertex, vertex_weights)

            loss_seg, loss_vertex, precision, recall=[torch.mean(val) for val in (loss_seg, loss_vertex, precision, recall)]

            if (train_cfg['eval_epoch']
                and epoch%train_cfg['eval_inter']==0
                and epoch>=train_cfg['eval_epoch_begin']) or args.test_model:
                if args.use_uncertainty_pnp:
                    mean,cov_inv=uncertain_eval_net(seg_pred,vertex_pred)
                    mean=mean.cpu().numpy()
                    cov_inv=cov_inv.cpu().numpy()
                else:
                    corner_pred=eval_net(seg_pred,vertex_pred).cpu().detach().numpy()
                pose=pose.cpu().numpy()

                b=pose.shape[0]
                pose_preds=[]
                for bi in range(b):
                    intri_type='use_intrinsic' if use_camera_intrinsic else 'linemod'
                    K=Ks[bi].cpu().numpy() if use_camera_intrinsic else None
                    if args.use_uncertainty_pnp:
                        pose_preds.append(evaluator.evaluate_uncertainty(mean[bi],cov_inv[bi],pose[bi],args.linemod_cls,
                                                                         intri_type,vote_type,intri_matrix=K))
                    else:
                        pose_preds.append(evaluator.evaluate(corner_pred[bi],pose[bi],args.linemod_cls,intri_type,
                                                             vote_type,intri_matrix=K))


                if args.save_inter_result:
                    mask_pr = torch.argmax(seg_pred, 1).cpu().detach().numpy()
                    mask_gt = mask.cpu().detach().numpy()
                    # assume batch size = 1
                    imsave(os.path.join(args.save_inter_dir, '{}_mask_pr.png'.format(idx)), mask_pr[0])
                    imsave(os.path.join(args.save_inter_dir, '{}_mask_gt.png'.format(idx)), mask_gt[0])
                    imsave(os.path.join(args.save_inter_dir, '{}_rgb.png'.format(idx)),
                           imagenet_to_uint8(image.cpu().detach().numpy()[0]))
                    save_pickle([pose_preds[0],pose[0]],os.path.join(args.save_inter_dir, '{}_pose.pkl'.format(idx)))

            vals=[loss_seg,loss_vertex,precision,recall]
            for rec,val in zip(recs,vals): rec.update(val)

    with torch.no_grad():
        batch_size = image.shape[0]
        nrow = 5 if batch_size > 5 else batch_size
        recorder.rec_segmentation(F.softmax(seg_pred, dim=1), num_classes=2, nrow=nrow,
                                  step=epoch, name='{}/image/seg'.format(val_prefix))
        recorder.rec_vertex(vertex_pred, vertex_weights, nrow=4, step=epoch, name='{}/image/ver'.format(val_prefix))

    losses_batch=OrderedDict()
    for name, rec in zip(recs_names, recs): losses_batch['{}/'.format(val_prefix) + name] = rec.avg
    if (train_cfg['eval_epoch']
        and epoch%train_cfg['eval_inter']==0
        and epoch>=train_cfg['eval_epoch_begin']) or args.test_model:
        proj_err,add,cm=evaluator.average_precision(False)
        losses_batch['{}/scalar/projection_error'.format(val_prefix)]=proj_err
        losses_batch['{}/scalar/add'.format(val_prefix)]=add
        losses_batch['{}/scalar/cm'.format(val_prefix)]=cm
    recorder.rec_loss_batch(losses_batch, epoch, epoch, val_prefix)
    for rec in recs: rec.reset()

    print('epoch {} {} cost {} s'.format(epoch,val_prefix,time.time()-test_begin))

def train_net():
    net=Resnet18_8s(ver_dim=vote_num*2, seg_dim=2)
    net=NetWrapper(net)
    net=DataParallel(net).cuda()

    optimizer = optim.Adam(net.parameters(), lr=train_cfg['lr'])
    model_dir=os.path.join(cfg.MODEL_DIR,train_cfg['model_name'])
    motion_model=train_cfg['motion_model']
    print('motion state {}'.format(motion_model))

    if args.test_model:
        begin_epoch=load_model(net.module.net, optimizer, model_dir, args.load_epoch)

        if args.normal:
            print('testing normal linemod ...')
            image_db = LineModImageDB(args.linemod_cls,has_render_set=False,
                                      has_fuse_set=False)
            test_db = image_db.test_real_set+image_db.val_real_set
            test_set = LineModDatasetRealAug(test_db, cfg.LINEMOD, vote_type, augment=False, use_motion=motion_model)
            test_sampler = SequentialSampler(test_set)
            test_batch_sampler = ImageSizeBatchSampler(test_sampler, train_cfg['test_batch_size'], False)
            test_loader = DataLoader(test_set, batch_sampler=test_batch_sampler, num_workers=0)
            prefix='test' if args.use_test_set else 'val'
            val(net, test_loader, begin_epoch, prefix, use_motion=motion_model)

        if args.occluded and args.linemod_cls in cfg.occ_linemod_cls_names:
            print('testing occluded linemod ...')
            occ_image_db = OcclusionLineModImageDB(args.linemod_cls)
            occ_test_db = occ_image_db.test_real_set
            occ_test_set = LineModDatasetRealAug(occ_test_db, cfg.OCCLUSION_LINEMOD, vote_type,
                                                 augment=False, use_motion=motion_model)
            occ_test_sampler = SequentialSampler(occ_test_set)
            occ_test_batch_sampler = ImageSizeBatchSampler(occ_test_sampler, train_cfg['test_batch_size'], False)
            occ_test_loader = DataLoader(occ_test_set, batch_sampler=occ_test_batch_sampler, num_workers=0)
            prefix='occ_test' if args.use_test_set else 'occ_val'
            val(net, occ_test_loader, begin_epoch, prefix, use_motion=motion_model)

        if args.truncated:
            print('testing truncated linemod ...')
            trun_image_db = TruncatedLineModImageDB(args.linemod_cls)
            print(len(trun_image_db.set))
            trun_image_set = LineModDatasetRealAug(trun_image_db.set, cfg.LINEMOD, vote_type, augment=False,
                                                   use_intrinsic=True, use_motion=motion_model)
            trun_test_sampler = SequentialSampler(trun_image_set)
            trun_test_batch_sampler = ImageSizeBatchSampler(trun_test_sampler, train_cfg['test_batch_size'], False)
            trun_test_loader = DataLoader(trun_image_set, batch_sampler=trun_test_batch_sampler, num_workers=0)
            prefix='trun_test'
            val(net, trun_test_loader, begin_epoch, prefix, True, use_motion=motion_model)

    else:
        begin_epoch=0
        if train_cfg['resume']:
            begin_epoch=load_model(net.module.net, optimizer, model_dir)


        image_db = LineModImageDB(args.linemod_cls,
                                  has_fuse_set=train_cfg['use_fuse'],
                                  has_render_set=True)

        train_db=[]
        train_db+=image_db.render_set
        if train_cfg['use_real_train']:
            train_db+=image_db.train_real_set
        if train_cfg['use_fuse']:
            train_db+=image_db.fuse_set

        train_set = LineModDatasetRealAug(train_db, cfg.LINEMOD, vote_type, augment=True, cfg=train_cfg['aug_cfg'], use_motion=motion_model)
        train_sampler = RandomSampler(train_set)
        train_batch_sampler = ImageSizeBatchSampler(train_sampler, train_cfg['train_batch_size'], False, cfg=train_cfg['aug_cfg'])
        train_loader = DataLoader(train_set, batch_sampler=train_batch_sampler, num_workers=12)

        val_db=image_db.val_real_set
        val_set = LineModDatasetRealAug(val_db, cfg.LINEMOD, vote_type, augment=False, cfg=train_cfg['aug_cfg'], use_motion=motion_model)
        val_sampler = SequentialSampler(val_set)
        val_batch_sampler = ImageSizeBatchSampler(val_sampler, train_cfg['test_batch_size'], False, cfg=train_cfg['aug_cfg'])
        val_loader = DataLoader(val_set, batch_sampler=val_batch_sampler, num_workers=12)

        if args.linemod_cls in cfg.occ_linemod_cls_names:
            occ_image_db=OcclusionLineModImageDB(args.linemod_cls)
            occ_val_db=occ_image_db.test_real_set[:len(occ_image_db.test_real_set)//2]
            occ_val_set = LineModDatasetRealAug(occ_val_db, cfg.OCCLUSION_LINEMOD, vote_type, augment=False, cfg=train_cfg['aug_cfg'], use_motion=motion_model)
            occ_val_sampler = SequentialSampler(occ_val_set)
            occ_val_batch_sampler = ImageSizeBatchSampler(occ_val_sampler, train_cfg['test_batch_size'], False, cfg=train_cfg['aug_cfg'])
            occ_val_loader = DataLoader(occ_val_set, batch_sampler=occ_val_batch_sampler, num_workers=12)

        for epoch in range(begin_epoch, train_cfg['epoch_num']):
            adjust_learning_rate(optimizer,epoch,train_cfg['lr_decay_rate'],train_cfg['lr_decay_epoch'])
            train(net, optimizer, train_loader, epoch)
            val(net, val_loader, epoch,use_motion=motion_model)
            if args.linemod_cls in cfg.occ_linemod_cls_names:
                val(net, occ_val_loader, epoch, 'occ_val',use_motion=motion_model)
            save_model(net.module.net, optimizer, epoch, model_dir)

# def save_dataset(dataset,prefix=''):
#     with open('assets/{}{}.txt'.format(prefix,args.linemod_cls),'w') as f:
#         for data in dataset: f.write(data['rgb_pth']+'\n')
#
# def save_poses_dataset(prefix=''):
#     print(np.asarray(poses_pr).shape)
#     np.save('assets/{}{}_pr.npy'.format(prefix,args.linemod_cls),np.asarray(poses_pr))
#     np.save('assets/{}{}_gt.npy'.format(prefix,args.linemod_cls),np.asarray(poses_gt))

if __name__ == "__main__":
    train_net()
    # save_poses_dataset('trun_')
