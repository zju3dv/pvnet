import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transforms3d.euler import euler2mat
from skimage.io import imsave


def visualize_bounding_box(rgb, corners_pred, corners_targets=None, centers_pred=None, centers_targets=None, save=False, save_fn=None):
    '''

    :param rgb:             torch tensor with size [b,3,h,w] or numpy array with size [b,h,w,3]
    :param corners_pred:    [b,1,8,2]
    :param corners_targets: [b,1,8,2] or None
    :param centers_pred:    [b,1,2] or None
    :param centers_targets:  [b,1,2] or None
    :param save:
    :param save_fn:
    :return:
    '''
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.permute(0, 2, 3, 1).detach().cpu().numpy()
    rgb = rgb.astype(np.uint8)

    batch_size = corners_pred.shape[0]
    for idx in range(batch_size):
        _, ax = plt.subplots(1)
        ax.imshow(rgb[idx])
        ax.add_patch(
            patches.Polygon(xy=corners_pred[idx, 0][[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='b'))
        ax.add_patch(
            patches.Polygon(xy=corners_pred[idx, 0][[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor='b'))
        if corners_targets is not None:
            ax.add_patch(patches.Polygon(xy=corners_targets[idx, 0][[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1,
                                         edgecolor='g'))
            ax.add_patch(patches.Polygon(xy=corners_targets[idx, 0][[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1,
                                         edgecolor='g'))
        if centers_pred is not None:
            ax.plot(centers_pred[idx, 0, 0],centers_pred[idx, 0, 1],'*')
        if centers_targets is not None:
            ax.plot(centers_targets[idx, 0, 0], centers_pred[idx, 0, 1], '*')
        if not save:
            plt.show()
        else:
            plt.savefig(save_fn.format(idx))
        plt.close()

def visualize_mask(mask_pred,mask_gt, save=False, save_fn=None):
    '''

    :param mask_pred:   [b,h,w]
    :param mask_gt:     [b,h,w]
    :return:
    '''
    b,h,w=mask_gt.shape
    for bi in range(b):
        img=np.zeros([h,w,3],np.uint8)
        img[np.logical_and(mask_gt[bi],mask_pred[bi])]=np.asarray([0,255,0])
        img[np.logical_and(np.logical_not(mask_gt[bi]),mask_pred[bi])]=np.asarray([[[255,255,0]]])
        img[np.logical_and(np.logical_not(mask_pred[bi]),mask_gt[bi])]=np.asarray([[[255,0,0]]])
        plt.imshow(img)
        if not save:
            plt.show()
        else:
            plt.savefig(save_fn.format(bi))
        plt.close()

def visualize_overlap_mask(img,mask,save_fn):
    b,h,w=mask.shape
    for bi in range(b):
        img[bi][mask[bi]>0]//=2
        img[bi][mask[bi]>0]+=np.asarray([0,128,0],np.uint8)
        imsave(save_fn.format(bi),img[bi])

def visualize_points_3d(pts1,pts2,K,h=480,w=640):
    '''

    :param pts1:  [pn,3] prediction
    :param pts2:  [pn,3] target
    :param K:     [3,3]
    :return:
    '''

    def get_pts_img(pts,R,T):
        img_pts,_=pts_to_img_pts(pts,np.identity(3),np.zeros(3),K)
        pts_img=img_pts_to_pts_img(img_pts,h,w)
        trans_pts=np.matmul(pts-T,R.transpose())+T
        trans_img_pts,_=pts_to_img_pts(trans_pts,np.identity(3),np.zeros(3),K)
        trans_pts_img=img_pts_to_pts_img(trans_img_pts,h,w)
        return pts_img,trans_pts_img

    def get_img(pts_img1,pts_img2):
        img=np.zeros([h,w,3],np.uint8)
        img[np.logical_and(pts_img1>0,pts_img2>0)]=np.asarray([0,255,0],np.uint8)
        img[np.logical_and(pts_img1>0,pts_img2==0)]=np.asarray([255,255,0],np.uint8)
        img[np.logical_and(pts_img1==0,pts_img2>0)]=np.asarray([255,0,0],np.uint8)
        return img


    T=np.mean(np.concatenate([pts1,pts2],0),0)[None,:]
    R=euler2mat(np.pi/2,0,0,'syzx')
    pts_img1, trans_pts_img1=get_pts_img(pts1,R,T)
    pts_img2, trans_pts_img2=get_pts_img(pts2,R,T)
    overlap1=get_img(pts_img1,pts_img2)
    overlap2=get_img(trans_pts_img1,trans_pts_img2)

    return overlap1,overlap2


def visualize_mask_multi_class(mask_pred, mask_gt, colors, save=False, save_fn=None):
    '''

    :param mask_pred:   [b,h,w]
    :param mask_gt:     [b,h,w]
    :param colors:      [cn,3]
    :return:
    '''
    b,h,w=mask_gt.shape
    cn,_=colors.shape
    for bi in range(b):
        img_pred=np.zeros([h,w,3],np.uint8)
        for ci in range(cn):
            img_pred[mask_pred[bi]==ci]=colors[ci]

        img_gt=np.zeros([h,w,3],np.uint8)
        for ci in range(cn):
            img_gt[mask_gt[bi]==ci]=colors[ci]

        plt.subplots(121)
        plt.imshow(img_pred)
        plt.subplots(122)
        plt.imshow(img_gt)
        if not save:
            plt.show()
        else:
            plt.savefig(save_fn.format(bi))
        plt.close()

def visualize_hypothesis(rgb, hyp_pts, hyp_counts, pts_target, save=False, save_fn=None):
    '''

    :param rgb:         b,h,w
    :param hyp_pts:     b,hn,vn,2
    :param hyp_counts:  b,hn,vn
    :param pts_target:  b,vn,2
    :param save:
    :param save_fn:
    :return:
    '''
    b,hn,vn,_=hyp_pts.shape
    _,h,w,_=rgb.shape
    for bi in range(b):
        for vi in range(vn):
            cur_hyp_counts=hyp_counts[bi,:,vi]  # [hn]
            cur_hyp_pts=hyp_pts[bi,:,vi]        # [hn,2]
            # mask=np.logical_and(np.logical_and(cur_hyp_pts[:,0]>-w*0.5,cur_hyp_pts[:,0]<w*1.5),
            #                     np.logical_and(cur_hyp_pts[:,1]>-h*0.5,cur_hyp_pts[:,1]<h*1.5))
            mask=np.logical_and(np.logical_and(cur_hyp_pts[:,0]>0,cur_hyp_pts[:,0]<w*1.0),
                                np.logical_and(cur_hyp_pts[:,1]>0,cur_hyp_pts[:,1]<h*1.0))
            cur_hyp_pts[np.logical_not(mask)]=0.0
            cur_hyp_counts[np.logical_not(mask)]=0

            cur_hyp_counts=cur_hyp_counts.astype(np.float32)
            colors=(cur_hyp_counts/cur_hyp_counts.max())#[:,None]#*np.array([[255,0,0]])
            plt.figure(figsize=(10,8))
            plt.imshow(rgb[bi])
            plt.scatter(cur_hyp_pts[:,0],cur_hyp_pts[:,1],c=colors,s=0.1,cmap='viridis')
            # plt.plot(pts_target[bi,vi,0],pts_target[bi,vi,1],'*',c='r')
            if save:
                plt.savefig(save_fn.format(bi,vi))
            else:
                plt.show()
            plt.close()

def visualize_voting_ellipse(rgb,mean,var,target,save=False, save_fn=None):
    '''

    :param rgb:     b,h,w,3
    :param mean:    b,vn,2
    :param var:     b,vn,2,2
    :param save:
    :param save_fn:
    :return:
    '''
    b,vn,_=mean.shape
    for bi in range(b):
        _, ax = plt.subplots(1)

        for vi in range(vn):
            cov=var[bi,vi]
            w,v=np.linalg.eig(cov)
            w*=50
            elp=patches.Ellipse(mean[bi,vi],w[0],w[1],np.arctan2(v[1,0],v[0,0])/np.pi*180,fill=False)
            ax.add_patch(elp)

        ax.plot(target[bi,:,0],target[bi,:,1],'*')
        ax.scatter(mean[bi,:,0],mean[bi,:,1],c=np.arange(vn))
        ax.imshow(rgb[bi])
        if save:
            plt.savefig(save_fn.format(bi))
        else:
            plt.show()
        plt.close()




def visualize_vanishing_points(rgb, van_cens, save=False, save_fn=None):
    b,h,w,_=rgb.shape
    cen=van_cens[:,3,:]  # [b,3]
    van=van_cens[:,:3,:] # [b,3,3]
    cen/=cen[:,2:]

    for bi in range(b):
        dir_2d=[]
        for di in range(3):
            dir=(van[bi,di,:]-cen[bi]*van[bi,di,2])[:2]
            dir/=np.linalg.norm(dir)
            dir_2d.append(dir)

        dir_2d=np.asarray(dir_2d)*20 # [4,2]
        _, ax = plt.subplots(1)
        ax.imshow(rgb[bi])

        ax.add_patch(patches.Arrow(x=cen[bi,0],y=cen[bi,1],dx=dir_2d[0,0],dy=dir_2d[0,1],linewidth=2,edgecolor='r'))
        ax.add_patch(patches.Arrow(x=cen[bi,0],y=cen[bi,1],dx=dir_2d[1,0],dy=dir_2d[1,1],linewidth=2,edgecolor='g'))
        ax.add_patch(patches.Arrow(x=cen[bi,0],y=cen[bi,1],dx=dir_2d[2,0],dy=dir_2d[2,1],linewidth=2,edgecolor='b'))
        if save:
            plt.savefig(save_fn.format(bi))
        else:
            plt.show()
        plt.close()

def visualize_points(rgb, pts_target, pts_pred=None, save=False, save_fn=None):
    '''

    :param rgb:             torch tensor with size [b,3,h,w] or numpy array with size [b,h,w,3]
    :param pts_target: [b,pn,2]
    :param pts_pred:   [b,pn,2]
    :param save:
    :param save_fn:
    :return:
    '''
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.permute(0, 2, 3, 1).detach().cpu().numpy()
    rgb = rgb.astype(np.uint8)

    batch_size = pts_target.shape[0]
    for idx in range(batch_size):
        _, ax = plt.subplots(1)
        ax.imshow(rgb[idx])
        ax.plot(pts_target[idx,:,0],pts_target[idx,:,1],'*')
        if pts_pred is not None:
            ax.plot(pts_pred[idx,:,0],pts_pred[idx,:,1],'*')
        if not save:
            plt.show()
        else:
            plt.savefig(save_fn.format(idx))
        plt.close()

def visualize_keypoints(rgb, pts_target, pts_pred=None, save=False, save_fn=None):
    rgb=rgb.astype(np.uint8)

    batch_size=rgb.shape[0]
    for bi in range(batch_size):
        _, ax = plt.subplots(1)
        ax.imshow(rgb[bi])
        ax.scatter(pts_target[bi,:,0],pts_target[bi,:,1],c=np.arange(pts_target.shape[1]))
        if pts_pred is not None:
            ax.scatter(pts_pred[bi,:,0],pts_pred[bi,:,1],c=np.arange(pts_pred.shape[1]))
        if not save:
            plt.show()
        else:
            plt.savefig(save_fn.format(bi))
        plt.close()


def imagenet_to_uint8(rgb,torch_format=True):
    '''

    :param rgb: [b,3,h,w]
    :return:
    '''
    if torch_format:
        if len(rgb.shape)==4:
            rgb = rgb.transpose(0, 2, 3, 1)
        else:
            rgb = rgb.transpose(1, 2, 0)
    rgb *= np.asarray([0.229, 0.224, 0.225])[None, None, :]
    rgb += np.asarray([0.485, 0.456, 0.406])[None, None, :]
    rgb *= 255
    rgb = rgb.astype(np.uint8)

    return rgb

def write_points(filename, pts, colors=None):
    has_color=pts.shape[1]>=6
    with open(filename, 'w') as f:
        for i,pt in enumerate(pts):
            if colors is None:
                if has_color:
                    f.write('{} {} {} {} {} {}\n'.format(pt[0],pt[1],pt[2],int(pt[3]),int(pt[4]),int(pt[5])))
                else:
                    f.write('{} {} {}\n'.format(pt[0],pt[1],pt[2]))

            else:
                if colors.shape[0]==pts.shape[0]:
                    f.write('{} {} {} {} {} {}\n'.format(pt[0],pt[1],pt[2],int(colors[i,0]),int(colors[i,1]),int(colors[i,2])))
                else:
                    f.write('{} {} {} {} {} {}\n'.format(pt[0],pt[1],pt[2],int(colors[0]),int(colors[1]),int(colors[2])))

def img_pts_to_pts_img(im_pts, img_row, img_col):
    pts_img=np.zeros([img_row,img_col],dtype=np.float32)
    for pt in im_pts:
        x,y = round(pt[0]), round(pt[1])
        x=int(img_col-1 if x>=img_col else x)
        y=int(img_row-1 if y>=img_row else y)
        x=0 if x<0 else x
        y=0 if y<0 else y
        pts_img[y,x]=1.0

    return pts_img

def img_pts_to_pts_img_colors(img, im_pts, img_rgbs):
    pts_img=img.copy()
    img_row,img_col,_=img.shape
    for pt,rgb in zip(im_pts,img_rgbs):
        x,y = round(pt[0]), round(pt[1])
        x=int(img_col-1 if x>=img_col else x)
        y=int(img_row-1 if y>=img_row else y)
        x=0 if x<0 else x
        y=0 if y<0 else y
        pts_img[y,x]=rgb

    return pts_img

def pts_to_img_pts(pts,R,T,K):
    img_pts=np.matmul(np.matmul(pts,R.transpose())+T[None,:],K.transpose())
    img_dpt=img_pts[:,2]
    img_pts=img_pts[:,:2]/img_pts[:,2:]
    return img_pts,img_dpt
