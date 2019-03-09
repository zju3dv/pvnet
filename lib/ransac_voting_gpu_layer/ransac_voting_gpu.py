import torch
import lib.ransac_voting_gpu_layer.ransac_voting as ransac_voting
import numpy as np

def log_msg(msg):
    # with open('ransac.log', 'a') as f:
    #     f.write(msg+'\n')
    pass

def ransac_voting_layer(mask, vertex, class_num, round_hyp_num, inlier_thresh=0.999, confidence=0.99, max_iter=20,
                        min_num=5,max_num=30000):
    '''
    :param mask:      [b,h,w]
    :param vertex:    [b,h,w,vn,2]
    :param class_num:
    :param round_hyp_num:
    :param inlier_thresh:
    :return:
    '''
    log_msg('ransac begin')
    b,h,w,vn,_=vertex.shape
    batch_win_pts=[]
    for bi in range(b):
        class_win_pts = []
        hyp_num=0
        for k in range(class_num-1):
            cur_mask=mask[bi]==k+1
            foreground=torch.sum(cur_mask)
            log_msg('get sum')

            # if too few points, just skip it
            if foreground<min_num:
                all_win_pts=torch.zeros([vn,2],dtype=torch.float32,device=mask.device)
                class_win_pts.append(torch.unsqueeze(all_win_pts,0)) # [1,vn,2]
                continue

            # if too many inliers, we randomly down sample it
            if foreground>max_num:
                selection=torch.zeros(cur_mask.shape,dtype=torch.float32,device=mask.device).uniform_(0,1)
                selected_mask=(selection<(max_num/foreground.float()))
                cur_mask*=selected_mask

            log_msg('test done')
            coords=torch.nonzero(cur_mask).float()   # [tn,2]
            coords=coords[:,[1,0]]
            log_msg('nonzero')
            direct=vertex[bi].masked_select(torch.unsqueeze(torch.unsqueeze(cur_mask,2),3)) # [tn,vn,2]
            direct=direct.view([coords.shape[0],vn,2])
            log_msg('mask select')
            tn=coords.shape[0]
            idxs=torch.zeros([round_hyp_num, vn, 2], dtype=torch.int32, device=mask.device).random_(0, direct.shape[0])
            log_msg('random sample')
            all_win_ratio=torch.zeros([vn],dtype=torch.float32,device=mask.device)
            all_win_pts=torch.zeros([vn,2],dtype=torch.float32,device=mask.device)
            log_msg('zeros')

            cur_iter=0
            while True:
                # generate hypothesis
                cur_hyp_pts=ransac_voting.generate_hypothesis(direct, coords, idxs) # [hn,vn,2]
                log_msg('generate_hypothesis')

                # voting for hypothesis
                cur_inlier = torch.zeros([round_hyp_num, vn, tn], dtype=torch.uint8, device=mask.device)
                ransac_voting.voting_for_hypothesis(direct, coords, cur_hyp_pts, cur_inlier, inlier_thresh) # [hn,vn,tn]
                log_msg('voting_for_hypothesis')

                # find max
                cur_inlier_counts=torch.sum(cur_inlier,2)    # [hn,vn]
                cur_win_counts,cur_win_idx=torch.max(cur_inlier_counts,0) # [vn]
                cur_win_pts=cur_hyp_pts[cur_win_idx, torch.arange(vn)]
                cur_win_ratio=cur_win_counts.float()/tn
                log_msg('find max')


                larger_mask=all_win_ratio<cur_win_ratio
                all_win_pts[larger_mask,:]=cur_win_pts[larger_mask,:]
                all_win_ratio[larger_mask]=cur_win_ratio[larger_mask]
                log_msg('mask larger')

                hyp_num+=round_hyp_num
                cur_iter+=1
                cur_min_ratio=torch.min(all_win_ratio)
                # print('cur_min_ratio {} cur_confidence {}'.format(cur_min_ratio,(1-(1-cur_min_ratio**2)**hyp_num)))
                log_msg('check condition')
                if (1-(1-cur_min_ratio**2)**hyp_num)>confidence or cur_iter>max_iter:
                    break

            class_win_pts.append(torch.unsqueeze(all_win_pts,0)) # [1,vn,2]

        batch_win_pts.append(torch.unsqueeze(torch.cat(class_win_pts,0),0)) # [1,cn,vn,2]
        log_msg('class append')

    batch_win_pts=torch.cat(batch_win_pts,0)
    log_msg('batch append')

    return batch_win_pts

def ransac_voting_layer_v2(mask, vertex, class_num, round_hyp_num, inlier_thresh=0.999, confidence=0.99, max_iter=20,
                           min_num=5,max_num=30000,refine_iter_num=1):
    '''
    :param mask:      [b,h,w]
    :param vertex:    [b,h,w,vn,2]
    :param class_num:
    :param round_hyp_num:
    :param inlier_thresh:
    :return: [b,cn,vn,2]
    '''

    log_msg('ransac begin')
    b,h,w,vn,_=vertex.shape
    batch_win_pts=[]
    for bi in range(b):
        class_win_pts = []
        hyp_num=0
        for k in range(class_num-1):
            cur_mask=mask[bi]==k+1
            foreground=torch.sum(cur_mask)
            log_msg('get sum')

            # if too few points, just skip it
            if foreground<min_num:
                all_win_pts=torch.zeros([vn,2],dtype=torch.float32,device=mask.device)
                class_win_pts.append(torch.unsqueeze(all_win_pts,0)) # [1,vn,2]
                continue

            # if too many inliers, we randomly down sample it
            if foreground>max_num:
                selection=torch.zeros(cur_mask.shape,dtype=torch.float32,device=mask.device).uniform_(0,1)
                selected_mask=(selection<(max_num/foreground.float()))
                cur_mask*=selected_mask

            log_msg('test done')
            coords=torch.nonzero(cur_mask).float()   # [tn,2]
            coords=coords[:,[1,0]]
            log_msg('nonzero')
            direct=vertex[bi].masked_select(torch.unsqueeze(torch.unsqueeze(cur_mask,2),3)) # [tn,vn,2]
            direct=direct.view([coords.shape[0],vn,2])
            log_msg('mask select')
            tn=coords.shape[0]
            idxs=torch.zeros([round_hyp_num, vn, 2], dtype=torch.int32, device=mask.device).random_(0, direct.shape[0])
            log_msg('random sample')

            all_win_ratio=torch.zeros([vn],dtype=torch.float32,device=mask.device)
            all_win_pts=torch.zeros([vn,2],dtype=torch.float32,device=mask.device)
            log_msg('zeros')

            cur_iter=0
            while True:
                # generate hypothesis
                cur_hyp_pts=ransac_voting.generate_hypothesis(direct, coords, idxs) # [hn,vn,2]
                log_msg('generate_hypothesis')

                # voting for hypothesis
                cur_inlier = torch.zeros([round_hyp_num, vn, tn], dtype=torch.uint8, device=mask.device)
                ransac_voting.voting_for_hypothesis(direct, coords, cur_hyp_pts, cur_inlier, inlier_thresh) # [hn,vn,tn]
                log_msg('voting_for_hypothesis')

                # find max
                cur_inlier_counts=torch.sum(cur_inlier,2)    # [hn,vn]
                cur_win_counts,cur_win_idx=torch.max(cur_inlier_counts,0) # [vn]
                cur_win_pts=cur_hyp_pts[cur_win_idx, torch.arange(vn)]
                cur_win_ratio=cur_win_counts.float()/tn
                log_msg('find max')


                larger_mask=all_win_ratio<cur_win_ratio
                all_win_pts[larger_mask,:]=cur_win_pts[larger_mask,:]
                all_win_ratio[larger_mask]=cur_win_ratio[larger_mask]
                log_msg('mask larger')

                hyp_num+=round_hyp_num
                cur_iter+=1
                cur_min_ratio=torch.min(all_win_ratio)
                log_msg('check condition')
                if (1-(1-cur_min_ratio**2)**hyp_num)>confidence or cur_iter>max_iter:
                    break

            normal=torch.zeros_like(direct)
            normal[:,:,0]=direct[:,:,1]
            normal[:,:,1]=-direct[:,:,0]
            # compute mean intersection again
            for k in range(refine_iter_num):
                all_inlier = torch.zeros([1, vn, tn], dtype=torch.uint8, device=mask.device)
                all_win_pts=torch.unsqueeze(all_win_pts,0) # [1,vn,2]
                ransac_voting.voting_for_hypothesis(direct, coords, all_win_pts, all_inlier, inlier_thresh)  # [1,vn,tn]


                log_msg('refine voting')
                refine_pts=[]
                for vi in range(vn):
                    cur_coords=coords[all_inlier[0,vi]] # in,2
                    if cur_coords.shape[0]==0:
                        refine_pts.append(torch.zeros([1,2]).cuda())
                        continue
                    cur_normal=normal[:,vi,:][all_inlier[0,vi]] # in,2

                    A=cur_normal                                   # [cn,2]
                    b=torch.sum(cur_normal*cur_coords,1)           # [cn]
                    refine_pt=torch.matmul(torch.pinverse(A),b)    # [2]
                    refine_pts.append(torch.unsqueeze(refine_pt,0))
                    log_msg('invers ')

                refine_pts=torch.cat(refine_pts,0)
                all_win_pts=refine_pts

            class_win_pts.append(torch.unsqueeze(all_win_pts,0)) # [1,vn,2]

        batch_win_pts.append(torch.unsqueeze(torch.cat(class_win_pts,0),0)) # [1,cn,vn,2]
        log_msg('class append')

    batch_win_pts=torch.cat(batch_win_pts,0)
    log_msg('batch append')

    return batch_win_pts


def ransac_voting_hypothesis(mask, vertex, round_hyp_num, inlier_thresh=0.999, min_num=5, max_num=30000):
    b, h, w, vn, _ = vertex.shape
    all_hyp_pts,all_inlier_counts=[],[]
    for bi in range(b):
        k=0
        cur_mask = mask[bi] == k + 1
        foreground = torch.sum(cur_mask)

        # if too few points, just skip it
        if foreground < min_num:
            cur_hyp_pts = torch.zeros([1, round_hyp_num, vn, 2], dtype=torch.float32, device=mask.device)
            all_hyp_pts.append(cur_hyp_pts)  # [1,vn,2]
            cur_inlier_counts = torch.ones([1, round_hyp_num, vn], dtype=torch.int64, device=mask.device).long()
            all_inlier_counts.append(cur_inlier_counts)
            continue

        # if too many inliers, we randomly down sample it
        if foreground > max_num:
            selection = torch.zeros(cur_mask.shape, dtype=torch.float32, device=mask.device).uniform_(0, 1)
            selected_mask = (selection < (max_num / foreground.float()))
            cur_mask *= selected_mask

        coords = torch.nonzero(cur_mask).float()  # [tn,2]
        coords = coords[:, [1, 0]]
        direct = vertex[bi].masked_select(torch.unsqueeze(torch.unsqueeze(cur_mask, 2), 3))  # [tn,vn,2]
        direct = direct.view([coords.shape[0], vn, 2])
        tn = coords.shape[0]
        idxs = torch.zeros([round_hyp_num, vn, 2], dtype=torch.int32, device=mask.device).random_(0,
                                                                                                  direct.shape[0])

        # generate hypothesis
        cur_hyp_pts = ransac_voting.generate_hypothesis(direct, coords, idxs)  # [hn,vn,2]

        # voting for hypothesis
        cur_inlier = torch.zeros([round_hyp_num, vn, tn], dtype=torch.uint8, device=mask.device)
        ransac_voting.voting_for_hypothesis(direct, coords, cur_hyp_pts, cur_inlier, inlier_thresh)  # [hn,vn,tn]
        cur_inlier_counts = torch.sum(cur_inlier, 2)  # [hn,vn]

        all_hyp_pts.append(torch.unsqueeze(cur_hyp_pts,0))
        all_inlier_counts.append(torch.unsqueeze(cur_inlier_counts,0))

    all_inlier_counts=torch.cat(all_inlier_counts, 0)

    return torch.cat(all_hyp_pts,0), all_inlier_counts  # [b,hn,vn,2] [b,hn,vn]

def estimate_voting_distribution(mask, vertex, round_hyp_num=256, min_hyp_num=4096, topk=128,
                                 inlier_thresh=0.99, min_num=5, max_num=30000):
    b, h, w, vn, _ = vertex.shape
    all_hyp_pts,all_inlier_ratio=[],[]
    for bi in range(b):
        k=0
        cur_mask = mask[bi] == k + 1
        foreground = torch.sum(cur_mask)

        # if too few points, just skip it
        if foreground < min_num:
            cur_hyp_pts = torch.zeros([1, round_hyp_num, vn, 2], dtype=torch.float32, device=mask.device).float()
            all_hyp_pts.append(cur_hyp_pts)  # [1,vn,2]
            cur_inlier_ratio = torch.ones([1, round_hyp_num, vn], dtype=torch.int64, device=mask.device).float()
            all_inlier_ratio.append(cur_inlier_ratio)
            continue

        # if too many inliers, we randomly down sample it
        if foreground > max_num:
            selection = torch.zeros(cur_mask.shape, dtype=torch.float32, device=mask.device).uniform_(0, 1)
            selected_mask = (selection < (max_num / foreground.float()))
            cur_mask *= selected_mask
            foreground = torch.sum(cur_mask)

        coords = torch.nonzero(cur_mask).float()  # [tn,2]
        coords = coords[:, [1, 0]]
        direct = vertex[bi].masked_select(torch.unsqueeze(torch.unsqueeze(cur_mask, 2), 3))  # [tn,vn,2]
        direct = direct.view([coords.shape[0], vn, 2])
        tn = coords.shape[0]

        round_num=np.ceil(min_hyp_num/round_hyp_num)
        cur_hyp_pts=[]
        cur_inlier_ratio=[]
        for round_idx in range(int(round_num)):
            idxs = torch.zeros([round_hyp_num, vn, 2], dtype=torch.int32, device=mask.device).random_(0, direct.shape[0])

            # generate hypothesis
            hyp_pts = ransac_voting.generate_hypothesis(direct, coords, idxs)  # [hn,vn,2]

            # voting for hypothesis
            inlier = torch.zeros([round_hyp_num, vn, tn], dtype=torch.uint8, device=mask.device)
            ransac_voting.voting_for_hypothesis(direct, coords, hyp_pts, inlier, inlier_thresh)  # [hn,vn,tn]
            inlier_ratio = torch.sum(inlier, 2)                     # [hn,vn]
            inlier_ratio=inlier_ratio.float()/foreground.float()    # ratio

            cur_hyp_pts.append(hyp_pts)
            cur_inlier_ratio.append(inlier_ratio)

        cur_hyp_pts=torch.cat(cur_hyp_pts,0)
        cur_inlier_ratio=torch.cat(cur_inlier_ratio,0)
        all_hyp_pts.append(torch.unsqueeze(cur_hyp_pts,0))
        all_inlier_ratio.append(torch.unsqueeze(cur_inlier_ratio,0))

    all_hyp_pts=torch.cat(all_hyp_pts, 0)               # b,hn,vn,2
    all_inlier_ratio=torch.cat(all_inlier_ratio, 0)     # b,hn,vn
    all_hyp_pts=all_hyp_pts.permute(0,2,1,3)            # b,vn,hn,2
    all_inlier_ratio=all_inlier_ratio.permute(0,2,1)    # b,vn,hn
    values, indexes=torch.topk(all_inlier_ratio,topk,dim=2,sorted=False)
    all_inlier_ratio=torch.zeros_like(all_inlier_ratio).scatter_(2, indexes, values)

    weighted_pts=torch.unsqueeze(all_inlier_ratio,3)*all_hyp_pts
    mean=torch.sum(weighted_pts,2)/torch.unsqueeze(torch.sum(all_inlier_ratio,2),2) # b,vn,2

    diff_pts=all_hyp_pts-torch.unsqueeze(mean,2)                  # b,vn,hn,2
    weighted_diff_pts = diff_pts * torch.unsqueeze(all_inlier_ratio, 3)
    cov=torch.matmul(diff_pts.transpose(2,3), weighted_diff_pts)  # b,vn,2,2
    cov/=torch.unsqueeze(torch.unsqueeze(torch.sum(all_inlier_ratio,2),2),3) # b,vn,2,2

    return mean,cov

def estimate_voting_distribution_with_mean(mask, vertex, mean, round_hyp_num=256, min_hyp_num=4096, topk=128,
                                           inlier_thresh=0.99, min_num=5, max_num=30000, output_hyp=False):
    b, h, w, vn, _ = vertex.shape
    all_hyp_pts,all_inlier_ratio=[],[]
    for bi in range(b):
        k=0
        cur_mask = mask[bi] == k + 1
        foreground = torch.sum(cur_mask)

        # if too few points, just skip it
        if foreground < min_num:
            cur_hyp_pts = torch.zeros([1, min_hyp_num, vn, 2], dtype=torch.float32, device=mask.device).float()
            all_hyp_pts.append(cur_hyp_pts)  # [1,vn,2]
            cur_inlier_ratio = torch.ones([1, min_hyp_num, vn], dtype=torch.int64, device=mask.device).float()
            all_inlier_ratio.append(cur_inlier_ratio)
            continue

        # if too many inliers, we randomly down sample it
        if foreground > max_num:
            selection = torch.zeros(cur_mask.shape, dtype=torch.float32, device=mask.device).uniform_(0, 1)
            selected_mask = (selection < (max_num / foreground.float()))
            cur_mask *= selected_mask
            foreground = torch.sum(cur_mask)

        coords = torch.nonzero(cur_mask).float()  # [tn,2]
        coords = coords[:, [1, 0]]
        direct = vertex[bi].masked_select(torch.unsqueeze(torch.unsqueeze(cur_mask, 2), 3))  # [tn,vn,2]
        direct = direct.view([coords.shape[0], vn, 2])
        tn = coords.shape[0]

        round_num=np.ceil(min_hyp_num/round_hyp_num)
        cur_hyp_pts=[]
        cur_inlier_ratio=[]
        for round_idx in range(int(round_num)):
            idxs = torch.zeros([round_hyp_num, vn, 2], dtype=torch.int32, device=mask.device).random_(0, direct.shape[0])

            # generate hypothesis
            hyp_pts = ransac_voting.generate_hypothesis(direct, coords, idxs)  # [hn,vn,2]

            # voting for hypothesis
            inlier = torch.zeros([round_hyp_num, vn, tn], dtype=torch.uint8, device=mask.device)
            ransac_voting.voting_for_hypothesis(direct, coords, hyp_pts, inlier, inlier_thresh)  # [hn,vn,tn]
            inlier_ratio = torch.sum(inlier, 2)                     # [hn,vn]
            inlier_ratio=inlier_ratio.float()/foreground.float()    # ratio

            cur_hyp_pts.append(hyp_pts)
            cur_inlier_ratio.append(inlier_ratio)

        cur_hyp_pts=torch.cat(cur_hyp_pts,0)
        cur_inlier_ratio=torch.cat(cur_inlier_ratio,0)
        all_hyp_pts.append(torch.unsqueeze(cur_hyp_pts,0))
        all_inlier_ratio.append(torch.unsqueeze(cur_inlier_ratio,0))

    all_hyp_pts=torch.cat(all_hyp_pts, 0)               # b,hn,vn,2
    all_inlier_ratio=torch.cat(all_inlier_ratio, 0)     # b,hn,vn

    # raw_hyp_pts=all_hyp_pts.permute(0,2,1,3).clone()
    # raw_hyp_ratio=all_inlier_ratio.permute(0,2,1).clone()

    all_hyp_pts=all_hyp_pts.permute(0,2,1,3)            # b,vn,hn,2
    all_inlier_ratio=all_inlier_ratio.permute(0,2,1)    # b,vn,hn
    thresh=torch.max(all_inlier_ratio,2)[0]-0.1         # b,vn
    all_inlier_ratio[all_inlier_ratio<torch.unsqueeze(thresh,2)]=0.0


    diff_pts=all_hyp_pts-torch.unsqueeze(mean,2)                  # b,vn,hn,2
    weighted_diff_pts = diff_pts * torch.unsqueeze(all_inlier_ratio, 3)
    cov=torch.matmul(diff_pts.transpose(2,3), weighted_diff_pts)  # b,vn,2,2
    cov/=torch.unsqueeze(torch.unsqueeze(torch.sum(all_inlier_ratio,2),2),3)+1e-3 # b,vn,2,2

    # if output_hyp:
    #     return mean,cov,all_hyp_pts,all_inlier_ratio,raw_hyp_pts,raw_hyp_ratio

    return mean, cov

def ransac_voting_vanish_point_layer(mask, vertex, round_hyp_num, inlier_thresh=0.999,
                                     confidence=0.99, max_iter=20, min_num=5,max_num=30000,refine_iter_num=1):
    b,h,w,vn,_=vertex.shape
    batch_win_pts=[]
    for bi in range(b):
        class_win_pts = []
        hyp_num=0
        for k in range(class_num-1):
            cur_mask=mask[bi]==k+1
            foreground=torch.sum(cur_mask)

            # if too few points, just skip it
            if foreground<min_num:
                all_win_pts=torch.zeros([vn,2],dtype=torch.float32,device=mask.device)
                class_win_pts.append(torch.unsqueeze(all_win_pts,0)) # [1,vn,2]
                continue

            # if too many inliers, we randomly down sample it
            if foreground>max_num:
                selection=torch.zeros(cur_mask.shape,dtype=torch.float32,device=mask.device).uniform_(0,1)
                selected_mask=(selection<(max_num/foreground.float()))
                cur_mask*=selected_mask

            coords=torch.nonzero(cur_mask).float()   # [tn,2]
            coords=coords[:,[1,0]]

            direct=vertex[bi].masked_select(torch.unsqueeze(torch.unsqueeze(cur_mask,2),3)) # [tn,vn,2]
            direct=direct.view([coords.shape[0],vn,2])

            tn=coords.shape[0]
            idxs=torch.zeros([round_hyp_num, vn, 2], dtype=torch.int32, device=mask.device).random_(0, direct.shape[0])

            all_win_ratio=torch.zeros([vn],dtype=torch.float32,device=mask.device)
            all_win_pts=torch.zeros([vn,3],dtype=torch.float32,device=mask.device)

            cur_iter=0
            while True:
                # generate hypothesis
                cur_hyp_pts=ransac_voting.generate_hypothesis_vanishing_point(direct, coords, idxs) # [hn,vn,3]

                # voting for hypothesis
                cur_inlier = torch.zeros([round_hyp_num, vn, tn], dtype=torch.uint8, device=mask.device)
                ransac_voting.voting_for_hypothesis_vanishing_point(direct, coords, cur_hyp_pts,
                                                                    cur_inlier, inlier_thresh) # [hn,vn,tn]
                cur_hyp_pts/=torch.norm(cur_hyp_pts,2,2,keepdim=True)
                # find max
                cur_inlier_counts=torch.sum(cur_inlier,2)    # [hn,vn]
                cur_win_counts,cur_win_idx=torch.max(cur_inlier_counts,0) # [vn]
                cur_win_pts=cur_hyp_pts[cur_win_idx, torch.arange(vn)]
                cur_win_ratio=cur_win_counts.float()/tn

                larger_mask=all_win_ratio<cur_win_ratio
                all_win_pts[larger_mask,:]=cur_win_pts[larger_mask,:]
                all_win_ratio[larger_mask]=cur_win_ratio[larger_mask]

                hyp_num+=round_hyp_num
                cur_iter+=1
                cur_min_ratio=torch.min(all_win_ratio)
                if (1-(1-cur_min_ratio**2)**hyp_num)>confidence or cur_iter>max_iter:
                    break

            normal = torch.zeros_like(direct)
            normal[:, :, 0] = direct[:, :, 1]
            normal[:, :, 1] = -direct[:, :, 0]
            # compute mean intersection again
            for k in range(refine_iter_num):
                all_inlier = torch.zeros([1, vn, tn], dtype=torch.uint8, device=mask.device)
                all_win_pts = torch.unsqueeze(all_win_pts, 0)  # [1,vn,3]
                ransac_voting.voting_for_hypothesis_vanishing_point(direct, coords, all_win_pts, all_inlier,
                                                                    inlier_thresh)  # [1,vn,tn]
                refine_pts = []
                for vi in range(vn):
                    cur_coords = coords[all_inlier[0, vi]]  # in,2
                    cur_normal = normal[:, vi, :][all_inlier[0, vi]]  # in,2
                    H=torch.cat([-cur_normal,torch.unsqueeze(torch.sum(cur_normal*cur_coords,1),1)],1) # in,3
                    U, S, V=torch.svd(H,some=True) # 3,3
                    refine_pt=V[:,2:].transpose(0,1)
                    # correct direction
                    if (refine_pt[0,0]-refine_pt[0,2]*cur_coords[0,0])*(-cur_normal[0,1])<0:
                        refine_pt=-refine_pt
                    refine_pts.append(refine_pt)

                refine_pts = torch.cat(refine_pts, 0)
                all_win_pts = refine_pts

            class_win_pts.append(torch.unsqueeze(all_win_pts, 0))  # [1,vn,2]

        batch_win_pts.append(torch.unsqueeze(torch.cat(class_win_pts, 0), 0))  # [1,cn,vn,2]
        log_msg('class append')

    batch_win_pts = torch.cat(batch_win_pts, 0)
    log_msg('batch append')

    return batch_win_pts

def b_inv(b_mat):
    '''
    code from
    https://stackoverflow.com/questions/46595157/how-to-apply-the-torch-inverse-function-of-pytorch-to-every-sample-in-the-batc
    :param b_mat:
    :return:
    '''
    eye = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat)
    b_inv, _ = torch.gesv(eye, b_mat)
    return b_inv

def ransac_voting_layer_v3(mask, vertex, round_hyp_num, inlier_thresh=0.999, confidence=0.99, max_iter=20,
                           min_num=5, max_num=30000):
    '''
    :param mask:      [b,h,w]
    :param vertex:    [b,h,w,vn,2]
    :param round_hyp_num:
    :param inlier_thresh:
    :return: [b,vn,2]
    '''
    b, h, w, vn, _ = vertex.shape
    batch_win_pts = []
    for bi in range(b):
        hyp_num = 0
        cur_mask = (mask[bi]).byte()
        foreground_num = torch.sum(cur_mask)

        # if too few points, just skip it
        if foreground_num < min_num:
            win_pts = torch.zeros([1, vn, 2], dtype=torch.float32, device=mask.device)
            batch_win_pts.append(win_pts)  # [1,vn,2]
            continue

        # if too many inliers, we randomly down sample it
        if foreground_num > max_num:
            selection = torch.zeros(cur_mask.shape, dtype=torch.float32, device=mask.device).uniform_(0, 1)
            selected_mask = (selection < (max_num / foreground_num.float()))
            cur_mask *= selected_mask

        coords = torch.nonzero(cur_mask).float()  # [tn,2]
        coords = coords[:, [1, 0]]
        direct = vertex[bi].masked_select(torch.unsqueeze(torch.unsqueeze(cur_mask, 2), 3))  # [tn,vn,2]
        direct = direct.view([coords.shape[0], vn, 2])
        tn = coords.shape[0]
        idxs = torch.zeros([round_hyp_num, vn, 2], dtype=torch.int32, device=mask.device).random_(0, direct.shape[0])
        all_win_ratio = torch.zeros([vn], dtype=torch.float32, device=mask.device)
        all_win_pts = torch.zeros([vn, 2], dtype=torch.float32, device=mask.device)

        cur_iter = 0
        while True:
            # generate hypothesis
            cur_hyp_pts = ransac_voting.generate_hypothesis(direct, coords, idxs)  # [hn,vn,2]

            # voting for hypothesis
            cur_inlier = torch.zeros([round_hyp_num, vn, tn], dtype=torch.uint8, device=mask.device)
            ransac_voting.voting_for_hypothesis(direct, coords, cur_hyp_pts, cur_inlier, inlier_thresh)  # [hn,vn,tn]

            # find max
            cur_inlier_counts = torch.sum(cur_inlier, 2)                   # [hn,vn]
            cur_win_counts, cur_win_idx = torch.max(cur_inlier_counts, 0)  # [vn]
            cur_win_pts = cur_hyp_pts[cur_win_idx, torch.arange(vn)]
            cur_win_ratio = cur_win_counts.float() / tn

            # update best point
            larger_mask = all_win_ratio < cur_win_ratio
            all_win_pts[larger_mask, :] = cur_win_pts[larger_mask, :]
            all_win_ratio[larger_mask] = cur_win_ratio[larger_mask]

            # check confidence
            hyp_num += round_hyp_num
            cur_iter += 1
            cur_min_ratio = torch.min(all_win_ratio)
            if (1 - (1 - cur_min_ratio ** 2) ** hyp_num) > confidence or cur_iter > max_iter:
                break

        # compute mean intersection again
        normal = torch.zeros_like(direct)   # [tn,vn,2]
        normal[:, :, 0] = direct[:, :, 1]
        normal[:, :, 1] = -direct[:, :, 0]
        all_inlier = torch.zeros([1, vn, tn], dtype=torch.uint8, device=mask.device)
        all_win_pts = torch.unsqueeze(all_win_pts, 0)  # [1,vn,2]
        ransac_voting.voting_for_hypothesis(direct, coords, all_win_pts, all_inlier, inlier_thresh)  # [1,vn,tn]

        # coords [tn,2] normal [vn,tn,2]
        all_inlier=torch.squeeze(all_inlier.float(),0)              # [vn,tn]
        normal=normal.permute(1,0,2)                                # [vn,tn,2]
        normal=normal*torch.unsqueeze(all_inlier,2)                 # [vn,tn,2] outlier is all zero

        b=torch.sum(normal*torch.unsqueeze(coords,0),2)             # [vn,tn]
        ATA=torch.matmul(normal.permute(0,2,1),normal)              # [vn,2,2]
        ATb=torch.sum(normal*torch.unsqueeze(b,2),1)                # [vn,2]
        all_win_pts=torch.matmul(b_inv(ATA),torch.unsqueeze(ATb,2)) # [vn,2,1]
        batch_win_pts.append(all_win_pts[None,:,:,0])

    batch_win_pts=torch.cat(batch_win_pts)
    return batch_win_pts

def ransac_voting_center(mask, vertex, round_hyp_num, inlier_thresh=0.99, confidence=0.999, max_iter=20, min_num=100):
    '''
    :param mask:      [b,h,w]
    :param vertex:    [b,h,w,2]
    :param round_hyp_num:
    :param inlier_thresh:
    :return: batch_instance_mask [b,h,w] max_instance_num [b]
    '''
    b, h, w, _ = vertex.shape
    vn=1
    batch_instance_mask = []
    batch_instance_num = []
    for bi in range(b):
        hyp_num = 0
        cur_mask = (mask[bi]).byte()
        foreground_num = torch.sum(cur_mask)

        # if too few points, just skip it
        if foreground_num < min_num:
            instance_mask = torch.zeros([h,w], dtype=torch.float32, device=mask.device)
            batch_instance_mask.append(instance_mask)
            batch_instance_num.append(0)
            continue

        coords_int = torch.nonzero(cur_mask)
        coords = coords_int.float()  # [tn,2]
        coords = coords[:, [1, 0]]
        direct = vertex[bi].masked_select(torch.unsqueeze(torch.unsqueeze(cur_mask, 2), 3))  # [tn,2]
        direct = direct.view([coords.shape[0], 1, 2])
        tn = coords.shape[0]
        idxs = torch.zeros([round_hyp_num, 1, 2], dtype=torch.int32, device=mask.device).random_(0, direct.shape[0])
        all_win_ratio = torch.zeros([1], dtype=torch.float32, device=mask.device)
        all_win_pts = torch.zeros([1, 2], dtype=torch.float32, device=mask.device)

        cur_iter = 0
        while True:
            # generate hypothesis
            cur_hyp_pts = ransac_voting.generate_hypothesis(direct, coords, idxs)  # [hn,1,2]

            # voting for hypothesis
            cur_inlier = torch.zeros([round_hyp_num, vn, tn], dtype=torch.uint8, device=mask.device)
            ransac_voting.voting_for_hypothesis(direct, coords, cur_hyp_pts, cur_inlier, inlier_thresh)  # [hn,1,tn]

            # find max
            cur_inlier_counts = torch.sum(cur_inlier, 2)                   # [hn,1]
            cur_win_counts, cur_win_idx = torch.max(cur_inlier_counts, 0)  # [1]
            cur_win_pts = cur_hyp_pts[cur_win_idx, torch.arange(vn)]
            cur_win_ratio = cur_win_counts.float() / tn

            # update best point
            larger_mask = all_win_ratio < cur_win_ratio
            all_win_pts[larger_mask, :] = cur_win_pts[larger_mask, :]
            all_win_ratio[larger_mask] = cur_win_ratio[larger_mask]

            # check confidence
            hyp_num += round_hyp_num
            cur_iter += 1
            cur_min_ratio = torch.min(all_win_ratio)
            if (1 - (1 - cur_min_ratio ** 2) ** hyp_num) > confidence or cur_iter > max_iter:
                break

        # compute mean intersection again
        all_inlier = torch.zeros([1, vn, tn], dtype=torch.uint8, device=mask.device)
        all_win_pts = torch.unsqueeze(all_win_pts, 0)
        ransac_voting.voting_for_hypothesis(direct, coords, all_win_pts, all_inlier, inlier_thresh)
        all_inlier=torch.squeeze(all_inlier.float(),0)              # [tn]

    return batch_instance_mask

def ransac_voting_layer_v4(mask, vertex, round_hyp_num, inlier_thresh=0.99, confidence=0.999, max_iter=20,
                           min_num=5, max_num=30000):
    '''
    :param mask:      [b,h,w]
    :param vertex:    [b,h,w,vn,2]
    :param round_hyp_num:
    :param inlier_thresh:
    :return: [b,vn,2]
    '''
    b, h, w, vn, _ = vertex.shape
    batch_win_pts = []
    batch_var = []
    for bi in range(b):
        hyp_num = 0
        cur_mask = (mask[bi]).byte()
        foreground_num = torch.sum(cur_mask)

        # if too few points, just skip it
        if foreground_num < min_num:
            win_pts = torch.zeros([1, vn, 2], dtype=torch.float32, device=mask.device)
            batch_win_pts.append(win_pts)  # [1,vn,2]
            batch_var.append(torch.ones([1,vn],dtype=torch.float32, device=mask.device))
            continue

        # if too many inliers, we randomly down sample it
        if foreground_num > max_num:
            selection = torch.zeros(cur_mask.shape, dtype=torch.float32, device=mask.device).uniform_(0, 1)
            selected_mask = (selection < (max_num / foreground_num.float()))
            cur_mask *= selected_mask

        coords = torch.nonzero(cur_mask).float()  # [tn,2]
        coords = coords[:, [1, 0]]
        direct = vertex[bi].masked_select(torch.unsqueeze(torch.unsqueeze(cur_mask, 2), 3))  # [tn,vn,2]
        direct = direct.view([coords.shape[0], vn, 2])
        tn = coords.shape[0]
        idxs = torch.zeros([round_hyp_num, vn, 2], dtype=torch.int32, device=mask.device).random_(0, direct.shape[0])
        all_win_ratio = torch.zeros([vn], dtype=torch.float32, device=mask.device)
        all_win_pts = torch.zeros([vn, 2], dtype=torch.float32, device=mask.device)

        cur_iter = 0
        while True:
            # generate hypothesis
            cur_hyp_pts = ransac_voting.generate_hypothesis(direct, coords, idxs)  # [hn,vn,2]

            # voting for hypothesis
            cur_inlier = torch.zeros([round_hyp_num, vn, tn], dtype=torch.uint8, device=mask.device)
            ransac_voting.voting_for_hypothesis(direct, coords, cur_hyp_pts, cur_inlier, inlier_thresh)  # [hn,vn,tn]

            # find max
            cur_inlier_counts = torch.sum(cur_inlier, 2)                   # [hn,vn]
            cur_win_counts, cur_win_idx = torch.max(cur_inlier_counts, 0)  # [vn]
            cur_win_pts = cur_hyp_pts[cur_win_idx, torch.arange(vn)]
            cur_win_ratio = cur_win_counts.float() / tn

            # update best point
            larger_mask = all_win_ratio < cur_win_ratio
            all_win_pts[larger_mask, :] = cur_win_pts[larger_mask, :]
            all_win_ratio[larger_mask] = cur_win_ratio[larger_mask]

            # check confidence
            hyp_num += round_hyp_num
            cur_iter += 1
            cur_min_ratio = torch.min(all_win_ratio)
            if (1 - (1 - cur_min_ratio ** 2) ** hyp_num) > confidence or cur_iter > max_iter:
                break

        # compute mean intersection again
        normal = torch.zeros_like(direct)   # [tn,vn,2]
        normal[:, :, 0] = direct[:, :, 1]
        normal[:, :, 1] = -direct[:, :, 0]
        all_inlier = torch.zeros([1, vn, tn], dtype=torch.uint8, device=mask.device)
        all_win_pts = torch.unsqueeze(all_win_pts, 0)  # [1,vn,2]
        ransac_voting.voting_for_hypothesis(direct, coords, all_win_pts, all_inlier, inlier_thresh)  # [1,vn,tn]

        # coords [tn,2] normal [vn,tn,2]
        all_inlier=torch.squeeze(all_inlier.float(),0)              # [vn,tn]
        normal=normal.permute(1,0,2)                                # [vn,tn,2]
        normal=normal*torch.unsqueeze(all_inlier,2)                 # [vn,tn,2] outlier is all zero

        b=torch.sum(normal*torch.unsqueeze(coords,0),2)             # [vn,tn]
        ATA=torch.matmul(normal.permute(0,2,1),normal)              # [vn,2,2]
        ATb=torch.sum(normal*torch.unsqueeze(b,2),1)                # [vn,2]
        all_win_pts=torch.matmul(b_inv(ATA),torch.unsqueeze(ATb,2)) # [vn,2,1]
        residual=torch.matmul(normal,all_win_pts)[:,:,0]-b          # [vn,tn]
        var=torch.sum(residual**2,1)/torch.sum(all_inlier,1)        # [vn]

        batch_win_pts.append(all_win_pts[None,:,:,0])
        batch_var.append(var[None,:])

    batch_win_pts=torch.cat(batch_win_pts)
    batch_var=torch.cat(batch_var)
    return batch_win_pts, batch_var


def ransac_voting_layer_v5(mask, vertex, round_hyp_num, inlier_thresh=0.999, confidence=0.99, max_iter=20,
                           min_num=5, max_num=100):
    '''
    :param mask:      [b,h,w]
    :param vertex:    [b,h,w,vn,2]
    :param round_hyp_num:
    :param inlier_thresh:
    :return: [b,vn,2] [b,vn,2,2]
    '''
    b, h, w, vn, _ = vertex.shape
    batch_win_pts, batch_confidence = [], []
    for bi in range(b):
        hyp_num = 0
        cur_mask = (mask[bi]).byte()
        foreground_num = torch.sum(cur_mask)

        # if too few points, just skip it
        if foreground_num < min_num:
            win_pts = torch.zeros([1, vn, 2], dtype=torch.float32, device=mask.device)
            pts_conf = torch.zeros([1, vn], dtype=torch.float32, device=mask.device)
            batch_win_pts.append(win_pts)      # [1,vn,2]
            batch_confidence.append(pts_conf)  # [1, vn]
            continue

        # if too many inliers, we randomly down sample it
        if foreground_num > max_num:
            selection = torch.zeros(cur_mask.shape, dtype=torch.float32, device=mask.device).uniform_(0, 1)
            selected_mask = (selection < (max_num / foreground_num.float()))
            cur_mask *= selected_mask
            # print(torch.sum(cur_mask))

        coords = torch.nonzero(cur_mask).float()  # [tn,2]
        coords = coords[:, [1, 0]]
        direct = vertex[bi].masked_select(torch.unsqueeze(torch.unsqueeze(cur_mask, 2), 3))  # [tn,vn,2]
        direct = direct.view([coords.shape[0], vn, 2])
        tn = coords.shape[0]
        idxs = torch.zeros([round_hyp_num, vn, 2], dtype=torch.int32, device=mask.device).random_(0, direct.shape[0])
        all_win_ratio = torch.zeros([vn], dtype=torch.float32, device=mask.device)
        all_win_pts = torch.zeros([vn, 2], dtype=torch.float32, device=mask.device)

        cur_iter = 0
        while True:
            # generate hypothesis
            cur_hyp_pts = ransac_voting.generate_hypothesis(direct, coords, idxs)  # [hn,vn,2]

            # voting for hypothesis
            cur_inlier = torch.zeros([round_hyp_num, vn, tn], dtype=torch.uint8, device=mask.device)
            ransac_voting.voting_for_hypothesis(direct, coords, cur_hyp_pts, cur_inlier, inlier_thresh)  # [hn,vn,tn]

            # find max
            cur_inlier_counts = torch.sum(cur_inlier, 2)                   # [hn,vn]
            cur_win_counts, cur_win_idx = torch.max(cur_inlier_counts, 0)  # [vn]
            cur_win_pts = cur_hyp_pts[cur_win_idx, torch.arange(vn)]
            cur_win_ratio = cur_win_counts.float() / tn

            # update best point
            larger_mask = all_win_ratio < cur_win_ratio
            all_win_pts[larger_mask, :] = cur_win_pts[larger_mask, :]
            all_win_ratio[larger_mask] = cur_win_ratio[larger_mask]

            # check confidence
            hyp_num += round_hyp_num
            cur_iter += 1
            cur_min_ratio = torch.min(all_win_ratio)
            if (1 - (1 - cur_min_ratio ** 2) ** hyp_num) > confidence or cur_iter > max_iter:
                break

        # compute mean intersection again
        normal = torch.zeros_like(direct)   # [tn,vn,2]
        normal[:, :, 0] = direct[:, :, 1]
        normal[:, :, 1] = -direct[:, :, 0]
        all_inlier = torch.zeros([1, vn, tn], dtype=torch.uint8, device=mask.device)
        all_win_pts = torch.unsqueeze(all_win_pts, 0)  # [1,vn,2]
        ransac_voting.voting_for_hypothesis(direct, coords, all_win_pts, all_inlier, inlier_thresh)  # [1,vn,tn]

        # coords [tn,2] normal [vn,tn,2]
        all_inlier=torch.squeeze(all_inlier.float(),0)              # [vn,tn]
        normal=normal.permute(1,0,2)                                # [vn,tn,2]
        normal=normal*torch.unsqueeze(all_inlier,2)                 # [vn,tn,2] outlier is all zero

        b=torch.sum(normal*torch.unsqueeze(coords,0),2)             # [vn,tn]
        ATA=torch.matmul(normal.permute(0,2,1),normal)              # [vn,2,2]
        ATb=torch.sum(normal*torch.unsqueeze(b,2),1)                # [vn,2]
        all_win_pts=torch.matmul(b_inv(ATA),torch.unsqueeze(ATb,2)) # [vn,2,1]

        all_inlier=torch.zeros([1, vn, tn], dtype=torch.uint8, device=mask.device)
        ransac_voting.voting_for_hypothesis(direct, coords, torch.unsqueeze(all_win_pts[:,:,0], 0), all_inlier, 0.999)
        pts_conf=torch.sum(all_inlier.int(),2).float()/tn # [1,vn]

        batch_win_pts.append(all_win_pts[None,:,:,0])
        batch_confidence.append(pts_conf)


    batch_win_pts=torch.cat(batch_win_pts)
    batch_confidence=torch.cat(batch_confidence)
    return batch_win_pts, batch_confidence

def ransac_voting_layer_v6(mask, vertex, round_hyp_num, inlier_thresh=0.999, confidence=0.99, max_iter=20,
                           min_num=5, max_num=100):
    '''
    :param mask:      [b,h,w]
    :param vertex:    [b,h,w,vn,2]
    :param round_hyp_num:
    :param inlier_thresh:
    :return: [b,vn,2] [b,vn,2,2]
    '''
    b, h, w, vn, _ = vertex.shape
    batch_win_pts, batch_confidence = [], []
    mask2=mask.byte()
    print(mask2.device)
    val=torch.sum(mask2)
    for bi in range(b):
        hyp_num = 0

        foreground_num = torch.sum(mask)
        cur_mask = mask.byte()[bi]

        # if too few points, just skip it
        if foreground_num < min_num:
            win_pts = torch.zeros([1, vn, 2], dtype=torch.float32, device=mask.device)
            pts_conf = torch.zeros([1, vn], dtype=torch.float32, device=mask.device)
            batch_win_pts.append(win_pts)      # [1,vn,2]
            batch_confidence.append(pts_conf)  # [1, vn]
            continue

        # if too many inliers, we randomly down sample it
        if foreground_num > max_num:
            selection = torch.zeros(cur_mask.shape, dtype=torch.float32, device=mask.device).uniform_(0, 1)
            selected_mask = (selection < (max_num / foreground_num.float()))
            cur_mask *= selected_mask

        coords = torch.nonzero(cur_mask).float()  # [tn,2]
        coords = coords[:, [1, 0]]
        direct = vertex[bi].masked_select(torch.unsqueeze(torch.unsqueeze(cur_mask, 2), 3))  # [tn,vn,2]
        direct = direct.view([coords.shape[0], vn, 2])
        tn = coords.shape[0]
        idxs = torch.zeros([round_hyp_num, vn, 2], dtype=torch.int32, device=mask.device).random_(0, direct.shape[0])
        all_win_ratio = torch.zeros([vn], dtype=torch.float32, device=mask.device)
        all_win_pts = torch.zeros([vn, 2], dtype=torch.float32, device=mask.device)

        cur_iter = 0
        while True:
            # generate hypothesis
            cur_hyp_pts = ransac_voting.generate_hypothesis(direct, coords, idxs)  # [hn,vn,2]

            # voting for hypothesis
            cur_inlier = torch.zeros([round_hyp_num, vn, tn], dtype=torch.uint8, device=mask.device)
            ransac_voting.voting_for_hypothesis(direct, coords, cur_hyp_pts, cur_inlier, inlier_thresh)  # [hn,vn,tn]

            # find max
            cur_inlier_counts = torch.sum(cur_inlier, 2)                   # [hn,vn]
            cur_win_counts, cur_win_idx = torch.max(cur_inlier_counts, 0)  # [vn]
            cur_win_pts = cur_hyp_pts[cur_win_idx, torch.arange(vn)]
            cur_win_ratio = cur_win_counts.float() / tn

            # update best point
            larger_mask = all_win_ratio < cur_win_ratio
            all_win_pts[larger_mask, :] = cur_win_pts[larger_mask, :]
            all_win_ratio[larger_mask] = cur_win_ratio[larger_mask]

            # check confidence
            hyp_num += round_hyp_num
            cur_iter += 1
            cur_min_ratio = torch.min(all_win_ratio)
            if (1 - (1 - cur_min_ratio ** 2) ** hyp_num) > confidence or cur_iter > max_iter:
                break

        # compute mean intersection again
        normal = torch.zeros_like(direct)   # [tn,vn,2]
        normal[:, :, 0] = direct[:, :, 1]
        normal[:, :, 1] = -direct[:, :, 0]
        all_inlier = torch.zeros([1, vn, tn], dtype=torch.uint8, device=mask.device)
        all_win_pts = torch.unsqueeze(all_win_pts, 0)  # [1,vn,2]
        ransac_voting.voting_for_hypothesis(direct, coords, all_win_pts, all_inlier, inlier_thresh)  # [1,vn,tn]

        # coords [tn,2] normal [vn,tn,2]
        all_inlier=torch.squeeze(all_inlier.float(),0)              # [vn,tn]
        normal=normal.permute(1,0,2)                                # [vn,tn,2]
        normal=normal*torch.unsqueeze(all_inlier,2)                 # [vn,tn,2] outlier is all zero

        b=torch.sum(normal*torch.unsqueeze(coords,0),2)             # [vn,tn]
        ATA=torch.matmul(normal.permute(0,2,1),normal)              # [vn,2,2]
        ATb=torch.sum(normal*torch.unsqueeze(b,2),1)                # [vn,2]
        all_win_pts=torch.matmul(b_inv(ATA),torch.unsqueeze(ATb,2)) # [vn,2,1]

        all_inlier=torch.zeros([1, vn, tn], dtype=torch.uint8, device=mask.device)
        ransac_voting.voting_for_hypothesis(direct, coords, torch.unsqueeze(all_win_pts[:,:,0], 0), all_inlier, 0.999)
        pts_conf=torch.sum(all_inlier.int(),2).float()/tn # [1,vn]

        batch_win_pts.append(all_win_pts[None,:,:,0])
        batch_confidence.append(pts_conf)


    batch_win_pts=torch.cat(batch_win_pts)
    batch_confidence=torch.cat(batch_confidence)
    return batch_win_pts, batch_confidence

def ransac_motion_voting(mask, vertex):
    '''

    :param mask:   b,h,w
    :param vertex: b,h,w,vn,2
    :return:
    '''
    b, h, w, vn, _ = vertex.shape
    pts=[]
    for bi in range(b):
        cur_mask=mask[bi].byte()
        coords=torch.nonzero(cur_mask).float()
        if coords.shape[0]<1:
            pts.append(torch.zeros([1,vn,2],dtype=torch.float32,device=vertex.device))
            continue
        coords=coords[:,(1,0)]
        cur_vert=vertex[bi]
        cur_vert=cur_vert[cur_mask]+torch.unsqueeze(coords,1)
        pt=torch.mean(cur_vert,0)
        pts.append(torch.unsqueeze(pt,0))

    return torch.cat(pts,0)


if __name__=="__main__":
    from lib.datasets.linemod_dataset import LineModDatasetRealAug,VotingType
    from lib.utils.data_utils import LineModImageDB
    from lib.utils.draw_utils import imagenet_to_uint8
    import numpy as np
    train_set = LineModDatasetRealAug(LineModImageDB('cat',has_fuse_set=False,has_ms_set=False).real_set)
    rgb, mask, vertex, vertex_weight, pose, gt_corners = train_set[np.random.randint(0,len(train_set)),480,640]

    h,w=mask.shape
    mask_0=torch.unsqueeze(mask, 0)
    # mask=torch.cat([mask_0,mask_0],0).cuda().int().contiguous()
    vertex=vertex.cuda()    # [16,h,w]
    vertex=vertex.permute(1,2,0).view(h,w,8,2)
    vertex_0=torch.unsqueeze(vertex, 0)
    # vertex=torch.cat([vertex_0,vertex_0],0).cuda().float().contiguous()
    vt_corners=ransac_voting_layer_v3(mask_0.cuda(),vertex_0.cuda(),500) # [1,1,8,2]
    print(vt_corners.shape)

    vt_corners=vt_corners.cpu().numpy()[0]
    gt_corners=gt_corners.numpy()[:,:2]

    print(vt_corners)
    print(gt_corners)
    print(vt_corners-gt_corners)

    import matplotlib.pyplot as plt
    plt.imshow(imagenet_to_uint8(rgb.cpu().numpy()))
    plt.plot(vt_corners[:,0],vt_corners[:,1],'*')
    plt.plot(gt_corners[:,0],gt_corners[:,1],'*')
    plt.show()

    # vote for vanishing point
    # train_set = LineModDatasetRealAug(LineModImageDB('cat',has_fuse_set=False,has_ms_set=False).real_set,vote_type=VotingType.VanPts)
    #
    # for k in np.random.choice(np.arange(len(train_set)),100):
    #     rgb, mask, vertex, vertex_weight, pose, van_pts = train_set[np.random.randint(k, len(train_set)), 480, 640]
    #
    #     h,w=mask.shape
    #     mask_0=torch.unsqueeze(mask, 0)
    #     vn,h,w=vertex.shape
    #     vn//=2
    #     vertex=vertex.cuda()    # [16,h,w]
    #     vertex=vertex.permute(1,2,0).view(h,w,vn,2)
    #     vertex_0=torch.unsqueeze(vertex, 0)
    #     # vertex=torch.cat([vertex_0,vertex_0],0).cuda().float().contiguous()
    #     vt_van_pts=ransac_voting_vanish_point_layer(mask_0.cuda(), vertex_0.cuda(), 2, 500, 0.999, 0.99, 20, 5, 30000, 0) # [1,1,8,2]
    #
    #     vt_van_pts=vt_van_pts.cpu().numpy()[0, 0]
    #     van_pts= van_pts.numpy()
    #
    #     ratio=vt_van_pts/van_pts
    #     try:
    #         assert(np.sum(ratio<0)==0)
    #         assert(np.sum(np.abs(ratio[:,0]-ratio[:,1])>1e-5)==0)
    #         assert(np.sum(np.abs(ratio[:,1]-ratio[:,2])>1e-5)==0)
    #     except AssertionError:
    #         print(ratio)
    #
    #     print(k)
    # assert(np.sum(np.abs(ratio[:,2]-ratio[:,3])>1e-6)==0)

    # import matplotlib.pyplot as plt
    # plt.imshow(imagenet_to_uint8(rgb.cpu().numpy()))
    # plt.plot(vt_van_pts[:, 0], vt_van_pts[:, 1], '*')
    # plt.plot(van_pts[:, 0], van_pts[:, 1], '*')
    # plt.show()




