import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true', dest='debug', default=False)
parser.add_argument('--rec', action='store_true', dest='rec', default=False)
parser.add_argument('--rec_name', default='resnet', type=str)
parser.add_argument('--net', action='store', dest='net', type=str)
parser.add_argument('--type', action='store', dest='type', type=str)

parser.add_argument('--cfg_file', type=str, default='configs/linemod_train.json')
parser.add_argument('--linemod_cls', type=str, default='cat')

parser.add_argument('--use_gt_mask', action='store_true', dest='use_gt_mask', default=False)
parser.add_argument('--test_model', action='store_true', dest='test_model', default=False)
parser.add_argument('--test_sequence', action='store_true', dest='test_sequence', default=False)
parser.add_argument('--use_test_set', action='store_true', dest='use_test_set', default=False)
parser.add_argument('--use_uncertainty_pnp', action='store_true', dest='use_uncertainty_pnp', default=False)

parser.add_argument('--visualize_bbox', action='store_true', dest='visualize_bbox', default=False)
parser.add_argument('--vote_refine_num', type=int, default=1)
parser.add_argument('--vote_inlier_thresh', type=int, default=0.99)

# following args used in visualize vertex
parser.add_argument('--normal', dest='normal', action='store_true')
parser.add_argument('--no-normal', dest='normal', action='store_false')
parser.add_argument('--occluded', dest='occluded', action='store_true')
parser.add_argument('--no-occluded', dest='occluded', action='store_false')
parser.add_argument('--truncated', dest='truncated', action='store_true')
parser.add_argument('--no-truncated', dest='truncated', action='store_false')
parser.add_argument('--printer', dest='printer', action='store_true')
parser.add_argument('--no-printer', dest='printer', action='store_false')
parser.set_defaults(occluded=True)
parser.set_defaults(normal=True)
parser.set_defaults(truncated=False)
parser.set_defaults(printer=False)

parser.add_argument('--load_epoch', type=int, default=-1)
parser.add_argument('--visual_num', type=int, default=32)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--visual_occluded', dest='visual_occluded', action='store_true', default=False)
parser.add_argument('--visual_special', dest='visual_special', action='store_true', default=False)
parser.add_argument('--visual_mask_out', dest='visual_mask_out', action='store_true', default=False)

parser.add_argument('--ycb_visual_set',type=str,default='val')
parser.add_argument('--ycb_visual_idx',type=int,default=1)
parser.add_argument('--record_detail_pr',type=bool,default=False)
parser.add_argument('--save_inter_result',type=bool,default=False)
parser.add_argument('--save_inter_dir',type=str,default='data/post_experiment')

parser.add_argument('--visualize_hyp_pts',dest='visualize_hyp_pts',action='store_true',)
parser.add_argument('--visualize_mask',dest='visualize_mask',action='store_true')
parser.add_argument('--visualize_points',dest='visualize_points',action='store_true')
parser.add_argument('--no-visualize_hyp_pts',dest='visualize_hyp_pts',action='store_false')
parser.add_argument('--no-visualize_mask',dest='visualize_mask',action='store_false')
parser.add_argument('--no-visualize_points',dest='visualize_points',action='store_false')
parser.set_defaults(visualize_hyp_pts=True)
parser.set_defaults(visualize_mask=True)
parser.set_defaults(visualize_points=True)

parser.add_argument('--ycb_single_idx',type=int,default=1)
parser.add_argument('--sequence_id', type=int, default=0)

parser.add_argument('--resize', dest='resize', action='store_true', default=False)

args = parser.parse_args()
