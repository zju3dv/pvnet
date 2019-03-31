from lib.utils.config import cfg
from lib.utils.arg_utils import args
import torch


def run_gen_mask():
    # OcclusionLindModDB: we need masks for each object
    from lib.utils.data_utils import OcclusionLineModDB, OcclusionLineModDBSyn
    db = OcclusionLineModDB()
    db.get_masks()


if __name__ == '__main__':
    globals()['run_' + args.type]()

