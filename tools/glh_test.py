import _init_path
import datetime
import glob
import os
import re
import time
from pathlib import Path
import torch
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils
import warnings
from pcdet.models import load_data_to_gpu

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)


def run_model_single(model, test_loader, eval_output_dir, logger, epoch_id):
    # 加载训练完成的模型
    """
    ckpt = "IA-SSD.pth"
    model.load_params_from_file(filename=ckpt, logger=logger, to_cpu=False)
    model.cuda()
    # ['frame_id', 'calib', 'gt_boxes', 'points', 'use_lead_xyz', 'image_shape', 'batch_size']
    model.eval()
    """
    for i, batch_dict in enumerate(test_loader):
        load_data_to_gpu(batch_dict)
        """
        pred_dicts, ret_dict = model(batch_dict)
        if i == 2:
            break
        """
def main():
    # 加载模型配置文件
    cfg_file = "cfgs/kitti_models/IA-SSD.yaml"
    cfg_from_yaml_file(cfg_file, cfg)
    cfg.TAG = Path(cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'
    # 加载训练好的模型
    logger = common_utils.create_logger("log_test.txt", rank=cfg.LOCAL_RANK)
    workers = 4
    batch_size = 4
    class_names = cfg.CLASS_NAMES
    dataset_cfg = cfg.DATA_CONFIG

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        batch_size=batch_size,
        dist=False,
        workers=workers,
        logger=logger,
        training=False
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_output_dir = output_dir / 'eval'

    # 进行一次模型推理
    with torch.no_grad():
        run_model_single(model, test_loader, eval_output_dir, logger, epoch_id=1)


if __name__ == '__main__':
    main()
