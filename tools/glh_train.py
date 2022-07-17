from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from train_utils.optimization import build_optimizer, build_scheduler
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pathlib import Path
import datetime
from pcdet.utils import common_utils
from pcdet.models import build_network
import os
def main():
    # 加载模型配置文件
    cfg_file = "cfgs/kitti_models/IA-SSD.yaml"
    cfg_from_yaml_file(cfg_file, cfg)
    cfg.TAG = Path(cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'
    # 日志文件设置
    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (cfg_file, output_dir))
    # 超参数设置
    epochs = cfg.OPTIMIZATION.NUM_EPOCHS
    batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    class_names = cfg.CLASS_NAMES
    dataset_cfg = cfg.DATA_CONFIG
    workers = 4

    # dataloader  network optimizer
    train_set, train_loader, train_sample = build_dataloader(
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        batch_size=batch_size,
        dist=False,
        workers=workers,
        logger=logger,
        training=True
    )
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)



if __name__ == "__main__":
    main()





