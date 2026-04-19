import argparse
import importlib
import logging
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import utils.trainer_epoch as trainer_epoch
from datasets.Levir_CD_smallcrop import get_dataloader as get_levir_dataloader
from datasets.WHU_CD import get_dataloader as get_whu_dataloader
from datasets.CLCD import get_dataloader as get_clcd_dataloader
from utils.logger import init_logger


def main():
    try:
        # 解析命令行参数
        parser = argparse.ArgumentParser()
        parser.add_argument('-m', '--model-module', type=str, help="模型所在python模块路径，如 models.TALS_CD", required=True)
        parser.add_argument('-c', '--model-class', type=str,default='TALS_CD', help="模型类名，如 TALS_CD", required=False)
        parser.add_argument('-d', '--dataset', type=str, choices=['LEVIR', 'LEVIR_CD', 'WHU', 'WHU_CD', 'CLCD'], help="数据集名称", required=True)
        parser.add_argument('-tb', '--train-batch-size', type=int, default=16, help="训练batch size")
        parser.add_argument('-vb', '--val-batch-size', type=int, default=4, help="验证/测试batch size")
        parser.add_argument('-tc', '--train-crop-size', type=int, default=512, help="训练裁剪大小")
        parser.add_argument('-vc', '--val-crop-size', type=int, default=512, help="验证/测试裁剪大小")
        parser.add_argument('-e', '--total-epoch', type=int, default=300, help="训练总epoch数")
        parser.add_argument('--optimizer', type=str, default='AdamW', choices=['AdamW', 'SGD'], help="优化器类型")
        parser.add_argument('--lr', type=float, default=0.001, help="学习率")
        parser.add_argument('--weight-decay', type=float, default=0.05, help="权重衰减")
        parser.add_argument('--momentum', type=float, default=0.9, help="SGD动量")
        parser.add_argument('--beta1', type=float, default=0.9, help="AdamW beta1")
        parser.add_argument('--beta2', type=float, default=0.999, help="AdamW beta2")
        parser.add_argument('--load-path', type=str, default=None, help="可选，预训练权重路径")
        parser.add_argument('-r', '--remark', type=str, help="日志备注", required=False)
        parser.add_argument('-g', '--gpu', type=int, default=0, help="GPU编号")
        parser.add_argument('--no-gpu', action='store_true', help="禁用GPU，使用CPU")
        parser.add_argument('--multi-gpu', type=str, default=None, help="多GPU逗号分隔，例如 0,1")
        parser.add_argument('--seed', type=int, default=None, help="随机种子，默认随机生成")
        input_args = parser.parse_args()

        # 使用确定性算法并设置随机种子
        # torch.use_deterministic_algorithms(True)
        seed = input_args.seed if input_args.seed is not None else random.randint(0, 2 ** 32 - 1)
        set_seed(seed)
        
        # 根据数据集选择对应的dataloader函数
        dataset_key = input_args.dataset.upper()
        if dataset_key in ['LEVIR', 'LEVIR_CD']:
            get_dataloader = get_levir_dataloader
            input_args.train_batch_size=16
            input_args.val_batch_size=8
            input_args.train_crop_size=512
            input_args.val_crop_size=1024
            input_args.total_epoch=1000
        elif dataset_key in ['WHU', 'WHU_CD']:
            get_dataloader = get_whu_dataloader
            input_args.train_batch_size=16
            input_args.val_batch_size=16
            input_args.train_crop_size=256
            input_args.val_crop_size=256
            input_args.total_epoch=100
        elif dataset_key == 'CLCD':
            get_dataloader = get_clcd_dataloader
            input_args.train_batch_size=16
            input_args.val_batch_size=16
            input_args.train_crop_size=512
            input_args.val_crop_size=512
            input_args.total_epoch=500
        else:
            raise ValueError(f"Unsupported dataset: {input_args.dataset}")

        # 组装训练配置（无需外部配置文件）
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_remark = input_args.remark or input_args.model_module.split('.')[-1]
        working_path = Path.cwd()
        args = {
            'model': load_class(input_args.model_module, input_args.model_class),
            'train_batch_size': input_args.train_batch_size,
            'val_batch_size': input_args.val_batch_size,
            'train_crop_size': input_args.train_crop_size,
            'val_crop_size': input_args.val_crop_size,
            'load_path': input_args.load_path,
            'total_epoch': input_args.total_epoch,
            'optimizer': {
                'type': input_args.optimizer,
                'lr': input_args.lr,
                'betas': (input_args.beta1, input_args.beta2),
                'weight_decay': input_args.weight_decay,
                'momentum': input_args.momentum
            },
            'use_amp': False,
            'gpu': not input_args.no_gpu,
            'gpu_id': input_args.gpu,
            'multi_gpu': [int(x) for x in input_args.multi_gpu.split(',')] if input_args.multi_gpu else None,
            'pred_dir': working_path / 'results' / dataset_key / input_args.model_class,
            'chkpt_dir': working_path / 'checkpoints' / dataset_key / input_args.model_class,
            'log_dir': working_path / 'logs' / dataset_key / input_args.model_class / f"{timestamp}_{log_remark}",
        }

        # 初始化模型
        model = args['model']()

        # 初始化日志
        init_logger(args['log_dir'])

        # 获取dataloader
        train_loader = get_dataloader(
            batch_size=args['train_batch_size'],
            crop_size=args['train_crop_size'],
            mode='train',
            num_workers=8,
            shuffle=True,
            seed=seed
        )
        val_loader = get_dataloader(
            batch_size=args['val_batch_size'],
            crop_size=args['val_crop_size'],
            mode='val',
            num_workers=8,
            shuffle=False,
            seed=seed
        )
        test_loader = get_dataloader(
            batch_size=args['val_batch_size'],
            crop_size=args['val_crop_size'],
            mode='test',
            num_workers=4,
            shuffle=False,
            seed=seed
        )

        # 初始化训练器并开始训练
        trainer = trainer_epoch.Trainer(args, model, train_loader, val_loader, test_loader, seed)
        trainer.train()

    except Exception as e:
        logging.error(f"Error in main: {e}")
        raise


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_class(module_path: str, class_name: str):
    """按模块路径与类名加载类。"""
    module = importlib.import_module(module_path)
    if not hasattr(module, class_name):
        raise ImportError(f"{class_name} not found in module {module_path}")
    return getattr(module, class_name)


if __name__ == '__main__':
    main()
