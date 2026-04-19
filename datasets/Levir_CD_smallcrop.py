import logging
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F

# 数据集根目录
# dataset_root = r'G:\Dataset\LEVIR-CD 2'
# dataset_root = r'D:\LEVIR-CD 2'
# dataset_root = '/root/autodl-tmp/LEVIR-CD'
dataset_root = '/home/xulifa/dataset/LEVIR_CD'


def get_dataloader(batch_size, crop_size, mode, num_workers, shuffle, seed=42):
    """获取数据加载器

    参数:
        root (str): 数据集根目录
        batch_size (int): 批次大小
        mode (str): 'train', 'val' 或 'test'
        shuffle (bool): 是否打乱数据

    返回:
        torch.utils.data.DataLoader: 数据加载器
    """
    logger = logging.getLogger()

    # 创建dataset
    dataset = LevirCD(root=dataset_root, mode=mode, crop_size=crop_size)

    # 创建dataloader
    g = torch.Generator()
    g.manual_seed(seed)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )

    logger.info(
        f"{mode}_batch_size: {batch_size}, "
        f"crop_size: {crop_size}, "
        f"总批次: {len(dataloader)}, "
        f"总样本数: {len(dataset)}, "
        f"数据集根目录: {dataset_root}")

    return dataloader


class LevirCD(Dataset):
    """LEVIR-CD数据集加载器"""

    def __init__(self, root, crop_size=256, mode='train'):
        """
        参数:
            root (str): 数据集根目录
            mode (str): 'train', 'val' 或 'test'
            crop_size (int): 裁剪大小
            transform (callable, optional): 额外的数据增强
        """
        self.logger = logging.getLogger()
        self.crop_size = crop_size
        self.mode = mode

        # 数据路径
        self.img_A_dir = os.path.join(root, mode, 'A')
        self.img_B_dir = os.path.join(root, mode, 'B')
        self.label_dir = os.path.join(root, mode, 'label')

        # 检查目录是否存在
        for dir_path in [self.img_A_dir, self.img_B_dir, self.label_dir]:
            if not os.path.exists(dir_path):
                self.logger.error(f"目录不存在: {dir_path}")
                raise FileNotFoundError(f"目录不存在: {dir_path}")

        # 获取文件列表
        img_A_files = set(os.listdir(self.img_A_dir))
        img_B_files = set(os.listdir(self.img_B_dir))
        label_files = set(os.listdir(self.label_dir))
        # 确保文件名一致
        common_files = img_A_files.intersection(img_B_files, label_files)
        if (len(common_files) != len(img_A_files) or
                len(common_files) != len(img_B_files) or
                len(common_files) != len(label_files)):
            e = f"某些文件在不同目录中不一致"
            self.logger.error(e)
            raise FileNotFoundError(e)

        self.files = sorted(common_files)
        
        # 计算不重叠裁剪的参数
        if mode != 'train':
            self.stride = crop_size
            self.num_crops_per_side = 1024 // crop_size
            self.num_crops = self.num_crops_per_side * self.num_crops_per_side

    def __getitem__(self, idx):
        # 读取图像
        if self.mode == 'train':
            img_name = self.files[idx]
            img_A_path = os.path.join(self.img_A_dir, img_name)
            img_B_path = os.path.join(self.img_B_dir, img_name)
            label_path = os.path.join(self.label_dir, img_name)

            img_A = Image.open(img_A_path)
            img_B = Image.open(img_B_path)
            label = Image.open(label_path).convert('L')

            # 应用变换
            img_A, img_B, label = train_transforms(img_A, img_B, label, self.crop_size)
        else:
            # 计算图像索引和裁剪索引
            img_idx = idx // self.num_crops
            crop_idx = idx % self.num_crops
            
            img_name = self.files[img_idx]
            img_A_path = os.path.join(self.img_A_dir, img_name)
            img_B_path = os.path.join(self.img_B_dir, img_name)
            label_path = os.path.join(self.label_dir, img_name)

            img_A = Image.open(img_A_path)
            img_B = Image.open(img_B_path)
            label = Image.open(label_path).convert('L')
            
            # 计算裁剪位置
            row = crop_idx // self.num_crops_per_side
            col = crop_idx % self.num_crops_per_side
            left = col * self.stride
            upper = row * self.stride
            right = left + self.crop_size
            lower = upper + self.crop_size
            
            # 裁剪图像
            img_A = img_A.crop((left, upper, right, lower))
            img_B = img_B.crop((left, upper, right, lower))
            label = label.crop((left, upper, right, lower))
            
        # 转换为Tensor
        img_A = F.to_tensor(img_A)
        img_B = F.to_tensor(img_B)
        label = F.to_tensor(label)

        return img_A, img_B, label

    def __len__(self):
        if self.mode == 'train':
            return len(self.files)
        else:
            return len(self.files) * self.num_crops


def train_transforms(img_A, img_B, label, crop_size):
    """训练集的数据增强操作"""
    # 随机旋转
    # angle = torch.randint(-180, 180, (1,)).item()
    # img_A = img_A.rotate(angle)
    # img_B = img_B.rotate(angle)
    # label = label.rotate(angle)

    # 随机裁剪
    i, j, h, w = transforms.RandomCrop.get_params(img_A, output_size=(crop_size, crop_size))
    img_A = F.crop(img_A, i, j, h, w)
    img_B = F.crop(img_B, i, j, h, w)
    label = F.crop(label, i, j, h, w)

    # 随机翻转
    if torch.rand(1) > 0.5:
        img_A = F.hflip(img_A)
        img_B = F.hflip(img_B)
        label = F.hflip(label)
    if torch.rand(1) > 0.5:
        img_A = F.vflip(img_A)
        img_B = F.vflip(img_B)
        label = F.vflip(label)

    # 颜色变换
    color_transform = transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.2, hue=0.1)
    img_A = color_transform(img_A)
    img_B = color_transform(img_B)
    return img_A, img_B, label


def val_test_transforms(img_A, img_B, label, crop_size):
    """验证集和测试集的数据变换操作"""
    # 这个函数不再使用，因为我们在__getitem__中直接处理了
    pass


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)    