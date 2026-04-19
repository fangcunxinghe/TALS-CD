import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image  # 用于保存图片
from tqdm import tqdm
from models.TALS_CD import TALS_CD as Net
# from models.mymodel.premodel.FAFANet_Large_small_MOE_mamba_encoder_direct_1 import FAFANet as Net
# from models.v5_ab.CNN_Mamba_SSM_learnScan_S_multiScial_vim import FAFANet as Net
# from models.v5_ab.CNN_Mamba_SSM_learnScan_S_multiScial import FAFANet as Net
# from datasets.eval_scan.CLCD import get_dataloader
# from datasets.eval_scan.Levir_CD import get_dataloader
from datasets.eval_scan.WHU_CD import get_dataloader
# from datasets.eval_scan.Levir_CD_smallcrop import get_dataloader
from utils.evaluator import AverageMeter, ConfusionMatrix

WORKING_PATH = Path.cwd()
SAVE_DIR = WORKING_PATH / "eval_results"  # 或指定绝对路径
BIN_DIR = SAVE_DIR / "pre"
RGB_DIR = SAVE_DIR / "rgb"
# 创建目录（若不存在则自动创建）
# BIN_DIR.mkdir(parents=True, exist_ok=True)
# RGB_DIR.mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def main():

    # 初始化模型
    model = Net()
    # 检查点路径

    load_path = r'checkpoints/WHU/TALS_CD/BestModel_epoch64_F1_0.9484.pth'
    
    # 设置设备
    device = torch.device(f"cuda:0")

    # 获取dataloader_WHU
    test_loader = get_dataloader(
        batch_size=16,
        crop_size=256,
        mode='test',
        num_workers=4,
        shuffle=False
    )
    # 获取dataloader_LEVIR
    # test_loader = get_dataloader(
    #     batch_size=4,
    #     crop_size=1024,
    #     mode='test',
    #     num_workers=4,
    #     shuffle=False
    # )
    # 获取dataloader_CLCD
    # test_loader = get_dataloader(
    #     batch_size=4,
    #     crop_size=512,
    #     mode='test',
    #     num_workers=4,
    #     shuffle=False
    # )

    # 加载检查点
    try:
        print(f"加载检查点：{load_path}")
        state_dict = torch.load(load_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        print(f"加载检查点失败：{e}")
        raise

    model = model.to(device)

    # 评估模式
    model.eval()

    # 初始化混淆矩阵评估器
    evaluator = ConfusionMatrix()
    # 初始化损失统计器
    loss_meter = AverageMeter()

    with tqdm(total=len(test_loader), desc="测试进度") as pbar:
        for i, (imgs_A, imgs_B, labels,filename) in enumerate(test_loader):
            imgs_A = imgs_A.to(device)
            imgs_B = imgs_B.to(device)
            labels = labels.to(device)

            outputs = model(imgs_A, imgs_B)
            loss = F.binary_cross_entropy_with_logits(outputs, labels)

            # 转换为 numpy 数组
            outputs = torch.sigmoid(outputs).cpu().numpy()  # 求sigmoid
            labels = labels.cpu().numpy()

            # 更新混淆矩阵评估器
            evaluator.update(outputs, labels)
            # 更新损失统计器
            loss_meter.update(loss.item(), labels.shape[0])

            # # 保存预测图片
            # for j in range(outputs.shape[0]):
            #     pred_img = outputs[j][0]   # [H, W]
            #     label_img = labels[j][0]   # [H, W]

            #     # 二值化预测
            #     pred_bin = (pred_img > 0.5).astype(np.uint8)
            #     label_bin = (label_img > 0.5).astype(np.uint8)

            #     # 创建彩色对比图（H, W, 3）
            #     h, w = pred_bin.shape
            #     color_map = np.zeros((h, w, 3), dtype=np.uint8)

            #     # 条件掩码
            #     tp = (pred_bin == 1) & (label_bin == 1)  # True Positive
            #     tn = (pred_bin == 0) & (label_bin == 0)  # True Negative
            #     fp = (pred_bin == 1) & (label_bin == 0)  # False Positive
            #     fn = (pred_bin == 0) & (label_bin == 1)  # False Negative

            #     # 赋颜色（RGB）
            #     color_map[tp] = [255, 255, 255]  # 白色
            #     color_map[tn] = [0, 0, 0]        # 黑色
            #     color_map[fp] = [255, 0, 0]      # 红色
            #     color_map[fn] = [0, 255, 0]      # 绿色

            #     # 转为 PIL 图片并保存
            #     result_img = Image.fromarray(color_map)
            #     result_img.save(RGB_DIR / f"{filename[j]}.png")
                
            #     # === 保存二值化预测图（黑白） ===
            #     bin_img = (pred_bin * 255).astype(np.uint8)
            #     Image.fromarray(bin_img).save(BIN_DIR / f"{filename[j]}.png")

            # 刷新进度条
            pbar.update(1)

    # 计算评估结果
    accuracy, precision, recall, f1_score, iou = evaluator.get_evaluate()
    print(
        f"测试集评估 "
        f"Loss: {loss_meter.avg:.4f} "
        f"Acc: {accuracy:.2%} "
        f"Pre: {precision:.2%} "
        f"Rec: {recall:.2%} "
        f"F1: {f1_score:.2%} "
        f"IoU: {iou:.2%}"
    )
    print(f"预测保存路径：{SAVE_DIR.resolve()}")  # 打印绝对路径


if __name__ == '__main__':
    main()
