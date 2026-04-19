import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from tensorboardX import SummaryWriter
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, PolynomialLR

from utils.evaluator import ConfusionMatrix, AverageMeter
# torch.set_float32_matmul_precision('high')

class Trainer:

    def __init__(self, args, model, train_loader, val_loader, test_loader, seed):
        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.seed = seed
        # 记录日志
        self.logger = logging.getLogger()
        self.logger.info(f"种子设置为: {self.seed}")
        
        # 设置训练设备
        if self.args['gpu']:
            self.device = torch.device(f"cuda:{self.args['gpu_id']}")
        else:
            self.device = torch.device("cpu")
        self.logger.info(f"使用设备: {self.device}")

        # 创建目录:检查点、日志、预测
        for dir_name in ['pred_dir', 'chkpt_dir', 'log_dir']:
            Path(self.args[dir_name]).mkdir(parents=True, exist_ok=True)

        # 初始化TensorBoard
        self.writer = SummaryWriter(self.args['log_dir'])
        # 保存相关配置文件
        with open(Path(self.args['log_dir']) / 'config.yaml', 'w') as f:
            yaml.dump(self.args, f)

        # 加载检查点
        if self.args['load_path']:
            try:
                self.logger.info(f"加载检查点:{self.args['load_path']}")
                state_dict = torch.load(self.args['load_path'], map_location=self.device, weights_only=True)
                self.model.load_state_dict(state_dict, strict=False)
            except Exception as e:
                self.logger.error(f"加载检查点失败:{e}")
                raise
        else:
            self.logger.info("未加载检查点，模型参数初始化（先去除）")
            # TODO: 会覆盖已有的权重
            # for module in self.model.modules():
            #     if isinstance(module, (nn.Conv2d, nn.Linear)):
            #         nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            #         if module.bias is not None:
            #             nn.init.constant_(module.bias, 0)

        # 多GPU训练
        if self.args['gpu'] and self.args['multi_gpu']:
            self.model = nn.DataParallel(self.model, device_ids=self.args['multi_gpu'])
            self.logger.info(f"使用多GPU:{self.args['multi_gpu']}")
        self.model = self.model.to(self.device)
        
        # 编译模型（PyTorch 2.0+）
        # if torch.__version__ >= "2.0":
        #     self.model = torch.compile(self.model)
        #     self.logger.info("使用torch.compile编译模型")
        # else:
        #     self.logger.info("PyTorch版本不支持torch.compile，跳过编译")

        # 创建优化器
        optimizer_config = self.args['optimizer']
        if optimizer_config['type'] == 'AdamW':
            self.optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=optimizer_config['lr'],
                betas=optimizer_config['betas'],
                weight_decay=optimizer_config['weight_decay']
            )
        elif optimizer_config['type'] == 'SGD':
            self.optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=optimizer_config['lr'],
                weight_decay=optimizer_config['weight_decay'],
                momentum=optimizer_config['momentum'],
                nesterov=True
            )
        else:
            e = f"未知优化器类型: {optimizer_config['type']}"
            self.logger.error(e)
            raise ValueError(e)
        self.logger.info(f"使用{optimizer_config['type']}优化器")

        # 初始化训练混合矩阵评估器
        self.train_evaluator = ConfusionMatrix()
        # 初始化验证混合矩阵评估器
        self.val_evaluator = ConfusionMatrix()

        # 初始化训练损失统计器
        self.train_loss_meter = AverageMeter()
        # 初始化验证损失统计器
        self.val_loss_meter = AverageMeter()

        # 训练epoch次数
        self.total_epoch = self.args['total_epoch']
        # 总迭代次数
        self.total_iter = self.total_epoch * len(self.train_loader)
        # self.logger.info(f"总训练epoch数: {self.total_epoch}, 每epoch迭代数: {len(self.train_loader)}, 总迭代数: {self.total_iter}")

        # 创建学习率调度器

        # SAM-CD
        # self.scheduler = PolynomialLR(self.optimizer, total_iters = self.total_epoch * len(self.train_loader), power=3.0)

        # BAN
        # 第一阶段:LinearLR（热身）
        warmup_iter = int(0.1 * self.total_epoch * len(self.train_loader))  # 热身iter
        warmup_scheduler = LinearLR(self.optimizer, start_factor=1e-6, total_iters=warmup_iter)
        # 第二阶段:CosineAnnealingLR or PolynomialLR
        cosine_scheduler = CosineAnnealingLR(self.optimizer, T_max=self.total_iter - warmup_iter, eta_min=1e-6)
        poly_scheduler = PolynomialLR(self.optimizer, total_iters=self.total_iter - warmup_iter, power=3.0)
        # 组合两个调度器
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_iter]
        )

    def train(self):
        """完整的训练流程"""
        best_f1 = 0.7  # 最佳f1（小于0.91不保存）
        best_epoch = 0  # 最佳epoch

        start_time = time.time()  # 训练开始时间
        last_notice_time = start_time  # 上次通知时间
        
        # 统计模型参数量
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        total_params = trainable_params + frozen_params
        self.logger.info(f"模型总参数量: {total_params:,},可训练: {trainable_params:,},冻结: {frozen_params:,}")


        try:
            for current_epoch in range(1, self.total_epoch+1):

                # 训练一个epoch
                self.train_epoch(current_epoch)
                if current_epoch % 1 == 0:
                    # 验证集评估
                    val_f1 = self.validate(current_epoch)
                    # 保存最佳模型
                    if val_f1 > best_f1:
                        best_f1 = val_f1  # 最佳F1
                        best_epoch = current_epoch  # 最佳epoch
                        torch.save(
                            self.model.state_dict() if not isinstance(self.model, nn.DataParallel)
                            else self.model.module.state_dict(),
                            Path(self.args['chkpt_dir']) / f"BestModel_epoch{current_epoch}_F1_{val_f1:.4f}.pth"
                        )

                # 打印训练时间进度
                current_time = time.time()  # 当前时间
                elapsed_time = current_time - start_time  # 已用时
                remaining_time = (self.total_epoch - current_epoch) * (elapsed_time / current_epoch)  # 剩余时间
                self.logger.info(
                    f"[{current_epoch}/{self.total_epoch}] 用时{current_time - last_notice_time:.2f}秒。"
                    f"总用时:{elapsed_time / 3600:.2f}小时。"
                    f"预计剩余时间:{remaining_time / 3600:.2f}小时"
                )
                last_notice_time = current_time  # 更新上次通知时间

        except Exception as e:
            self.logger.error(f"训练出错: {e}")
            raise
        finally:
            self.writer.close()
            self.logger.info(
                f"训练结束。最佳f1分数: {best_f1:.4f}，在第 {best_epoch} epoch。"
                f"总用时: {(time.time() - start_time) / 3600:.2f}小时"
            )

        # 训练完成后测试模型
        self.test(best_epoch, best_f1)

    def train_epoch(self, current_epoch):
        """训练一个epoch"""
        # 训练模式
        self.model.train()

        # 重置训练评估器
        self.train_evaluator.reset()
        # 重置训练损失统计器
        self.train_loss_meter.reset()

        for i, (imgs_A, imgs_B, labels) in enumerate(self.train_loader):
            imgs_A = imgs_A.to(self.device)
            imgs_B = imgs_B.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(imgs_A, imgs_B)
            loss = F.binary_cross_entropy_with_logits(outputs, labels)

            loss.backward()
            self.optimizer.step()

            # 学习率调度
            self.scheduler.step()

            # 更新指标
            with torch.no_grad():
                outputs = torch.sigmoid(outputs.detach()).cpu().numpy()
                labels = labels.cpu().detach().numpy()
                self.train_evaluator.update(outputs, labels)
                self.train_loss_meter.update(loss.item(), labels.shape[0])

        # 训练损失
        train_loss = self.train_loss_meter.avg
        # 计算评估结果
        accuracy, precision, recall, f1_score, iou = self.train_evaluator.get_evaluate()

        # 记录评价指标并打印日志
        self.logger_and_writer('train', current_epoch, train_loss, accuracy, precision, recall, f1_score, iou)


    @torch.no_grad()
    def validate(self, current_epoch):
        """验证模型"""
        # 评估模式
        self.model.eval()

        # 重置验证混淆矩阵评估器
        self.val_evaluator.reset()
        # 重置验证损失统计器
        self.val_loss_meter.reset()

        for i, (imgs_A, imgs_B, labels) in enumerate(self.val_loader):
            imgs_A = imgs_A.to(self.device)
            imgs_B = imgs_B.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(imgs_A, imgs_B)
            loss = F.binary_cross_entropy_with_logits(outputs, labels)

            # 转换为 numpy 数组
            outputs = torch.sigmoid(outputs).cpu().numpy()  # 求sigmoid
            labels = labels.cpu().numpy()

            # 更新验证混淆矩阵评估器
            self.val_evaluator.update(outputs, labels)
            # 更新验证损失统计器
            self.val_loss_meter.update(loss.item(), labels.shape[0])

        # 计算评估结果
        accuracy, precision, recall, f1_score, iou = self.val_evaluator.get_evaluate()

        # 记录评价指标并打印日志
        self.logger_and_writer('val', current_epoch, self.val_loss_meter.avg, accuracy, precision, recall, f1_score,
                               iou)

        return f1_score

    @torch.no_grad()
    def test(self, best_epoch, best_f1):
        """测试模型"""
        # 评估模式
        self.model.eval()

        # 初始化混淆矩阵评估器
        evaluator = ConfusionMatrix()
        # 初始化损失统计器
        loss_meter = AverageMeter()

        # 加载检查点
        file_name = f"BestModel_epoch{best_epoch}_F1_{best_f1:.4f}.pth"
        load_path = Path(self.args['chkpt_dir']) / file_name
        try:
            self.logger.info(f"加载检查点：{load_path}")
            state_dict = torch.load(load_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            self.logger.error(f"加载检查点失败：{e}")
            raise

        for i, (imgs_A, imgs_B, labels) in enumerate(self.test_loader):
            imgs_A = imgs_A.to(self.device)
            imgs_B = imgs_B.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(imgs_A, imgs_B)
            loss = F.binary_cross_entropy_with_logits(outputs, labels)

            # 转换为 numpy 数组
            outputs = torch.sigmoid(outputs).cpu().numpy()  # 求sigmoid
            labels = labels.cpu().numpy()

            # 更新混淆矩阵评估器
            evaluator.update(outputs, labels)
            # 更新损失统计器
            loss_meter.update(loss.item(), labels.shape[0])

        # 计算评估结果
        accuracy, precision, recall, f1_score, iou = evaluator.get_evaluate()
        self.logger.info(
            f"测试集评估 "
            f"Loss: {loss_meter.avg:.4f} "
            f"Acc: {accuracy:.2%} "
            f"Pre: {precision:.2%} "
            f"Rec: {recall:.2%} "
            f"F1: {f1_score:.2%} "
            f"IoU: {iou:.2%}"
        )

        # 修改模型文件名为最终模型
        new_name = file_name.replace("BestModel", "FinalModel")
        new_name = f"{f1_score:.4f}_" + new_name
        new_path = load_path.parent / new_name
        # 执行重命名
        load_path.rename(new_path)
        self.logger.info(f"最终模型文件为：{new_path}")

    def logger_and_writer(self, mode, current_epoch, loss, accuracy, precision, recall, f1_score, iou):
        """
        记录评价指标并打印日志
        :param mode: 模式（训练或验证）
        :param current_epoch: 当前epoch
        :param loss: 损失值
        :param accuracy: 准确率
        :param precision: 精确率
        :param recall: 召回率
        :param f1_score: F1分数
        :param iou: IoU
        """
        # 记录评价指标
        self.writer.add_scalar(f'loss/{mode}', loss, current_epoch)
        self.writer.add_scalar(f'Acc/{mode}', accuracy, current_epoch)
        self.writer.add_scalar(f'Pre/{mode}', precision, current_epoch)
        self.writer.add_scalar(f'Rec/{mode}', recall, current_epoch)
        self.writer.add_scalar(f'F1/{mode}', f1_score, current_epoch)
        self.writer.add_scalar(f'IoU/{mode}', iou, current_epoch)
        if mode == 'train':
            self.writer.add_scalar(f'lr', self.optimizer.param_groups[0]['lr'], current_epoch)

        # 打印日志
        if mode == 'train':
            self.logger.info(
                f"[{current_epoch}/{self.total_epoch}] {mode}: "
                f"Loss/{mode}: {loss:.4f}, "
                f"Acc/{mode}: {accuracy:.2%}, "
                f"lr: {self.optimizer.param_groups[0]['lr']:.6f}"
            )
        else:
            self.logger.info(
                f"[{current_epoch}/{self.total_epoch}] {mode}: "
                f"Loss/{mode}: {loss:.4f}, "
                f"Acc/{mode}: {accuracy:.2%}, "
                f"F1/{mode}: {f1_score:.2%}, "
                f"IoU/{mode}: {iou:.2%}, "
                f"lr: {self.optimizer.param_groups[0]['lr']:.6f}"
            )
