import numpy as np


class AverageMeter(object):
    """计算并存储平均值和当前值"""

    def __init__(self):
        self.val = None  # 当前值
        self.avg = None  # 平均值
        self.sum = None  # 总和
        self.count = None  # 计数
        # 初始化
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, batch_size):
        self.val = val
        self.sum += val * batch_size
        self.count += batch_size
        self.avg = self.sum / self.count


class ConfusionMatrix(object):
    """计算混淆矩阵"""

    def __init__(self):
        self.tp = None  # 真正例
        self.tn = None  # 真反例
        self.fp = None  # 假正例
        self.fn = None  # 假反例
        # 初始化
        self.reset()

    def reset(self):
        """重置"""
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def update(self, y_pred, y_true, threshold=0.5):
        """
        批量更新评价指标
        :param y_pred: 预测标签，四维numpy数组，形状为 (batch_size, 1, height, width)，经过sigmoid后的值
        :param y_true: 真实标签，四维numpy数组，形状为 (batch_size, 1, height, width)，值为0或1
        :param threshold: 阈值，用于将预测结果二值化，默认为0.5
        """
        # 判断 y_true 和 y_pred 维度是否相同
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"输入的 y_true 和 y_pred 维度必须相同，当前 y_true 形状为 {y_true.shape}，y_pred 形状为 {y_pred.shape}")

        # 去除 Channel 维度
        y_pred = y_pred.squeeze(axis=1)
        y_true = y_true.squeeze(axis=1)

        # 二值化预测结果
        y_pred_binary = (y_pred > threshold).astype(int)

        # 展平数组
        y_pred_flat = y_pred_binary.flatten()
        y_true_flat = y_true.flatten()

        # 计算并更新混淆矩阵元素
        self.tp += np.sum((y_true_flat == 1) & (y_pred_flat == 1))  # 真正例
        self.tn += np.sum((y_true_flat == 0) & (y_pred_flat == 0))  # 真反例
        self.fp += np.sum((y_true_flat == 0) & (y_pred_flat == 1))  # 假正例
        self.fn += np.sum((y_true_flat == 1) & (y_pred_flat == 0))  # 假反例

    def get_evaluate(self):
        """
        计算遥感变化检测图像二分类分割的评价指标
        :return: 准确率、精确率、召回率、F1分数、IoU
        """
        total_tp = self.tp
        total_tn = self.tn
        total_fp = self.fp
        total_fn = self.fn

        # 计算评价指标
        total_pixels = total_tp + total_tn + total_fp + total_fn
        accuracy = (total_tp + total_tn) / total_pixels if total_pixels > 0 else 0
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        iou = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0

        return accuracy, precision, recall, f1_score, iou
    
    
class ConfusionMatrixMulti(object):
    """多类别混淆矩阵与指标

    - 统一按 K 类（含背景）计算整体准确率、宏平均 Precision/Recall/F1、mIoU
    - 预测输入为 (B,H,W) 的整型类别图；标签输入同维度
    """

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.cm = None  # K x K 矩阵：rows=true, cols=pred
        self.reset()

    def reset(self):
        self.cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, y_pred: np.ndarray, y_true: np.ndarray):
        """
        :param y_pred: (B,H,W) 整型类别
        :param y_true: (B,H,W) 整型类别
        """
        if y_pred.shape != y_true.shape:
            raise ValueError(f"y_pred 与 y_true 形状不一致: {y_pred.shape} vs {y_true.shape}")

        # 展平
        y_pred_flat = y_pred.reshape(-1)
        y_true_flat = y_true.reshape(-1)

        # # 检查并校验标签合法性（y_true）
        # valid_true = (y_true_flat >= 0) & (y_true_flat < self.num_classes)
        # if not np.all(valid_true):
        #     invalid_indices_true = np.where(~valid_true)[0]
        #     invalid_values_true = y_true_flat[invalid_indices_true]
        #     raise ValueError(
        #         f"检测到非法标签！\n"
        #         f"非法值索引：{invalid_indices_true}\n"
        #         f"非法值内容：{invalid_values_true}\n"
        #         f"合法标签范围应是：0 ≤ 标签 < {self.num_classes}"
        #     )

        # # 检查并校验预测值合法性（y_pred）
        # valid_pred = (y_pred_flat >= 0) & (y_pred_flat < self.num_classes)
        # if not np.all(valid_pred):
        #     invalid_indices_pred = np.where(~valid_pred)[0]
        #     invalid_values_pred = y_pred_flat[invalid_indices_pred]
        #     raise ValueError(
        #         f"检测到非法预测值！\n"
        #         f"非法值索引：{invalid_indices_pred}\n"
        #         f"非法值内容：{invalid_values_pred}\n"
        #         f"合法预测值范围应是：0 ≤ 预测值 < {self.num_classes}"
        #     )

        # 累计混淆矩阵
        idx = y_true_flat * self.num_classes + y_pred_flat
        bincount = np.bincount(idx, minlength=self.num_classes * self.num_classes)
        self.cm += bincount.reshape(self.num_classes, self.num_classes)

    def get_evaluate(self, return_per_class=False):
        cm = self.cm.astype(np.float64)
        eps = 1e-7

        # overall accuracy
        acc = np.trace(cm) / (cm.sum() + eps)

        # per-class precision/recall/F1
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp  # 列和 - 对角线（每类被错误预测为该类的数量）
        fn = cm.sum(axis=1) - tp  # 行和 - 对角线（每类被错误预测为其他类的数量）

        precision_c = tp / (tp + fp + eps)
        recall_c = tp / (tp + fn + eps)
        f1_c = 2 * precision_c * recall_c / (precision_c + recall_c + eps)

        # 宏平均（忽略NaN，比如无样本的类）
        precision_macro = float(np.nanmean(precision_c))
        recall_macro = float(np.nanmean(recall_c))
        f1_macro = float(np.nanmean(f1_c))

        # IoU & mIoU（语义分割标准IoU）
        iou_c = tp / (tp + fp + fn + eps)
        # 可选：将无样本类的IoU设为NaN，避免拉低mIoU
        # iou_c = np.where((tp + fp + fn) == 0, np.nan, tp / (tp + fp + fn + eps))
        miou = float(np.nanmean(iou_c))

        if return_per_class:
            return acc, precision_macro, recall_macro, f1_macro, miou, precision_c, recall_c, f1_c, iou_c
        return acc, precision_macro, recall_macro, f1_macro, miou

    def print_cm(self):
        """打印混淆矩阵（调试用）"""
        print(f"\n混淆矩阵（类别数：{self.num_classes}，行=真实标签，列=预测标签）：")
        print(self.cm)
        
    def get_per_class_f1(self):
        """
        返回每一类的F1值
        :return: np.ndarray (num_classes,) - 按类别索引排序的F1值，例如 [0.9, 0.85, 0.7] 对应类0、类1、类2的F1
        """
        cm = self.cm.astype(np.float64)
        eps = 1e-7  # 避免分母为0

        # 计算每类的TP/FP/FN
        tp = np.diag(cm)  # 对角线为True Positive
        fp = cm.sum(axis=0) - tp  # 列和 - TP = False Positive
        fn = cm.sum(axis=1) - tp  # 行和 - TP = False Negative

        # 计算每类Precision/Recall/F1
        precision_c = tp / (tp + fp + eps)
        recall_c = tp / (tp + fn + eps)
        f1_c = 2 * precision_c * recall_c / (precision_c + recall_c + eps)

        return f1_c[1:]  # 排除背景类，返回前景类的F1