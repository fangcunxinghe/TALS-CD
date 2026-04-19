import timm
import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from models.base.BasicModule import BasicConv, BasicConvTranspose
from models.base.vim_encoder import vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2
from models.base.models_mamba import create_block


class MoDE(nn.Module):
    "MoDE"
    def __init__(self, in_channels, has_up):
        super(MoDE, self).__init__()
        self.in_channels = in_channels
        self.has_up = has_up
        self.router = Router(in_channels,out_num=3)
        
        # 多尺度融合后加工
        self.fusionConv = nn.Sequential(
            BasicConv(in_channels, in_channels, 3, padding=1),
            BasicConv(in_channels, in_channels, 3, padding=1)
        )
        
        # 若有上采样
        if self.has_up:
            self.up = nn.Sequential(
                BasicConvTranspose(in_channels, in_channels, kernel_size=2, stride=2),
            )
        
    def forward(self, M_l1,M_ls):
        B = M_l1.shape[0]
        weights=self.router(M_l1)
        if self.has_up:
            M_l1 = self.up(M_l1)
        y = M_l1 + sum(weights[:, i].view(B, 1, 1, 1) * M_ls[i] for i in range(3))
        y = self.fusionConv(y)

        return y

class FeatureExtractionModule(nn.Module):
    """变化特征提取模块"""

    def __init__(self):
        super(FeatureExtractionModule, self).__init__()
        self.feature_extraction_layer_4 = FeatureExtractionLayer_Mamba(40, 16)
        self.feature_extraction_layer_3 = FeatureExtractionLayer(40, 16)
        self.feature_extraction_layer_2 = FeatureExtractionLayer(24, 16)
        self.feature_extraction_layer_1 = FeatureExtractionLayer(16, 16)
        self.mode3=MoDE(16,False)
        self.mode2=MoDE(16,True)
        self.mode1=MoDE(16,True)
        

    def forward(self, As, Bs):
        fuses1 = self.feature_extraction_layer_1(As[0], Bs[0])
        fuses2 = self.feature_extraction_layer_2(As[1], Bs[1])
        fuses3 = self.feature_extraction_layer_3(As[2], Bs[2])
        fuses4, window_weights = self.feature_extraction_layer_4(As[3], Bs[3])
        fuses3 = self.mode3(fuses4,fuses3)
        fuses2 = self.mode2(fuses3,fuses2)
        fuses1 = self.mode1(fuses2,fuses1)
        return fuses1, window_weights


class FeatureExtractionLayer_Mamba(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(FeatureExtractionLayer_Mamba, self).__init__()

        # 变化提取单元
        self.cell_1 = ChangeExtractionCell_Mamba(in_channels, out_channels, 1)
        self.cell_2 = ChangeExtractionCell_Mamba(in_channels, out_channels, 1)
        self.cell_3 = ChangeExtractionCell_Mamba(in_channels, out_channels, 1)
        
        # 融合后加工
        self.fusionConv = nn.Sequential(
            BasicConv(out_channels, out_channels, 3, padding=1),
            BasicConv(out_channels, out_channels, 3, padding=1)
        )

    def forward(self, A, B):
        fus1, window_weights1 = self.cell_1(A, B)
        fus2, window_weights2 = self.cell_2(A, B)
        fus3, window_weights3 = self.cell_3(A, B)
        fuses = fus1 + fus2 + fus3  # [B,C,H,W]
        
        y = self.fusionConv(fuses)
        return y, (window_weights1, window_weights2, window_weights3)

class FeatureExtractionLayer(nn.Module):
    """变化特征提取层"""

    def __init__(self, in_channels, out_channels):
        super(FeatureExtractionLayer, self).__init__()

        # 变化提取单元
        self.cell_1 = ChangeExtractionCell_1(in_channels, out_channels)
        self.cell_2 = ChangeExtractionCell_2(in_channels, out_channels)
        self.cell_3 = ChangeExtractionCell_3(in_channels, out_channels)

    def forward(self, A, B):
        fus1 = self.cell_1(A, B)
        fus2 = self.cell_2(A, B)
        fus3 = self.cell_3(A, B)
        fuses = torch.stack([fus1, fus2, fus3], dim=0)  # [sources,B,C,H,W]

        return fuses
    
    
class DeformableWindowReorder(torch.autograd.Function):
    """
    Window-level deformable reordering
    Forward:  hard sorting (topk)
    Backward: manually assigned gradient to ordering score
    """

    @staticmethod
    def forward(ctx, x, score):
        """
        x:     (B, N, D)  window features
        score: (B, N)     ordering score (smaller = earlier)
        """
        B, N, D = x.shape

        # hard sorting
        _, indices = torch.topk(score, k=N, dim=1, largest=True)
        x_sorted = torch.gather(
            x, 1, indices.unsqueeze(-1).expand(-1, -1, D)
        ).contiguous()

        ctx.save_for_backward(score, indices)
        return x_sorted, indices

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        """
        grad_output: (B, N, D)
        """
        score, indices = ctx.saved_tensors
        B, N, D = grad_output.shape

        # gradient to x
        grad_x = torch.zeros(B, N, D, device=grad_output.device)
        grad_x.scatter_add_(
            1,
            indices.unsqueeze(-1).expand(-1, -1, D),
            grad_output
        )

        # gradient to score (path deformation signal)
        # intuition: encourage windows contributing more to move forward
        grad_score = grad_output.mean(dim=2)  # (B, N)
        grad_score = grad_score - grad_score.mean(dim=1, keepdim=True)

        return grad_x, grad_score

 
class MambaSnakeScanProcessor(nn.Module):
    """Mamba可学习的S蛇形扫描处理器"""
    
    def __init__(self, in_channels, window_height=2, window_width=2):
        super(MambaSnakeScanProcessor, self).__init__()
        self.in_channels = in_channels # 输入CNN特征的通道数
        self.window_h = window_height  # 窗口高度
        self.window_w = window_width   # 窗口宽度
        self.window_size = window_height * window_width  # 窗口内总元素数
        # 计算每个窗口的重要性权重
        self.window_weight_conv = nn.Sequential(
            BasicConv(in_channels, in_channels, 7, padding=3),
            BasicConv(in_channels, in_channels, 5, padding=2),
            BasicConv(in_channels, in_channels, 3, padding=1),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=1,  # 每个窗口输出1个权重
                kernel_size=(window_height, window_width),  # 卷积核=窗口大小（刚好覆盖1个窗口）
                stride=(window_height, window_width),       # 步长=窗口大小（不重叠，逐个扫窗口）
                padding=0
            )
        ) 
        
        self.mambaBlockVim=create_block(
            in_channels,
            d_state=16,
            ssm_cfg=None,
            norm_epsilon=1e-5,
            rms_norm=True,
            residual_in_fp32=True,
            fused_add_norm=False,
            layer_idx=0,
            if_bimamba=False,
            bimamba_type='v2',
            drop_path=0.1,
            if_divide_out=True,
            init_layer_scale=None
        )
        
        # Vim的Mamba块
        self.mambaBlock=create_block(
            in_channels,
            d_state=16,
            ssm_cfg=None,
            norm_epsilon=1e-5,
            rms_norm=True,
            residual_in_fp32=True,
            fused_add_norm=False,
            layer_idx=0,
            if_bimamba=True,
            bimamba_type='v2',
            drop_path=0.1,
            if_divide_out=True,
            init_layer_scale=None
        )
        # 前置线性层
        self.linearFirst=nn.Linear(in_channels,in_channels)
        self.linearFirstVim=nn.Linear(in_channels,in_channels)
        # 归一化层
        self.batchNorm=nn.BatchNorm2d(in_channels)
        
    def _mamba_scan_fn(self, mamba_input):
        """Mamba加工函数:输入为(B, seq, C)"""
        
        # 前置线性层
        y = self.linearFirst(mamba_input)
        # Mamba处理
        y = self.mambaBlock(y)[0]
        # 归一化
        # y = self.layerNorm(y)
        return y
        
    def forward(self, cnn_features):
        """
        完整流程：
        1. 特征图划分为多个窗口
        2. 计算每个窗口的重要性权重
        3. 窗口间按权重降序排序（重要窗口优先）
        4. 每个排序后的窗口内按蛇形排序
        5. 展平为(B, seq, C)输入Mamba
        6. Mamba处理后,逆向还原为原始空间布局
        """
        # 1. 解析输入特征形状
        B, C, H, W = cnn_features.shape  # B=批次，C=通道，H=高，W=宽
        window_h, window_w = self.window_h, self.window_w
        window_size = self.window_size

        # 检查特征图尺寸是否能被窗口大小整除（否则窗口无法完整覆盖）
        assert H % window_h == 0 and W % window_w == 0, \
            f"特征图{H}×{W}必须能被窗口{window_h}×{window_w}整除"
        
        # 2. 计算窗口数量
        num_windows_h = H // window_h  # 高度方向窗口数
        num_windows_w = W // window_w  # 宽度方向窗口数
        num_windows = num_windows_h * num_windows_w  # 总窗口数
        seq_length = num_windows * window_size  # 序列总长度
        
        # -------------------------- 步骤1：特征图划分为窗口 --------------------------
        # 原始形状(B,C,H,W) → 拆分为窗口结构：(B, 窗口行数, 窗口列数, C, 窗口高, 窗口宽)
        windowed = cnn_features.view(
            B, C, 
            num_windows_h, window_h,  # 高度方向：拆为“窗口行数+窗口高”
            num_windows_w, window_w   # 宽度方向：拆为“窗口列数+窗口宽”
        ).permute(0, 2, 4, 1, 3, 5).contiguous()  # 调整维度顺序，让窗口信息在前

        # -------------------------- 步骤2：计算每个窗口的重要性权重 --------------------------
        # 用卷积计算窗口权重图：(B,C,H,W) → (B,1,num_windows_h,num_windows_w)
        window_weight_map = self.window_weight_conv(cnn_features)
        # 展平权重：(B,1,num_windows_h,num_windows_w) → (B, 总窗口数)
        window_weights = window_weight_map.view(B, num_windows)
        window_weights = torch.sigmoid(window_weights)
        
        # -------------------------- 步骤3：窗口间按重要性降序排序 --------------------------
        # 对每个样本的窗口权重降序排序，获取排序索引
        windows_flat = windowed.view(B, num_windows, -1)  # (B, N, C*wh*ww)

        # deformable hard reordering
        sorted_windows, sort_idx = DeformableWindowReorder.apply(
            windows_flat,
            window_weights
        )
        
        # -------------------------- 步骤4：窗口内按蛇形排序 --------------------------
        # 生成蛇形索引
        snake_indices = self._generate_snake_indices(device=cnn_features.device)

        # 每个窗口展平为1维：(B, 总窗口数, C, 窗口高, 窗口宽) → (B, 总窗口数, C, 窗口大小)
        flattened_sorted_windows = sorted_windows.view(B, num_windows, C, -1)

        # 应用蛇形排序：重排窗口内元素顺序
        snake_sorted_windows = flattened_sorted_windows[:, :, :, snake_indices]
        
        # -------------------------- 步骤5：展平为(B, seq, C)输入Mamba --------------------------
        # 调整维度：(B, 总窗口数, C, 窗口大小) → (B, 总窗口数, 窗口大小, C)（让元素维度在前）
        per_window_seq = snake_sorted_windows.permute(0, 1, 3, 2).contiguous()
        
        # 展平为序列：(B, 总窗口数×窗口大小, C) → (B, seq, C)
        mamba_input = per_window_seq.view(B, seq_length, C)
        
        # -------------------------- 步骤6：Mamba处理 --------------------------
        # 调用Mamba扫描函数（输入输出均为(B, seq, C)）
        mamba_output = self._mamba_scan_fn(mamba_input)
        
        # -------------------------- 步骤7：逆向还原为原始空间布局 --------------------------
        # 还原步骤1：将Mamba输出拆回“排序后的窗口+蛇形顺序”结构
        # (B, seq, C) → (B, 总窗口数, 窗口大小, C)
        scanned_per_window = mamba_output.view(B, num_windows, window_size, C)

        # 还原步骤2：窗口内逆蛇形排序（恢复窗口内原始元素顺序）
        # 计算蛇形索引的逆索引
        _, inv_snake_indices = torch.sort(snake_indices)
        # 调整维度：(B, 总窗口数, 窗口大小, C) → (B, 总窗口数, C, 窗口大小)
        scanned_per_window = scanned_per_window.permute(0, 1, 3, 2).contiguous()
        # 应用逆蛇形排序：恢复窗口内原始顺序
        inv_snake_windows = scanned_per_window[:, :, :, inv_snake_indices]

        # 硬逆排序：用伪硬索引的逆序恢复窗口原始顺序  
        # 计算逆索引：inv_sort_idx[sort_idx] = 原始位置（如sort_idx=[2,0,1] → inv_sort_idx=[1,2,0]）  
        inv_sort_idx = torch.argsort(sort_idx, dim=1)  # (B, N)，逆索引  
        # 硬逆重排：用逆索引恢复窗口顺序（离散操作，梯度断裂）  
        inv_snake_windows = inv_snake_windows.view(B, num_windows, C, window_h, window_w)  # 恢复形状 (B, N, C, wh, ww)  
        batch_indices = torch.arange(B, device=cnn_features.device).unsqueeze(1).expand(-1, num_windows)  # (B, N)  
        restored_windows = inv_snake_windows[batch_indices, inv_sort_idx]  # (B, N, C, wh, ww)，硬逆重排  


        # 还原步骤4：拼接窗口为原始特征图形状
        # 重塑为窗口空间布局：(B, 总窗口数, C, 窗口高, 窗口宽) → (B, 窗口行数, 窗口列数, C, 窗口高, 窗口宽)
        restored_windows = restored_windows.view(B, num_windows_h, num_windows_w, C, window_h, window_w)
        # 调整维度并拼接：(B, 窗口行数, 窗口列数, C, 窗口高, 窗口宽) → (B, C, H, W)
        restored_features = restored_windows.permute(0, 3, 1, 4, 2, 5).contiguous()
        restored_features = restored_features.view(B, C, H, W)
        
        # cnn_features展平为序列
        cnn_features_flat = rearrange(cnn_features, 'b c h w -> b (h w) c')
        # 结合Vim的Mamba块处理
        cnn_lin=  self.linearFirstVim(cnn_features_flat)
        mamba_main= self.mambaBlockVim(cnn_lin)[0]
        # 还原形为(B,C,H,W)
        mamba_main= rearrange(mamba_main, 'b (h w) c -> b c h w', h=H, w=W)
        # 融合Mamba主路与窗口重排支路
        mamba_tiaozhi= self.batchNorm(mamba_main * torch.sigmoid(restored_features))
        
        return mamba_tiaozhi + cnn_features, window_weight_map  # 残差连接
    
    def _generate_snake_indices(self, device):
        """
        生成窗口内蛇形扫描索引:奇数行(0开始)正向,偶数行反向
        例:3*4窗口 → 索引[0,1,2,3,7,6,5,4,8,9,10,11]
        """
        snake_indices = []
        for row in range(self.window_h):
            if row % 2 == 0:  # 第0、2、4...行：正向（左→右）
                row_idx = torch.arange(self.window_w, device=device)
            else:  # 第1、3、5...行：反向（右→左）
                row_idx = torch.arange(self.window_w - 1, -1, -1, device=device)
            # 计算当前行在窗口展平后的起始位置（如第1行起始=1×4=4）
            start_pos = row * self.window_w
            snake_indices.append(start_pos + row_idx)
        # 拼接所有行的索引，得到窗口内蛇形顺序（如3×4→12个索引）
        return torch.cat(snake_indices, dim=0)

class ChangeExtractionCell_Mamba(nn.Module):

    def __init__(self, in_channels, out_channels, window_size):
        super(ChangeExtractionCell_Mamba, self).__init__()

        # 提取前先加工
        self.convA = BasicConv(in_channels, in_channels, 3, padding=1)
        self.convB = BasicConv(in_channels, in_channels, 3, padding=1)

        # 提取后加工
        self.extractConv = nn.Sequential(
            BasicConv(in_channels * 2, in_channels, 1),
            BasicConv(in_channels, out_channels, 1),
            BasicConv(out_channels, out_channels, 3, padding=1)
            # BasicConv(in_channels * 2, out_channels, 3, padding=1)
        )
        # 提取后加工2
        self.extractConv2 = nn.Sequential(
            BasicConv(out_channels, out_channels, 3, padding=1)
        )
        
        # mamba蛇形扫描处理
        self.mambaSnakeScan = MambaSnakeScanProcessor(
            in_channels=out_channels,
            window_height = window_size,
            window_width = window_size
        )  

    def forward(self, A, B):
        # 差异特征提取先加工特征
        A = self.convA(A)
        B = self.convB(B)
        
         # 提取差异特征
        diff = torch.cat([A, B], 1)
        # 提取后加工
        y_cnn = self.extractConv(diff)
        
        mamba_outputs, window_weights = self.mambaSnakeScan(y_cnn)
        
        # 提取后加工
        y = self.extractConv2(mamba_outputs)

        return y, window_weights
    
class ChangeExtractionCell_1(nn.Module):
    """变化提取单元1——|A-B|+concat"""

    def __init__(self, in_channels, out_channels):
        super(ChangeExtractionCell_1, self).__init__()

        # 提取前先加工
        self.convA = BasicConv(in_channels, in_channels, 3, padding=1)
        self.convB = BasicConv(in_channels, in_channels, 3, padding=1)

        # 提取后加工
        self.extractConv = nn.Sequential(
            BasicConv(in_channels * 3, in_channels, 1),
            BasicConv(in_channels, out_channels, 1),
            BasicConv(out_channels, out_channels, 3, padding=1)
        )

    def forward(self, A, B):
        # 差异特征提取先加工特征
        A = self.convA(A)
        B = self.convB(B)

        # 提取差异特征
        sub = torch.abs(A - B)
        diff = torch.cat([sub, A, B], 1)
        # 提取后加工
        y = self.extractConv(diff)

        return y

class ChangeExtractionCell_2(nn.Module):
    """变化提取单元2——Cosin通道"""

    def __init__(self, in_channels, out_channels):
        super(ChangeExtractionCell_2, self).__init__()

        # 余弦相似度可学习参数
        self.scale = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.bias = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        # 提取前先加工
        self.convA = BasicConv(in_channels, in_channels, 3, padding=1)
        self.convB = BasicConv(in_channels, in_channels, 3, padding=1)

        # 提取后加工
        self.extractConv = nn.Sequential(
            BasicConv(in_channels * 2, in_channels, 1),
            BasicConv(in_channels, out_channels, 1),
            BasicConv(out_channels, out_channels, 3, padding=1)
        )

    def forward(self, A, B):
        # 差异特征提取先加工特征
        A = self.convA(A)
        B = self.convB(B)

        # 提取差异特征(cos余弦相似度)

        # 确保输入形状一致
        assert A.shape == B.shape, "两张图像的形状必须相同"

        b, c, h, w = A.shape  # 保存原始形状用于结果重塑

        # 在空间维度上展平，得到 [B, C, H*W]
        A_flat = A.view(b, c, -1)
        B_flat = B.view(b, c, -1)

        # 对每个空间位置的特征向量进行L2归一化
        A_norm = F.normalize(A_flat, p=2, dim=2)  # [B, C, H*W]
        B_norm = F.normalize(B_flat, p=2, dim=2)  # [B, C, H*W]

        # 计算每个通道的全局余弦相似度
        cos_similarity = torch.sum(A_norm * B_norm, dim=2, keepdim=True)  # [B, C, 1]
        cos_similarity = cos_similarity.view(b, c, 1, 1)  # 重塑回 [B, C, 1, 1]
        # 转换为变化特征
        change_score = torch.sigmoid((1.0 - cos_similarity) * self.scale + self.bias)

        # 求变化
        A = A * change_score
        B = B * change_score
        # 提取差异特征
        diff = torch.concat([A, B], dim=1)
        # 提取后加工
        y = self.extractConv(diff)

        return y


class ChangeExtractionCell_3(nn.Module):
    """变化提取单元3——Cosin空间"""

    def __init__(self, in_channels, out_channels):
        super(ChangeExtractionCell_3, self).__init__()
        # 余弦相似度可学习参数
        self.scale = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.bias = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        # 提取前先加工
        self.convA = BasicConv(in_channels, in_channels, 3, padding=1)
        self.convB = BasicConv(in_channels, in_channels, 3, padding=1)

        # 提取后加工
        self.extractConv = nn.Sequential(
            BasicConv(in_channels * 2, in_channels, 1),
            BasicConv(in_channels, out_channels, 1),
            BasicConv(out_channels, out_channels, 3, padding=1)
        )

    def forward(self, A, B):
        # 差异特征提取先加工特征
        A = self.convA(A)
        B = self.convB(B)

        # 提取差异特征(cos余弦相似度)

        # 确保输入形状一致
        assert A.shape == B.shape, "两张图像的形状必须相同"

        b, c, h, w = A.shape  # 保存原始形状用于结果重塑

        # 在空间维度上展平，得到 [B, C, H*W]
        A_flat = A.view(b, c, -1)
        B_flat = B.view(b, c, -1)

        # 对每个位置的通道向量进行L2归一化
        A_norm = F.normalize(A_flat, p=2, dim=1)  # [B, C, H*W]
        B_norm = F.normalize(B_flat, p=2, dim=1)  # [B, C, H*W]

        # 计算每个位置的通道向量余弦相似度
        cos_similarity = torch.sum(A_norm * B_norm, dim=1, keepdim=True)  # [B, 1, H*W]

        cos_similarity = cos_similarity.view(b, 1, h, w)  # 重塑回 [B, 1, H, W]

        # 转换为变化特征
        change_score = torch.sigmoid((1.0 - cos_similarity) * self.scale + self.bias)

        # 求变化
        A = A * change_score
        B = B * change_score
        diff = torch.concat([A, B], dim=1)
        # 提取后加工
        y = self.extractConv(diff)
        return y


class Router(nn.Module):
    """变化路由"""

    def __init__(self, input_dim, hidden_dim=32, out_num=3):
        super(Router, self).__init__()
        # 全局平均池化层
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, out_num)
        )

    def forward(self, x):
        # x.shape = B,C,H,W
        x = self.pool(x)
        x = rearrange(x, "b c h w -> b (c h w)")
        x = self.mlp(x)
        soft_weight = torch.sigmoid(x)

        return soft_weight


class TALS_CD(nn.Module):
    def __init__(self):
        super(TALS_CD, self).__init__()

        # mobile_net_v3 特征提取
        self.mobile_net_v3_A = timm.create_model(
            'mobilenetv3_large_100.ra_in1k',
            pretrained=False,
            features_only=True
        )
        self.mobile_net_v3_B = timm.create_model(
            'mobilenetv3_large_100.ra_in1k',
            pretrained=False,
            features_only=True
        )
        # 读取预训练权重文件
        weight_path = 'pretrain/mobilenetv3_large_100.bin'
        state_dict = torch.load(weight_path)
        # 过滤掉不需要的键
        state_dict = {k: v for k, v in state_dict.items() if k in self.mobile_net_v3_A.state_dict()}
        # 加载MobileNet V3预训练权重
        self.mobile_net_v3_A.load_state_dict(state_dict, strict=True)
        self.mobile_net_v3_B.load_state_dict(state_dict, strict=True)
        # 删除不必要的模块
        del self.mobile_net_v3_A.blocks[6]
        del self.mobile_net_v3_A.blocks[5]
        del self.mobile_net_v3_A.blocks[4]
        del self.mobile_net_v3_A.blocks[3]
        del self.mobile_net_v3_B.blocks[6]
        del self.mobile_net_v3_B.blocks[5]
        del self.mobile_net_v3_B.blocks[4]
        del self.mobile_net_v3_B.blocks[3]

        self.fe = FeatureExtractionModule()

        self.classifier = nn.Sequential(
            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 4, 1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 1, 1),
        )
        
        self.embedA=nn.Conv2d(40,192,kernel_size=4,stride=4)
        self.embedB=self.embedA
        self.re_embedA=nn.ConvTranspose2d(192,40,kernel_size=4,stride=4)
        self.re_embedB=self.re_embedA
        
        self.vim=vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2()

         # 读取预训练权重文件
        weight_path = 'pretrain/vim_t_midclstok_ft_78p3acc.pth'
        state_dict = torch.load(weight_path)['model']
        # 过滤掉不需要的键
        state_dict = {k: v for k, v in state_dict.items() if k in self.vim.state_dict()}
        # 加载权重
        self.vim.load_state_dict(state_dict, strict=True)


    def forward(self, A, B):
        layer1_A, layer2_A, layer3_A = self.mobile_net_v3_A(A)
        layer1_B, layer2_B, layer3_B = self.mobile_net_v3_B(B)
        # embed
        layer4_A_embed=self.embedA(layer3_A)
        layer4_B_embed=self.embedB(layer3_B)
        
        b, c, h, w = layer4_A_embed.shape
        layer4_A_embed = rearrange(layer4_A_embed, "b c h w -> b (h w) c")
        layer4_B_embed = rearrange(layer4_B_embed, "b c h w -> b (h w) c")
        # print(f"Mamba_In: {layer4_A_embed.shape}")
        
        # vim 特征提取
        layer4_A_embed=self.vim(layer4_A_embed)
        layer4_B_embed=self.vim(layer4_B_embed)
        # print(f"Mamba_Out: {layer4_A_embed.shape}")

        # 重新排列维度 b (h w) c -> b c h w
        layer4_A_embed = rearrange(layer4_A_embed, "b (h w) c -> b c h w", h=h, w=w)
        layer4_B_embed = rearrange(layer4_B_embed, "b (h w) c -> b c h w", h=h, w=w)
        
        layer4_A=self.re_embedA(layer4_A_embed)
        layer4_B=self.re_embedB(layer4_B_embed)
        # print(f"Mamba_Re_Embed: {layer4_A.shape}")
        
        # 差异特征提取与融合
        fus, window_weights = self.fe([layer1_A, layer2_A, layer3_A, layer4_A], [layer1_B, layer2_B, layer3_B,layer4_B])
        
        y = self.classifier(fus)
        
        # print(f"Final_Out: {y.shape}")
        
        return y, window_weights
    