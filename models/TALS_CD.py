import timm
import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from models.base.BasicModule import BasicConv, BasicConvTranspose
from models.base.vim_encoder import vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2
from models.base.models_mamba import create_block

class FeatureExtractionModule(nn.Module):

    def __init__(self):
        super(FeatureExtractionModule, self).__init__()
        self.feature_extraction_layer_1 = FeatureExtractionLayer_Mamba(40, 24, is_first=True, is_last=False)
        self.feature_extraction_layer_2 = FeatureExtractionLayer(24, 16, is_first=False, is_last=False)
        self.feature_extraction_layer_3 = FeatureExtractionLayer(16, 16, is_first=False, is_last=True)

    def forward(self, As, Bs):
        fuses1, weights1 = self.feature_extraction_layer_1(As[3], Bs[3])
        fuses2, weights2 = self.feature_extraction_layer_2(As[1], Bs[1], fuses1, weights1)
        fuses3, _ = self.feature_extraction_layer_3(As[0], Bs[0], fuses2, weights2)

        return fuses3


class FeatureExtractionLayer_Mamba(nn.Module):

    def __init__(self, in_channels, out_channels, is_first=False, is_last=False):
        super(FeatureExtractionLayer_Mamba, self).__init__()
        self.is_first = is_first
        self.is_last = is_last

        self.cell_1 = ChangeExtractionCell_Mamba(in_channels, out_channels, 2,is_first)
        self.cell_2 = ChangeExtractionCell_Mamba(in_channels, out_channels, 4,is_first)
        self.cell_3 = ChangeExtractionCell_Mamba(in_channels, out_channels, 8,is_first)

        if not is_last:
            self.router1 = Router(out_channels)
            self.router2 = Router(out_channels)
            self.router3 = Router(out_channels)

    def forward(self, A, B, pre=None, pre_weights=None):
        if not self.is_first:
            fus1 = self.cell_1(A, B, pre, pre_weights[0])
            fus2 = self.cell_2(A, B, pre, pre_weights[1])
            fus3 = self.cell_3(A, B, pre, pre_weights[2])
        else:
            fus1 = self.cell_1(A, B)
            fus2 = self.cell_2(A, B)
            fus3 = self.cell_3(A, B)
        fuses = torch.stack([fus1, fus2, fus3], dim=0)  # [sources,B,C,H,W]

        if not self.is_last:
            weight1 = self.router1(fus1)  # [B,router_num]
            weight2 = self.router2(fus2)  # [B,router_num]
            weight3 = self.router3(fus3)  # [B,router_num]

            weights = torch.stack([weight1, weight2, weight3], dim=0)  # [sources,B,router_num]
            weights = weights.permute(2, 0, 1).unsqueeze(3).unsqueeze(3).unsqueeze(3)  # [router_num,sources,B,1,1,1]

            return fuses, weights
        else:
            return fuses, None

class FeatureExtractionLayer(nn.Module):
    """FeatureExtractionLayer"""

    def __init__(self, in_channels, out_channels, is_first=False, is_last=False):
        super(FeatureExtractionLayer, self).__init__()
        self.is_first = is_first
        self.is_last = is_last

        self.cell_1 = ChangeExtractionCell_1(in_channels, out_channels, is_first)
        self.cell_2 = ChangeExtractionCell_2(in_channels, out_channels, is_first)
        self.cell_3 = ChangeExtractionCell_3(in_channels, out_channels, is_first)

        if not is_last:
            self.router1 = Router(out_channels)
            self.router2 = Router(out_channels)
            self.router3 = Router(out_channels)

    def forward(self, A, B, pre=None, pre_weights=None):
        if not self.is_first:
            fus1 = self.cell_1(A, B, pre, pre_weights[0])
            fus2 = self.cell_2(A, B, pre, pre_weights[1])
            fus3 = self.cell_3(A, B, pre, pre_weights[2])
        else:
            fus1 = self.cell_1(A, B)
            fus2 = self.cell_2(A, B)
            fus3 = self.cell_3(A, B)
        fuses = torch.stack([fus1, fus2, fus3], dim=0)  # [sources,B,C,H,W]

        if not self.is_last:
            weight1 = self.router1(fus1)  # [B,router_num]
            weight2 = self.router2(fus2)  # [B,router_num]
            weight3 = self.router3(fus3)  # [B,router_num]

            weights = torch.stack([weight1, weight2, weight3], dim=0)  # [sources,B,router_num]
            weights = weights.permute(2, 0, 1).unsqueeze(3).unsqueeze(3).unsqueeze(3)  # [router_num,sources,B,1,1,1]

            return fuses, weights
        else:
            return fuses, None
        
        
class MambaSnakeScanProcessor(nn.Module):
    """MambaSnakeScanProcessor"""
    
    def __init__(self, in_channels, window_height=2, window_width=2):
        super(MambaSnakeScanProcessor, self).__init__()
        self.in_channels = in_channels 
        self.window_h = window_height 
        self.window_w = window_width 
        self.window_size = window_height * window_width 
        
        self.window_weight_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=1,  
            kernel_size=(window_height, window_width), 
            stride=(window_height, window_width), 
            padding=0
        )
        
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
        self.linearFirst=nn.Linear(in_channels,in_channels)
        self.layerNorm=nn.LayerNorm(in_channels)
        
    def _mamba_scan_fn(self, mamba_input):     
        
        y = self.linearFirst(mamba_input)
        y = self.mambaBlock(y)[0]
        y = self.layerNorm(y)
        return y
        
    def forward(self, cnn_features):
        B, C, H, W = cnn_features.shape
        window_h, window_w = self.window_h, self.window_w
        window_size = self.window_size
        assert H % window_h == 0 and W % window_w == 0, \
            f"The feature map of size {H}×{W} must be divisible by the window of size {window_h}×{window_w}."
        num_windows_h = H // window_h
        num_windows_w = W // window_w
        num_windows = num_windows_h * num_windows_w
        seq_length = num_windows * window_size
        
        windowed = cnn_features.view(
            B, C, 
            num_windows_h, window_h,  
            num_windows_w, window_w   
        ).permute(0, 2, 4, 1, 3, 5).contiguous()  

        windowed = windowed.view(B, num_windows, C, window_h, window_w)
        
        # (B,C,H,W) → (B,1,num_windows_h,num_windows_w)
        window_weight_map = self.window_weight_conv(cnn_features)
        # (B,1,num_windows_h,num_windows_w) → (B, num_windows, 1)
        window_weights = window_weight_map.view(B, 1, num_windows).permute(0, 2, 1).contiguous()
        # Sigmoid
        window_weights = torch.sigmoid(window_weights)
        
        _, window_sorted_indices = torch.sort(-window_weights.squeeze(-1), dim=1)

        batch_indices = torch.arange(B, device=cnn_features.device).unsqueeze(1).expand(-1, num_windows)

        sorted_windows = windowed[batch_indices, window_sorted_indices]
        snake_indices = self._generate_snake_indices(device=cnn_features.device)
        flattened_sorted_windows = sorted_windows.view(B, num_windows, C, -1)
        snake_sorted_windows = flattened_sorted_windows[:, :, :, snake_indices]
        per_window_seq = snake_sorted_windows.permute(0, 1, 3, 2).contiguous()
        
        mamba_input = per_window_seq.view(B, seq_length, C)
        # (B, seq, C)
        mamba_output = self._mamba_scan_fn(mamba_input)
        scanned_per_window = mamba_output.view(B, num_windows, window_size, C)
        _, inv_snake_indices = torch.sort(snake_indices)

        scanned_per_window = scanned_per_window.permute(0, 1, 3, 2).contiguous()
        
        inv_snake_windows = scanned_per_window[:, :, :, inv_snake_indices]
        _, inv_window_indices = torch.sort(window_sorted_indices, dim=1)
        inv_sorted_windows = inv_snake_windows.view(B, num_windows, C, window_h, window_w)
        restored_windows = inv_sorted_windows[batch_indices, inv_window_indices]

        restored_windows = restored_windows.view(B, num_windows_h, num_windows_w, C, window_h, window_w)
        # (B, C, H, W)
        restored_features = restored_windows.permute(0, 3, 1, 4, 2, 5).contiguous()
        restored_features = restored_features.view(B, C, H, W)
        
        return restored_features
    
    def _generate_snake_indices(self, device):
        snake_indices = []
        for row in range(self.window_h):
            if row % 2 == 0:  # 0、2、4...
                row_idx = torch.arange(self.window_w, device=device)
            else:  # 1、3、5...
                row_idx = torch.arange(self.window_w - 1, -1, -1, device=device)
            start_pos = row * self.window_w
            snake_indices.append(start_pos + row_idx)
            
        return torch.cat(snake_indices, dim=0)
    

class ChangeExtractionCell_Mamba(nn.Module):

    def __init__(self, in_channels, out_channels, window_size, is_first=False):
        super(ChangeExtractionCell_Mamba, self).__init__()
        self.is_first = is_first

        self.convA = BasicConv(in_channels, in_channels, 3, padding=1)
        self.convB = BasicConv(in_channels, in_channels, 3, padding=1)

        self.extractConv = nn.Sequential(
            BasicConv(in_channels * 2, in_channels, 1),
            BasicConv(in_channels, out_channels, 1),
            BasicConv(out_channels, out_channels, 3, padding=1)
            # BasicConv(in_channels * 2, out_channels, 3, padding=1)
        )
        self.extractConv2 = nn.Sequential(
            BasicConv(out_channels, out_channels, 3, padding=1)
        )
        
        self.mambaSnakeScan = MambaSnakeScanProcessor(
            in_channels=out_channels,
            window_height = window_size,
            window_width = window_size
        )
        
        if not self.is_first:
            self.up = nn.Sequential(
                # BasicConv(in_channels, out_channels, 1),
                # BasicConv(out_channels, out_channels, 3, padding=1),
                BasicConvTranspose(in_channels, out_channels, kernel_size=2, stride=2),
            )
            self.fusionConv = nn.Sequential(
                BasicConv(out_channels * 2, out_channels, 1),
                BasicConv(out_channels, out_channels, 3, padding=1)
            )
            

    def forward(self, A, B, pre=None, pre_weight=None):
        A = self.convA(A)
        B = self.convB(B)
        
        # sub = torch.abs(A - B)
        diff = torch.cat([A, B], 1)
        y_cnn = self.extractConv(diff)
        
        mamba_outputs = self.mambaSnakeScan(y_cnn)

        y = self.extractConv2(mamba_outputs)

        if not self.is_first:
            pre = pre * pre_weight
            pre = torch.sum(pre, dim=0)
            pre_up = self.up(pre)
            y_fus = torch.cat([y, pre_up], dim=1)
            y = self.fusionConv(y_fus)

        return y
    
class ChangeExtractionCell_1(nn.Module):
    """|A-B|+concat"""

    def __init__(self, in_channels, out_channels, is_first=False):
        super(ChangeExtractionCell_1, self).__init__()
        self.is_first = is_first

        self.convA = BasicConv(in_channels, in_channels, 3, padding=1)
        self.convB = BasicConv(in_channels, in_channels, 3, padding=1)

        self.extractConv = nn.Sequential(
            BasicConv(in_channels * 3, in_channels, 1),
            BasicConv(in_channels, out_channels, 1),
            BasicConv(out_channels, out_channels, 3, padding=1)
        )

        if not self.is_first:
            self.up = nn.Sequential(
                # BasicConv(in_channels, out_channels, 1),
                # BasicConv(out_channels, out_channels, 3, padding=1),
                BasicConvTranspose(in_channels, out_channels, kernel_size=2, stride=2),
            )

            self.fusionConv = nn.Sequential(
                BasicConv(out_channels * 2, out_channels, 1),
                BasicConv(out_channels, out_channels, 3, padding=1)
            )

    def forward(self, A, B, pre=None, pre_weight=None):
        A = self.convA(A)
        B = self.convB(B)

        sub = torch.abs(A - B)
        diff = torch.cat([sub, A, B], 1)
        y = self.extractConv(diff)

        if not self.is_first:
            pre = pre * pre_weight
            pre = torch.sum(pre, dim=0)
            pre_up = self.up(pre)
            y_fus = torch.cat([y, pre_up], dim=1)
            y = self.fusionConv(y_fus)

        return y


class ChangeExtractionCell_2(nn.Module):

    def __init__(self, in_channels, out_channels, is_first=False):
        super(ChangeExtractionCell_2, self).__init__()
        self.is_first = is_first

        self.scale = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.bias = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        self.convA = BasicConv(in_channels, in_channels, 3, padding=1)
        self.convB = BasicConv(in_channels, in_channels, 3, padding=1)

        self.extractConv = nn.Sequential(
            BasicConv(in_channels * 2, in_channels, 1),
            BasicConv(in_channels, out_channels, 1),
            BasicConv(out_channels, out_channels, 3, padding=1)
        )

        if not self.is_first:
            self.up = nn.Sequential(
                # BasicConv(in_channels, out_channels, 1),
                # BasicConv(out_channels, out_channels, 3, padding=1),
                BasicConvTranspose(in_channels, out_channels, kernel_size=2, stride=2),
            )

            self.fusionConv = nn.Sequential(
                BasicConv(out_channels * 2, out_channels, 1),
                BasicConv(out_channels, out_channels, 3, padding=1)
            )

    def forward(self, A, B, pre=None, pre_weight=None):
        A = self.convA(A)
        B = self.convB(B)

        b, c, h, w = A.shape

        # [B, C, H*W]
        A_flat = A.view(b, c, -1)
        B_flat = B.view(b, c, -1)

        # L2
        A_norm = F.normalize(A_flat, p=2, dim=2)  # [B, C, H*W]
        B_norm = F.normalize(B_flat, p=2, dim=2)  # [B, C, H*W]

        cos_similarity = torch.sum(A_norm * B_norm, dim=2, keepdim=True)  # [B, C, 1]
        cos_similarity = cos_similarity.view(b, c, 1, 1)  # [B, C, 1, 1]

        change_score = torch.sigmoid((1.0 - cos_similarity) * self.scale + self.bias)

        A = A * change_score
        B = B * change_score
        diff = torch.concat([A, B], dim=1)

        y = self.extractConv(diff)

        if not self.is_first:
            pre = pre * pre_weight
            pre = torch.sum(pre, dim=0)
            pre_up = self.up(pre)
            y_fus = torch.cat([y, pre_up], dim=1)
            y = self.fusionConv(y_fus)

        return y


class ChangeExtractionCell_3(nn.Module):

    def __init__(self, in_channels, out_channels, is_first=False):
        super(ChangeExtractionCell_3, self).__init__()
        self.is_first = is_first

        self.scale = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.bias = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        self.convA = BasicConv(in_channels, in_channels, 3, padding=1)
        self.convB = BasicConv(in_channels, in_channels, 3, padding=1)

        self.extractConv = nn.Sequential(
            BasicConv(in_channels * 2, in_channels, 1),
            BasicConv(in_channels, out_channels, 1),
            BasicConv(out_channels, out_channels, 3, padding=1)
        )

        if not self.is_first:
            self.up = nn.Sequential(
                # BasicConv(in_channels, out_channels, 1),
                # BasicConv(out_channels, out_channels, 3, padding=1),
                BasicConvTranspose(in_channels, out_channels, kernel_size=2, stride=2),
            )

            self.fusionConv = nn.Sequential(
                BasicConv(out_channels * 2, out_channels, 1),
                BasicConv(out_channels, out_channels, 3, padding=1)
            )

    def forward(self, A, B, pre=None, pre_weight=None):
        A = self.convA(A)
        B = self.convB(B)

        b, c, h, w = A.shape 

        # [B, C, H*W]
        A_flat = A.view(b, c, -1)
        B_flat = B.view(b, c, -1)

        # L2
        A_norm = F.normalize(A_flat, p=2, dim=1)  # [B, C, H*W]
        B_norm = F.normalize(B_flat, p=2, dim=1)  # [B, C, H*W]
        cos_similarity = torch.sum(A_norm * B_norm, dim=1, keepdim=True)  # [B, 1, H*W]

        cos_similarity = cos_similarity.view(b, 1, h, w)  # [B, 1, H, W]

        change_score = torch.sigmoid((1.0 - cos_similarity) * self.scale + self.bias)

        A = A * change_score
        B = B * change_score

        diff = torch.concat([A, B], dim=1)

        y = self.extractConv(diff)

        if not self.is_first:
            pre = pre * pre_weight
            pre = torch.sum(pre, dim=0)
            pre_up = self.up(pre)
            y_fus = torch.cat([y, pre_up], dim=1)
            y = self.fusionConv(y_fus)

        return y


class Router(nn.Module):

    def __init__(self, input_dim, hidden_dim=32, out_num=3):
        super(Router, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            # nn.ReLU(True),
            nn.Linear(hidden_dim, out_num)
        )

    def forward(self, x):
        # x.shape = B,C,H,W
        x = self.pool(x)
        x = rearrange(x, "b c h w -> b (c h w)")
        x = self.mlp(x)
        soft_weight = torch.sigmoid(x)

        return soft_weight


class TALSCD(nn.Module):
    def __init__(self):
        super(TALSCD, self).__init__()

        # mobile_net_v3
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
        weight_path = '/home/xlf/change_detection_experiment/pretrain/mobilenetv3_large_100.bin'
        state_dict = torch.load(weight_path)
        state_dict = {k: v for k, v in state_dict.items() if k in self.mobile_net_v3_A.state_dict()}
        # load MobileNet V3
        self.mobile_net_v3_A.load_state_dict(state_dict, strict=True)
        self.mobile_net_v3_B.load_state_dict(state_dict, strict=True)
        # remove more model
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

        weight_path = '/home/xlf/vim_t_midclstok_ft_78p3acc.pth'
        state_dict = torch.load(weight_path)['model']
        state_dict = {k: v for k, v in state_dict.items() if k in self.vim.state_dict()}
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
        
        # vim
        layer4_A_embed=self.vim(layer4_A_embed)
        layer4_B_embed=self.vim(layer4_B_embed)
        # print(f"Mamba_Out: {layer4_A_embed.shape}")

        # b (h w) c -> b c h w
        layer4_A_embed = rearrange(layer4_A_embed, "b (h w) c -> b c h w", h=h, w=w)
        layer4_B_embed = rearrange(layer4_B_embed, "b (h w) c -> b c h w", h=h, w=w)
        
        layer4_A=self.re_embedA(layer4_A_embed)
        layer4_B=self.re_embedB(layer4_B_embed)
        # print(f"Mamba_Re_Embed: {layer4_A.shape}")
        
        fus = self.fe([layer1_A, layer2_A, layer3_A, layer4_A], [layer1_B, layer2_B, layer3_B,layer4_B])
        fus = torch.sum(fus, dim=0)
        
        # print(f"Fus_Out: {fus.shape}")
        
        y = self.classifier(fus)
        
        # print(f"Final_Out: {y.shape}")
        
        return y
    