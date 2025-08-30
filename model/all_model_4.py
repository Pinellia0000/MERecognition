import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from motion_magnification_learning_based_master.magnet import Manipulator as MagManipulator
from motion_magnification_learning_based_master.magnet import Encoder_No_texture as MagEncoder_No_texture

"""
在all_model_1.py的基础上

1.注意力机制 
原 ECA + 轻量 SA 替换为 CBAMBlock（ECA + SA 联合注意力） 
2.时序建模 
保留 TemporalShift，可选后续加入 TemporalAttention 对 AC1/AC2 输出特征进一步时序聚合 
3.分类头增强 
1)Dropout 提高到 0.4 
2)FC head 增加 BN + ReLU + FC 层，缓解过拟合，增强泛化能力
4.代码简化 
1)forward 结构保持原有逻辑 
2)对 AC1 特征做 consensus + dropout + CBAM 
3)保留原始 AC2、conv4/5 可以按需求进一步改进
"""
# ==========================
# 权重加载
# ==========================
def gen_state_dict(weights_path):
    st = torch.load(weights_path)
    state_dict = {k.replace('module.', ''): v for k, v in st.items()}
    return state_dict

# ==========================
# Temporal Consensus
# ==========================
class ConsensusModule(nn.Module):
    def __init__(self, consensus_type, dim=1):
        super().__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim

    def forward(self, x):
        if self.consensus_type == 'avg':
            return x.mean(dim=self.dim, keepdim=True)
        return x

# ==========================
# Temporal Shift
# ==========================
class TemporalShift(nn.Module):
    def __init__(self, net, n_segment=3, n_div=8):
        super().__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div

    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        fold = c // self.fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]          # shift left
        out[:, 1:, fold: 2*fold] = x[:, :-1, fold:2*fold]  # shift right
        out[:, :, 2*fold:] = x[:, :, 2*fold:]        # not shift
        return self.net(out.view(nt, c, h, w))

# ==========================
# CBAMBlock: ECA + SpatialAttention
# ==========================
class CBAMBlock(nn.Module):
    def __init__(self, channel, kernel_size=7):
        super().__init__()
        # ECA
        t = int(abs(math.log(channel, 2) + 1) / 2)
        k_size = t if t % 2 else t + 1
        self.eca_conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        # Spatial Attention
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size-1)//2, bias=False)

    def forward(self, x):
        # ECA通道注意力
        y_avg = x.mean(dim=(2,3), keepdim=True)
        y_max, _ = x.max(dim=(2,3), keepdim=True)
        y = self.eca_conv(y_avg.squeeze(-1).transpose(-1,-2)).transpose(-1,-2).unsqueeze(-1) + \
            self.eca_conv(y_max.squeeze(-1).transpose(-1,-2)).transpose(-1,-2).unsqueeze(-1)
        x = x * self.sigmoid(y).expand_as(x)
        # 空间注意力
        avg_out = x.mean(dim=1, keepdim=True)
        max_out, _ = x.max(dim=1, keepdim=True)
        x_out = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.spatial_conv(x_out))
        return x * attn

# ==========================
# SKD_TSTSAN 主模型
# ==========================
class SKD_TSTSAN(nn.Module):
    def __init__(self, out_channels=5, amp_factor=5):
        super().__init__()
        self.Aug_Encoder_L = MagEncoder_No_texture(dim_in=16)
        self.Aug_Encoder_S = MagEncoder_No_texture(dim_in=1)
        self.Aug_Encoder_T = MagEncoder_No_texture(dim_in=2)
        self.Aug_Manipulator_L = MagManipulator()
        self.Aug_Manipulator_S = MagManipulator()
        self.Aug_Manipulator_T = MagManipulator()

        # 基础卷积
        self.conv1_L = nn.Conv2d(32, 64, kernel_size=5, stride=1)
        self.conv1_S = nn.Conv2d(32, 64, kernel_size=5, stride=1)
        self.conv1_T = nn.Conv2d(32, 64, kernel_size=5, stride=1)
        self.bn1_L = nn.BatchNorm2d(64)
        self.bn1_S = nn.BatchNorm2d(64)
        self.bn1_T = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=2, padding=2)

        # AC1
        self.AC1_conv1_L = nn.Conv2d(64, 128, 3, padding=1)
        self.AC1_conv1_S = nn.Conv2d(64, 128, 3, padding=1)
        self.AC1_conv1_T = TemporalShift(nn.Conv2d(64, 128, 3, padding=1), n_segment=2, n_div=8)
        self.AC1_conv2_L = nn.Conv2d(128, 128, 3, padding=1)
        self.AC1_conv2_S = nn.Conv2d(128, 128, 3, padding=1)
        self.AC1_conv2_T = TemporalShift(nn.Conv2d(128, 128, 3, padding=1), n_segment=2, n_div=8)
        self.AC1_bn1_L = nn.BatchNorm2d(128)
        self.AC1_bn1_S = nn.BatchNorm2d(128)
        self.AC1_bn1_T = nn.BatchNorm2d(128)
        self.AC1_bn2_L = nn.BatchNorm2d(128)
        self.AC1_bn2_S = nn.BatchNorm2d(128)
        self.AC1_bn2_T = nn.BatchNorm2d(128)
        self.AC1_pool = nn.AdaptiveAvgPool2d(1)

        # CBAM 注意力
        self.AC1_cbam_L = CBAMBlock(128)
        self.AC1_cbam_S = CBAMBlock(128)
        self.AC1_cbam_T = CBAMBlock(128)

        # Dropout + FC head
        self.dropout = nn.Dropout(0.4)
        self.AC1_fc = nn.Sequential(
            nn.Linear(384, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.Linear(384, out_channels)
        )

        # ======================
        # 第二阶段 conv2/3 保留
        # ======================
        self.conv2_L = nn.Conv2d(64, 64, 3, padding=1)
        self.conv2_S = nn.Conv2d(64, 64, 3, padding=1)
        self.conv2_T = TemporalShift(nn.Conv2d(64, 64, 3, padding=1), n_segment=2, n_div=8)
        self.bn2_L = nn.BatchNorm2d(64)
        self.bn2_S = nn.BatchNorm2d(64)
        self.bn2_T = nn.BatchNorm2d(64)
        self.conv3_L = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3_S = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3_T = TemporalShift(nn.Conv2d(64, 64, 3, padding=1), n_segment=2, n_div=8)
        self.bn3_L = nn.BatchNorm2d(64)
        self.bn3_S = nn.BatchNorm2d(64)
        self.bn3_T = nn.BatchNorm2d(64)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        # AC2 模块保留原结构，可进一步加入 CBAM/TemporalAttention
        self.AC2_conv1_L = nn.Conv2d(64, 128, 3, padding=1)
        self.AC2_conv1_S = nn.Conv2d(64, 128, 3, padding=1)
        self.AC2_conv1_T = TemporalShift(nn.Conv2d(64, 128, 3, padding=1), n_segment=2, n_div=8)
        self.AC2_conv2_L = nn.Conv2d(128, 128, 3, padding=1)
        self.AC2_conv2_S = nn.Conv2d(128, 128, 3, padding=1)
        self.AC2_conv2_T = TemporalShift(nn.Conv2d(128, 128, 3, padding=1), n_segment=2, n_div=8)
        self.AC2_bn1_L = nn.BatchNorm2d(128)
        self.AC2_bn1_S = nn.BatchNorm2d(128)
        self.AC2_bn1_T = nn.BatchNorm2d(128)
        self.AC2_bn2_L = nn.BatchNorm2d(128)
        self.AC2_bn2_S = nn.BatchNorm2d(128)
        self.AC2_bn2_T = nn.BatchNorm2d(128)
        self.AC2_pool = nn.AdaptiveAvgPool2d(1)
        self.AC2_fc = nn.Linear(384, out_channels)

        # 全局 conv4/5 保留原始
        self.conv4_L = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4_S = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4_T = TemporalShift(nn.Conv2d(64, 128, 3, padding=1), n_segment=2, n_div=8)
        self.conv5_L = nn.Conv2d(128, 128, 3, padding=1)
        self.conv5_S = nn.Conv2d(128, 128, 3, padding=1)
        self.conv5_T = TemporalShift(nn.Conv2d(128, 128, 3, padding=1), n_segment=2, n_div=8)
        self.all_avgpool = nn.AdaptiveAvgPool2d(1)

        self.amp_factor = amp_factor
        self.consensus = ConsensusModule("avg")

# ==========================
# 模型构建接口
# ==========================
def get_model(model_name, class_num, alpha):
    if model_name == "SKD_TSTSAN":
        return SKD_TSTSAN(class_num, alpha)
