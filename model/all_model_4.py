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

# ===============================
# 状态字典生成工具函数
# ===============================
def gen_state_dict(weights_path):
    st = torch.load(weights_path)
    st_ks = list(st.keys())
    st_vs = list(st.values())
    state_dict = {}
    for st_k, st_v in zip(st_ks, st_vs):
        state_dict[st_k.replace('module.', '')] = st_v
    return state_dict


# ===============================
# Segment consensus 模块
# ===============================
class ConsensusModule(torch.nn.Module):
    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim

    def forward(self, input):
        return SegmentConsensus(self.consensus_type, self.dim)(input)


class SegmentConsensus(torch.nn.Module):
    def __init__(self, consensus_type, dim=1):
        super(SegmentConsensus, self).__init__()
        self.consensus_type = consensus_type
        self.dim = dim
        self.shape = None

    def forward(self, input_tensor):
        self.shape = input_tensor.size()
        if self.consensus_type == 'avg':
            output = input_tensor.mean(dim=self.dim, keepdim=True)
        elif self.consensus_type == 'identity':
            output = input_tensor
        else:
            output = None
        return output


# ===============================
# Temporal Shift 模块
# ===============================
class TemporalShift(nn.Module):
    def __init__(self, net, n_segment=3, n_div=8, inplace=False):
        super(TemporalShift, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace

    def forward(self, x):
        x = self.shift(x, self.n_segment, fold_div=self.fold_div, inplace=self.inplace)
        return self.net(x)

    @staticmethod
    def shift(x, n_segment, fold_div=3, inplace=False):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)
        fold = c // fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift
        return out.view(nt, c, h, w)


# ===============================
# CBAM 模块（ECA + SA）
# ===============================
class eca_layer_2d_v2(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        t = int(abs(math.log(channel, 2) + 1) / 2)
        k_size = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y_avg = self.avg_pool(x)
        y_max = self.max_pool(x)
        y_avg = self.conv(y_avg.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y_max = self.conv(y_max.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y_avg + y_max)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_out = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(x_out))


class CBAMBlock(nn.Module):
    def __init__(self, channels, kernel_size=7):
        super().__init__()
        self.channel_att = eca_layer_2d_v2(channels)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


# ===============================
# Temporal Attention
# ===============================
class TemporalAttention(nn.Module):
    def __init__(self, feature_dim, n_head=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=n_head, batch_first=True)

    def forward(self, x):
        x_attn, _ = self.attn(x, x, x)
        return x_attn.mean(dim=1)


# ===============================
# SKD_TSTSAN 完整可训练版本
# ===============================
class SKD_TSTSAN(nn.Module):
    def __init__(self, out_channels=5, amp_factor=5):
        super().__init__()
        # Encoder & Manipulator
        self.Aug_Encoder_L = MagEncoder_No_texture(dim_in=16)
        self.Aug_Encoder_S = MagEncoder_No_texture(dim_in=1)
        self.Aug_Encoder_T = MagEncoder_No_texture(dim_in=2)
        self.Aug_Manipulator_L = MagManipulator()
        self.Aug_Manipulator_S = MagManipulator()
        self.Aug_Manipulator_T = MagManipulator()
        self.amp_factor = amp_factor

        # ----------------------
        # 卷积阶段
        # ----------------------
        self.conv1_L = nn.Conv2d(32, 64, 5, 1)
        self.conv1_S = nn.Conv2d(32, 64, 5, 1)
        self.conv1_T = nn.Conv2d(32, 64, 5, 1)
        self.bn1_L = nn.BatchNorm2d(64)
        self.bn1_S = nn.BatchNorm2d(64)
        self.bn1_T = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(5, 2, 2)

        # AC1
        self.AC1_conv1_L = nn.Conv2d(64, 128, 3, 1, 1)
        self.AC1_conv1_S = nn.Conv2d(64, 128, 3, 1, 1)
        self.AC1_conv1_T = TemporalShift(nn.Conv2d(64, 128, 3, 1, 1), n_segment=2)
        self.AC1_bn1_L = nn.BatchNorm2d(128)
        self.AC1_bn1_S = nn.BatchNorm2d(128)
        self.AC1_bn1_T = nn.BatchNorm2d(128)
        self.AC1_conv2_L = nn.Conv2d(128, 128, 3, 1, 1)
        self.AC1_conv2_S = nn.Conv2d(128, 128, 3, 1, 1)
        self.AC1_conv2_T = TemporalShift(nn.Conv2d(128, 128, 3, 1, 1), n_segment=2)
        self.AC1_bn2_L = nn.BatchNorm2d(128)
        self.AC1_bn2_S = nn.BatchNorm2d(128)
        self.AC1_bn2_T = nn.BatchNorm2d(128)

        # CBAM
        self.CBAM1_L = CBAMBlock(64)
        self.CBAM1_S = CBAMBlock(64)
        self.CBAM1_T = CBAMBlock(64)
        self.CBAM2_L = CBAMBlock(128)
        self.CBAM2_S = CBAMBlock(128)
        self.CBAM2_T = CBAMBlock(128)

        # AC2
        self.conv2_L = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv2_S = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv2_T = TemporalShift(nn.Conv2d(64, 64, 3, 1, 1), n_segment=2)
        self.bn2_L = nn.BatchNorm2d(64)
        self.bn2_S = nn.BatchNorm2d(64)
        self.bn2_T = nn.BatchNorm2d(64)

        self.conv3_L = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv3_S = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv3_T = TemporalShift(nn.Conv2d(64, 64, 3, 1, 1), n_segment=2)
        self.bn3_L = nn.BatchNorm2d(64)
        self.bn3_S = nn.BatchNorm2d(64)
        self.bn3_T = nn.BatchNorm2d(64)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.4)

        # 分类头
        self.fc_AC1 = nn.Sequential(
            nn.Linear(128 * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, out_channels)
        )
        self.fc_final = nn.Sequential(
            nn.Linear(64 * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, out_channels)
        )

        # Consensus
        self.consensus = ConsensusModule("avg")

    # ===============================
    # forward
    # ===============================
    def forward(self, input):
        # input 分段处理
        x1 = input[:, 2:18, :, :]
        x1_onset = input[:, 18:34, :, :]
        x2 = input[:, 0, :, :].unsqueeze(1)
        x2_onset = input[:, 1, :, :].unsqueeze(1)
        x3 = input[:, 34:, :, :]
        bsz = x1.size(0)
        x3 = x3.view(bsz * 2, 2, 48, 48)
        x3_onset = torch.zeros_like(x3)

        # Motion Augmentation
        motion_x1_onset = self.Aug_Encoder_L(x1_onset)
        motion_x1 = self.Aug_Encoder_L(x1)
        x1 = self.Aug_Manipulator_L(motion_x1_onset, motion_x1, self.amp_factor)

        motion_x2_onset = self.Aug_Encoder_S(x2_onset)
        motion_x2 = self.Aug_Encoder_S(x2)
        x2 = self.Aug_Manipulator_S(motion_x2_onset, motion_x2, self.amp_factor)

        motion_x3_onset = self.Aug_Encoder_T(x3_onset)
        motion_x3 = self.Aug_Encoder_T(x3)
        x3 = self.Aug_Manipulator_T(motion_x3_onset, motion_x3, self.amp_factor)

        # conv1 + CBAM1
        x1 = self.conv1_L(x1)
        x1 = self.bn1_L(x1)
        x1 = self.relu(x1)
        x1 = self.CBAM1_L(x1)
        x1 = self.maxpool(x1)
        x2 = self.conv1_S(x2)
        x2 = self.bn1_S(x2)
        x2 = self.relu(x2)
        x2 = self.CBAM1_S(x2)
        x2 = self.maxpool(x2)
        x3 = self.conv1_T(x3)
        x3 = self.bn1_T(x3)
        x3 = self.relu(x3)
        x3 = self.CBAM1_T(x3)
        x3 = self.maxpool(x3)

        # AC1
        AC1_x1 = self.AC1_conv1_L(x1)
        AC1_x1 = self.AC1_bn1_L(AC1_x1)
        AC1_x1 = self.relu(AC1_x1)
        AC1_x1 = self.CBAM2_L(AC1_x1)
        AC1_x1 = self.AC1_conv2_L(AC1_x1)
        AC1_x1 = self.AC1_bn2_L(AC1_x1)
        AC1_x1 = self.relu(AC1_x1)
        AC1_x1_pool = self.avgpool(AC1_x1)
        AC1_x1_all = AC1_x1_pool.view(AC1_x1_pool.size(0), -1)

        AC1_x2 = self.AC1_conv1_S(x2)
        AC1_x2 = self.AC1_bn1_S(AC1_x2)
        AC1_x2 = self.relu(AC1_x2)
        AC1_x2 = self.CBAM2_S(AC1_x2)
        AC1_x2 = self.AC1_conv2_S(AC1_x2)
        AC1_x2 = self.AC1_bn2_S(AC1_x2)
        AC1_x2 = self.relu(AC1_x2)
        AC1_x2_pool = self.avgpool(AC1_x2)
        AC1_x2_all = AC1_x2_pool.view(AC1_x2_pool.size(0), -1)

        AC1_x3 = self.AC1_conv1_T(x3)
        AC1_x3 = self.AC1_bn1_T(AC1_x3)
        AC1_x3 = self.relu(AC1_x3)
        AC1_x3 = self.CBAM2_T(AC1_x3)
        AC1_x3 = self.AC1_conv2_T(AC1_x3)
        AC1_x3 = self.AC1_bn2_T(AC1_x3)
        AC1_x3 = self.relu(AC1_x3)
        AC1_x3_pool = self.avgpool(AC1_x3)
        AC1_x3_all = AC1_x3_pool.view(bsz, -1)

        AC1_feature = torch.cat((AC1_x1_all, AC1_x2_all, AC1_x3_all), 1)
        AC1_out = self.fc_AC1(self.dropout(AC1_feature))

        # AC2 (conv2 + conv3)
        x1 = self.conv2_L(x1)
        x1 = self.bn2_L(x1)
        x1 = self.relu(x1)
        x1 = self.conv3_L(x1)
        x1 = self.bn3_L(x1)
        x1 = self.relu(x1)
        x2 = self.conv2_S(x2)
        x2 = self.bn2_S(x2)
        x2 = self.relu(x2)
        x2 = self.conv3_S(x2)
        x2 = self.bn3_S(x2)
        x2 = self.relu(x2)
        x3 = self.conv2_T(x3)
        x3 = self.bn2_T(x3)
        x3 = self.relu(x3)
        x3 = self.conv3_T(x3)
        x3 = self.bn3_T(x3)
        x3 = self.relu(x3)

        # 特征融合 & 分类
        x1_pool = self.avgpool(x1).view(bsz, -1)
        x2_pool = self.avgpool(x2).view(bsz, -1)
        x3_pool = self.avgpool(x3).view(bsz, -1)
        final_feature = torch.cat((x1_pool, x2_pool, x3_pool), 1)
        final_out = self.fc_final(self.dropout(final_feature))

        return final_out, AC1_out, final_feature


# ===============================
# 获取模型函数
# ===============================
def get_model(model_name, class_num, alpha):
    if model_name == "SKD_TSTSAN":
        return SKD_TSTSAN(class_num, alpha)
