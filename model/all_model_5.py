import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

"""
修正版 SKD_TSTSAN_v4
- 自动计算 T 分支 n_segment，避免 reshape 报错
- 保留 all_model_5 的各项改动（共享 ECA/SA、TemporalShift、TemporalAttention）
- 支持任意输入 H,W 和任意通道数
"""

def gen_state_dict(weights_path):
    st = torch.load(weights_path, map_location='cpu')
    state_dict = {}
    for k, v in st.items():
        state_dict[k.replace('module.', '')] = v
    return state_dict


class ConsensusModule(nn.Module):
    def __init__(self, consensus_type: str, dim: int = 1):
        super().__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim

    def forward(self, x):
        if self.consensus_type == 'avg':
            return x.mean(dim=self.dim, keepdim=True)
        elif self.consensus_type == 'identity':
            return x
        else:
            raise ValueError(f"Unsupported consensus type: {self.consensus_type}")


class TemporalShift(nn.Module):
    def __init__(self, net: Optional[nn.Module] = None, n_segment: int = 3, n_div: int = 8, inplace: bool = False):
        super().__init__()
        self.net = net if net is not None else nn.Identity()
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace

    def forward(self, x):
        x = self.shift(x, self.n_segment, fold_div=self.fold_div, inplace=self.inplace)
        return self.net(x)

    @staticmethod
    def shift(x: torch.Tensor, n_segment: int, fold_div: int = 8, inplace: bool = False) -> torch.Tensor:
        nt, c, h, w = x.shape
        if n_segment <= 1:
            return x
        n_batch = nt // n_segment
        x_view = x.view(n_batch, n_segment, c, h, w)

        fold = max(1, c // fold_div)
        out = x_view.clone()
        out[:, :-1, :fold] = x_view[:, 1:, :fold]
        out[:, 1:, fold:2 * fold] = x_view[:, :-1, fold:2 * fold]
        return out.view(nt, c, h, w)


class ECALayer2D(nn.Module):
    def __init__(self, channel: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        t = max(1, int(abs(math.log(max(2, channel), 2) + 1) / 2))
        k_size = t if (t % 2 == 1) else (t + 1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y_avg = self.avg_pool(x)
        y_max = self.max_pool(x)
        y_avg = y_avg.squeeze(-1).transpose(-1, -2)
        y_max = y_max.squeeze(-1).transpose(-1, -2)
        y_avg = self.conv(y_avg).transpose(-1, -2).unsqueeze(-1)
        y_max = self.conv(y_max).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y_avg + y_max)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_out = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(x_out))
        return x * attn


class TemporalAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(4, channels // max(1, reduction))
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        self.act = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor, n_segment: int = 2) -> torch.Tensor:
        bt, c, h, w = x.size()
        if n_segment <= 1:
            return x.view(bt // n_segment, c, h, w)
        b = bt // n_segment
        x = x.view(b, n_segment, c, h, w)
        desc = x.mean(dim=[3, 4])
        logits = self.fc2(self.act(self.fc1(desc)))
        weights = self.softmax(logits.squeeze(-1)).view(b, n_segment, 1, 1, 1)
        out = (x * weights).sum(dim=1)
        return out


class Bottleneck(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, mid_ch: Optional[int] = None,
                 use_eca: bool = False, use_sa: bool = False):
        super().__init__()
        if mid_ch is None:
            mid_ch = max(1, out_ch // 2)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.down = None
        if in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        self.use_eca = use_eca
        self.use_sa = use_sa
        if use_eca:
            self.eca = ECALayer2D(out_ch)
        if use_sa:
            self.sa = SpatialAttention()

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.use_eca:
            out = self.eca(out)
        if self.use_sa:
            out = self.sa(out)
        if self.down is not None:
            identity = self.down(identity)
        out = self.relu(out + identity)
        return out


class ClassifierHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, p: float = 0.4):
        super().__init__()
        mid = max(4, in_dim // 2)
        self.head = nn.Sequential(
            nn.Linear(in_dim, mid),
            nn.LayerNorm(mid),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(mid, num_classes)
        )

    def forward(self, x):
        return self.head(x)


def make_bottleneck_stage(in_ch: int, out_ch: int, num_blocks: int = 2,
                          use_eca: bool = False, use_sa: bool = False) -> nn.Sequential:
    layers = [Bottleneck(in_ch, out_ch, use_eca=use_eca, use_sa=use_sa)]
    for _ in range(1, num_blocks):
        layers.append(Bottleneck(out_ch, out_ch, use_eca=use_eca, use_sa=use_sa))
    return nn.Sequential(*layers)


def make_temporal_stage(in_ch: int, out_ch: int, num_blocks: int = 2, n_segment: int = 2, n_div: int = 8,
                        use_sa: bool = False):
    layers = [TemporalShift(Bottleneck(in_ch, out_ch, use_sa=use_sa), n_segment=n_segment, n_div=n_div)]
    for _ in range(1, num_blocks):
        layers.append(TemporalShift(Bottleneck(out_ch, out_ch, use_sa=use_sa), n_segment=n_segment, n_div=n_div))
    return nn.Sequential(*layers)


class SKD_TSTSAN_v4(nn.Module):
    def __init__(self, num_classes: int = 5, amp_factor: int = 5, n_segment_t: Optional[int] = None):
        super().__init__()
        from motion_magnification_learning_based_master.magnet import Manipulator as MagManipulator, \
            Encoder_No_texture as MagEncoder_No_texture

        # T 分支 n_segment 允许自动推断
        self.n_segment_t = n_segment_t

        # 保留 Aug_Encoder_L/S
        self.Aug_Encoder_L = MagEncoder_No_texture(dim_in=16)
        self.Aug_Encoder_S = MagEncoder_No_texture(dim_in=1)

        # T 分支 dim_in 先用 1 占位，forward 时根据输入自动 reshape
        self.Aug_Encoder_T_in_ch = 2
        self.Aug_Encoder_T = MagEncoder_No_texture(dim_in=self.Aug_Encoder_T_in_ch)

        self.Aug_Manipulator_L = MagManipulator()
        self.Aug_Manipulator_S = MagManipulator()
        self.Aug_Manipulator_T = MagManipulator()

        self.stem_L = nn.Sequential(nn.Conv2d(16, 64, 5, stride=1, padding=2, bias=False),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.stem_S = nn.Sequential(nn.Conv2d(1, 64, 5, stride=1, padding=2, bias=False),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.stem_T = nn.Sequential(nn.Conv2d(self.Aug_Encoder_T_in_ch, 64, 5, stride=1, padding=2, bias=False),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.pool5 = nn.MaxPool2d(kernel_size=5, stride=2, padding=2)

        self.shared_eca_64 = ECALayer2D(64)
        self.shared_sa = SpatialAttention()

        # AC1
        self.ac1_L = make_bottleneck_stage(64, 128, num_blocks=2, use_eca=True, use_sa=True)
        self.ac1_S = make_bottleneck_stage(64, 128, num_blocks=2, use_sa=True)
        self.ac1_T = make_temporal_stage(64, 128, num_blocks=2, n_segment=2, use_sa=True)
        self.ac1_pool = nn.AdaptiveAvgPool2d(1)
        self.ta_128 = TemporalAttention(128)
        self.ac1_head = ClassifierHead(128 * 3, num_classes)

        # 中间层
        self.mid_L = nn.Sequential(Bottleneck(64, 64, use_eca=True), Bottleneck(64, 64, use_eca=True),
                                   nn.AvgPool2d(3, stride=2, padding=1))
        self.mid_S = nn.Sequential(Bottleneck(64, 64), Bottleneck(64, 64), nn.AvgPool2d(3, stride=2, padding=1))
        self.mid_T = nn.Sequential(TemporalShift(Bottleneck(64, 64), n_segment=2),
                                   TemporalShift(Bottleneck(64, 64), n_segment=2),
                                   nn.AvgPool2d(3, stride=2, padding=1))

        # AC2
        self.ac2_L = make_bottleneck_stage(64, 128, num_blocks=2, use_eca=True)
        self.ac2_S = make_bottleneck_stage(64, 128, num_blocks=2)
        self.ac2_T = make_temporal_stage(64, 128, num_blocks=2, n_segment=2)
        self.ac2_pool = nn.AdaptiveAvgPool2d(1)
        self.ta_128_b = TemporalAttention(128)
        self.ac2_head = ClassifierHead(128 * 3, num_classes)

        # final
        self.final_L = make_bottleneck_stage(64, 128, num_blocks=2, use_eca=True)
        self.final_S = make_bottleneck_stage(64, 128, num_blocks=2)
        self.final_T = make_temporal_stage(64, 128, num_blocks=2, n_segment=2)
        self.final_pool = nn.AdaptiveAvgPool2d(1)
        self.ta_128_c = TemporalAttention(128)
        self.final_head = ClassifierHead(128 * 3, num_classes)

        self.consensus = ConsensusModule('avg')
        self.amp_factor = amp_factor

        self._init_weights()

    @staticmethod
    def _global_pool_flat(x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=[2, 3])

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                if getattr(m, 'weight', None) is not None:
                    nn.init.ones_(m.weight)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, input: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        # L/S/T 分支切片
        x1 = input[:, 2:18, :, :]
        x1_onset = input[:, 18:34, :, :]
        x2 = input[:, 0, :, :].unsqueeze(1)
        x2_onset = input[:, 1, :, :].unsqueeze(1)
        x3 = input[:, 34:, :, :]

        bsz, c3, h, w = x3.shape

        # 自动推断 n_segment_t
        if self.n_segment_t is None:
            for seg in reversed(range(1, c3 + 1)):
                if c3 % seg == 0:
                    self.n_segment_t = seg
                    break

        per_seg_ch = c3 // self.n_segment_t
        self.Aug_Encoder_T_in_ch = per_seg_ch  # 更新 encoder 输入通道
        x3 = x3.reshape(bsz * self.n_segment_t, per_seg_ch, h, w)
        x3_onset = torch.zeros_like(x3)

        # Augmentation
        motion_x1_onset = self.Aug_Encoder_L(x1_onset)
        motion_x1 = self.Aug_Encoder_L(x1)
        x1 = self.Aug_Manipulator_L(motion_x1_onset, motion_x1, self.amp_factor)

        motion_x2_onset = self.Aug_Encoder_S(x2_onset)
        motion_x2 = self.Aug_Encoder_S(x2)
        x2 = self.Aug_Manipulator_S(motion_x2_onset, motion_x2, self.amp_factor)

        motion_x3_onset = self.Aug_Encoder_T(x3_onset)
        motion_x3 = self.Aug_Encoder_T(x3)
        x3 = self.Aug_Manipulator_T(motion_x3_onset, motion_x3, self.amp_factor)

        # stem + shared attention + pool
        x1 = self.stem_L(x1)
        x1 = self.shared_eca_64(x1)
        x1 = self.shared_sa(x1)
        x1 = self.pool5(x1)

        x2 = self.stem_S(x2)
        x2 = self.shared_sa(x2)
        x2 = self.pool5(x2)

        x3 = self.stem_T(x3)
        x3 = self.shared_sa(x3)
        x3 = self.pool5(x3)

        # AC1
        ac1_x1 = self.ac1_L(x1)
        ac1_x2 = self.ac1_S(x2)
        ac1_x3 = self.ac1_T(x3)
        ac1_x3 = self.ta_128(ac1_x3, n_segment=self.n_segment_t)
        ac1_feat = torch.cat([self._global_pool_flat(self.ac1_pool(ac1_x1)),
                              self._global_pool_flat(self.ac1_pool(ac1_x2)),
                              self._global_pool_flat(self.ac1_pool(ac1_x3))], dim=1)
        ac1_logits = self.ac1_head(ac1_feat)

        # 中间层
        x1m = self.mid_L(x1)
        x2m = self.mid_S(x2)
        x3m = self.mid_T(x3)

        # AC2
        ac2_x1 = self.ac2_L(x1m)
        ac2_x2 = self.ac2_S(x2m)
        ac2_x3 = self.ac2_T(x3m)
        ac2_x3 = self.ta_128_b(ac2_x3, n_segment=self.n_segment_t)
        ac2_feat = torch.cat([self._global_pool_flat(self.ac2_pool(ac2_x1)),
                              self._global_pool_flat(self.ac2_pool(ac2_x2)),
                              self._global_pool_flat(self.ac2_pool(ac2_x3))], dim=1)
        ac2_logits = self.ac2_head(ac2_feat)

        # final
        f1 = self.final_L(x1m)
        f2 = self.final_S(x2m)
        f3 = self.final_T(x3m)
        f3 = self.ta_128_c(f3, n_segment=self.n_segment_t)
        final_feat = torch.cat([self._global_pool_flat(self.final_pool(f1)),
                                self._global_pool_flat(self.final_pool(f2)),
                                self._global_pool_flat(self.final_pool(f3))], dim=1)
        final_logits = self.final_head(final_feat)

        return final_logits, ac1_logits, ac2_logits, final_feat, ac1_feat, ac2_feat


def get_model(model_name: str, class_num: int, alpha: int):
    if model_name in ["SKD_TSTSAN", "SKD_TSTSAN_v4"]:
        return SKD_TSTSAN_v4(class_num, alpha)
    raise ValueError(f"Unknown model name: {model_name}")
