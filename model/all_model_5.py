import torch
import torch.nn as nn
import math
from typing import List, Optional, Tuple

"""

在all_model_4.py的基础上

1.删除硬编码尺寸：原版本 48x48 固定，会导致在不同数据预处理或裁剪下出错。改为 x3.shape[-2:] 或 torch.zeros_like(x3) 保证对任意输入形状更鲁棒。

2.统一 stage 构造：make_bottleneck_stage / make_temporal_stage 减少重复、便于后续更改 block 数量或通道数（例如做轻量/重型配置）。

3.TemporalShift 更稳健：原来用 zeros_like(x) 并手工赋值容易出现类型/grad 的细节 bug，改为 clone() 并处理 fold 最小为 1，避免 fold=0 情况。

4.共享 attention：你已经提出 stage 内共享 ECA/SA，这里保留 self.shared_eca_64/self.shared_sa，避免重复实例化和参数冗余。

5.初始化覆盖更全：对 LayerNorm 也做了初始化；torch.load(..., map_location='cpu') 避免在无 GPU 环境下加载报错。

6.TemporalAttention 更通用：支持任意 n_segment，便于以后对不同时间窗口做试验（例如 2、3、4 段）。

7.保持接口兼容：get_model 与返回值顺序均保留，方便替换到现有训练/评估 pipeline。

修改版 SKD_TSTSAN_v2

改动要点：
1. x3 T 分支 reshape 自动适配 n_segment，增加 assert 防止通道不匹配。
2. 保留共享 ECA/SA，TemporalShift，TemporalAttention。
3. Stem + Bottleneck + Head 接口保持不变。
4. 适用于任意 H,W 输入。
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


class SegmentConsensus(ConsensusModule):
    pass


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
    layers = []
    layers.append(Bottleneck(in_ch, out_ch, use_eca=use_eca, use_sa=use_sa))
    for _ in range(1, num_blocks):
        layers.append(Bottleneck(out_ch, out_ch, use_eca=use_eca, use_sa=use_sa))
    return nn.Sequential(*layers)


def make_temporal_stage(in_ch: int, out_ch: int, num_blocks: int = 2, n_segment: int = 2, n_div: int = 8,
                        use_sa: bool = False):
    layers = []
    layers.append(TemporalShift(Bottleneck(in_ch, out_ch, use_sa=use_sa), n_segment=n_segment, n_div=n_div))
    for _ in range(1, num_blocks):
        layers.append(TemporalShift(Bottleneck(out_ch, out_ch, use_sa=use_sa), n_segment=n_segment, n_div=n_div))
    return nn.Sequential(*layers)


class SKD_TSTSAN_v2(nn.Module):
    def __init__(self, num_classes: int = 5, amp_factor: int = 5, n_segment_t: int = 2):
        super().__init__()
        from motion_magnification_learning_based_master.magnet import (
            Manipulator as MagManipulator,
            Encoder_No_texture as MagEncoder_No_texture,
        )
        self.Aug_Encoder_L = MagEncoder_No_texture(dim_in=16)
        self.Aug_Encoder_S = MagEncoder_No_texture(dim_in=1)
        self.Aug_Encoder_T = MagEncoder_No_texture(dim_in=2)
        self.Aug_Manipulator_L = MagManipulator()
        self.Aug_Manipulator_S = MagManipulator()
        self.Aug_Manipulator_T = MagManipulator()

        # stem
        self.stem_L = nn.Sequential(nn.Conv2d(16, 64, 5, stride=1, padding=2, bias=False),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.stem_S = nn.Sequential(nn.Conv2d(1, 64, 5, stride=1, padding=2, bias=False),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.stem_T = nn.Sequential(nn.Conv2d(2, 64, 5, stride=1, padding=2, bias=False),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.pool5 = nn.MaxPool2d(kernel_size=5, stride=2, padding=2)

        # 共享注意力
        self.shared_eca_64 = ECALayer2D(64)
        self.shared_sa = SpatialAttention()

        # AC1
        self.ac1_L = make_bottleneck_stage(64, 128, num_blocks=2, use_eca=True, use_sa=True)
        self.ac1_S = make_bottleneck_stage(64, 128, num_blocks=2, use_sa=True)
        self.ac1_T = make_temporal_stage(64, 128, num_blocks=2, n_segment=n_segment_t, use_sa=True)
        self.ac1_pool = nn.AdaptiveAvgPool2d(1)
        self.ta_128 = TemporalAttention(128, reduction=8)
        self.ac1_head = ClassifierHead(128 * 3, num_classes, p=0.4)

        # 中间层
        self.mid_L = nn.Sequential(Bottleneck(64, 64, use_eca=True),
                                   Bottleneck(64, 64, use_eca=True),
                                   nn.AvgPool2d(kernel_size=3, stride=2, padding=1))
        self.mid_S = nn.Sequential(Bottleneck(64, 64),
                                   Bottleneck(64, 64),
                                   nn.AvgPool2d(kernel_size=3, stride=2, padding=1))
        self.mid_T = nn.Sequential(TemporalShift(Bottleneck(64, 64), n_segment=n_segment_t),
                                   TemporalShift(Bottleneck(64, 64), n_segment=n_segment_t),
                                   nn.AvgPool2d(kernel_size=3, stride=2, padding=1))

        # AC2
        self.ac2_L = make_bottleneck_stage(64, 128, num_blocks=2, use_eca=True)
        self.ac2_S = make_bottleneck_stage(64, 128, num_blocks=2)
        self.ac2_T = make_temporal_stage(64, 128, num_blocks=2, n_segment=n_segment_t)
        self.ac2_pool = nn.AdaptiveAvgPool2d(1)
        self.ta_128_b = TemporalAttention(128)
        self.ac2_head = ClassifierHead(128 * 3, num_classes, p=0.4)

        # final
        self.final_L = make_bottleneck_stage(64, 128, num_blocks=2, use_eca=True)
        self.final_S = make_bottleneck_stage(64, 128, num_blocks=2)
        self.final_T = make_temporal_stage(64, 128, num_blocks=2, n_segment=n_segment_t)
        self.final_pool = nn.AdaptiveAvgPool2d(1)
        self.ta_128_c = TemporalAttention(128)
        self.final_head = ClassifierHead(128 * 3, num_classes, p=0.4)

        self.consensus = ConsensusModule('avg')
        self.amp_factor = amp_factor
        self.n_segment_t = n_segment_t

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
        x1 = input[:, 2:18, :, :]
        x1_onset = input[:, 18:34, :, :]
        x2 = input[:, 0, :, :].unsqueeze(1)
        x2_onset = input[:, 1, :, :].unsqueeze(1)
        x3 = input[:, 34:, :, :]

        bsz = x1.size(0)

        # ---------- T 分支 reshape ----------
        c3 = x3.shape[1]
        assert c3 % self.n_segment_t == 0, f"x3 channels {c3} not divisible by n_segment_t={self.n_segment_t}"
        per_seg_ch = c3 // self.n_segment_t
        x3 = x3.reshape(bsz * self.n_segment_t, per_seg_ch, *x3.shape[-2:])
        assert per_seg_ch == self.Aug_Encoder_T.dim_in, f"T branch per_seg_ch {per_seg_ch} != Aug_Encoder_T dim_in {self.Aug_Encoder_T.dim_in}"

        x3_onset = torch.zeros_like(x3)

        # ---------- Augmentation ----------
        motion_x1_onset = self.Aug_Encoder_L(x1_onset)
        motion_x1 = self.Aug_Encoder_L(x1)
        x1 = self.Aug_Manipulator_L(motion_x1_onset, motion_x1, self.amp_factor)

        motion_x2_onset = self.Aug_Encoder_S(x2_onset)
        motion_x2 = self.Aug_Encoder_S(x2)
        x2 = self.Aug_Manipulator_S(motion_x2_onset, motion_x2, self.amp_factor)

        motion_x3_onset = self.Aug_Encoder_T(x3_onset)
        motion_x3 = self.Aug_Encoder_T(x3)
        x3 = self.Aug_Manipulator_T(motion_x3_onset, motion_x3, self.amp_factor)

        # ---------- Stem + Shared Attention ----------
        x1 = self.stem_L(x1);
        x1 = self.shared_eca_64(x1);
        x1 = self.shared_sa(x1);
        x1 = self.pool5(x1)
        x2 = self.stem_S(x2);
        x2 = self.shared_sa(x2);
        x2 = self.pool5(x2)
        x3 = self.stem_T(x3);
        x3 = self.shared_sa(x3);
        x3 = self.pool5(x3)

        # ---------------- AC1 ----------------
        ac1_x1 = self.ac1_L(x1)
        ac1_x2 = self.ac1_S(x2)
        ac1_x3 = self.ac1_T(x3)
        ac1_x3 = self.ta_128(ac1_x3, n_segment=self.n_segment_t)
        ac1_x1_g = self._global_pool_flat(self.ac1_pool(ac1_x1))
        ac1_x2_g = self._global_pool_flat(self.ac1_pool(ac1_x2))
        ac1_x3_g = self._global_pool_flat(self.ac1_pool(ac1_x3))
        ac1_feat = torch.cat([ac1_x1_g, ac1_x2_g, ac1_x3_g], dim=1)
        ac1_logits = self.ac1_head(ac1_feat)

        # 中间层
        x1m = self.mid_L(x1)
        x2m = self.mid_S(x2)
        x3m = self.mid_T(x3)

        # ---------------- AC2 ----------------
        ac2_x1 = self.ac2_L(x1m)
        ac2_x2 = self.ac2_S(x2m)
        ac2_x3 = self.ac2_T(x3m)
        ac2_x3 = self.ta_128_b(ac2_x3, n_segment=self.n_segment_t)
        ac2_x1_g = self._global_pool_flat(self.ac2_pool(ac2_x1))
        ac2_x2_g = self._global_pool_flat(self.ac2_pool(ac2_x2))
        ac2_x3_g = self._global_pool_flat(self.ac2_pool(ac2_x3))
        ac2_feat = torch.cat([ac2_x1_g, ac2_x2_g, ac2_x3_g], dim=1)
        ac2_logits = self.ac2_head(ac2_feat)

        # ---------------- Final ----------------
        final_x1 = self.final_L(x1m)
        final_x2 = self.final_S(x2m)
        final_x3 = self.final_T(x3m)
        final_x3 = self.ta_128_c(final_x3, n_segment=self.n_segment_t)
        final_x1_g = self._global_pool_flat(self.final_pool(final_x1))
        final_x2_g = self._global_pool_flat(self.final_pool(final_x2))
        final_x3_g = self._global_pool_flat(self.final_pool(final_x3))
        final_feat = torch.cat([final_x1_g, final_x2_g, final_x3_g], dim=1)
        final_logits = self.final_head(final_feat)

        return ac1_logits, ac2_logits, final_logits, ac1_feat, ac2_feat, final_feat


def get_model(model_name: str, class_num: int, alpha: int):
    if model_name in ["SKD_TSTSAN", "SKD_TSTSAN_v2"]:
        return SKD_TSTSAN_v2(class_num, alpha)
    raise ValueError(f"Unknown model name: {model_name}")
