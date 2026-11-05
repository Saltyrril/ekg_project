"""1D-сверточные модели для 12-канальной ЭКГ. Вход (N, 12, T), выход — логиты (N, C) без sigmoid.

Sigmoid не входит в модель: BCEWithLogitsLoss сам считает его численно устойчиво,
вероятности получаются снаружи (`torch.sigmoid(logits)`).
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn


def _conv_bn_relu(in_ch: int, out_ch: int, kernel_size: int, stride: int = 1) -> nn.Sequential:
    """Conv1d (без bias, т.к. следом BatchNorm) -> BatchNorm1d -> ReLU. Длину не меняет при stride=1."""
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=kernel_size // 2, bias=False),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(inplace=True),
    )


class PoolHead(nn.Module):
    """Голова: [среднее ‖ максимум] по времени -> Dropout -> Linear.

    Среднее ловит «общий фон» записи, максимум — редкие локальные события (патологический комплекс).
    """

    def __init__(self, in_features: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, F, T) -> логиты (N, C)."""
        pooled = torch.cat([x.mean(dim=-1), x.amax(dim=-1)], dim=1)
        return self.fc(self.dropout(pooled))


class CNN1d(nn.Module):
    """Простая 1D-CNN: по одному блоку Conv-BN-ReLU-MaxPool(2) на ступень, затем PoolHead.

    При T=1000 и 5 ступенях длина сокращается 1000 -> 31, число каналов растёт 32 -> 256.
    """

    def __init__(
        self,
        in_channels: int = 12,
        num_classes: int = 5,
        channels: Sequence[int] = (32, 64, 128, 256, 256),
        kernel_size: int = 7,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_channels
        for ch in channels:
            layers += [_conv_bn_relu(prev, ch, kernel_size), nn.MaxPool1d(2)]
            prev = ch
        self.features = nn.Sequential(*layers)
        self.head = PoolHead(prev, num_classes, dropout)
        _init_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, 12, T) -> логиты (N, C)."""
        return self.head(self.features(x))


class BasicBlock1d(nn.Module):
    """Остаточный блок: y = ReLU(F(x) + shortcut(x)), F = Conv-BN-ReLU-[Dropout]-Conv-BN.

    Если меняется длина (stride=2) или число каналов, shortcut — свёртка 1×1 + BN, иначе тождество.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int, dropout: float) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm1d(out_ch)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, in_ch, T) -> (N, out_ch, ceil(T / stride))."""
        out = self.drop(self.relu(self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class ResNet1d(nn.Module):
    """ResNet-18 для 1D: stem (k=15, stride 2) -> 4 ступени BasicBlock1d -> PoolHead.

    При T=1000: 1000 -> 500 (stem) -> 500 / 250 / 125 / 63 (ступени, каналы base_width × 1/2/4/8).
    """

    def __init__(
        self,
        in_channels: int = 12,
        num_classes: int = 5,
        base_width: int = 32,
        layers: Sequence[int] = (2, 2, 2, 2),
        kernel_size: int = 7,
        stem_kernel_size: int = 15,
        block_dropout: float = 0.1,
        head_dropout: float = 0.3,
        zero_init_residual: bool = True,
    ) -> None:
        super().__init__()
        self.stem = _conv_bn_relu(in_channels, base_width, stem_kernel_size, stride=2)
        stages: list[nn.Module] = []
        prev = base_width
        for i, n_blocks in enumerate(layers):
            width = base_width * 2**i
            for j in range(n_blocks):
                stride = 2 if (i > 0 and j == 0) else 1  # понижаем длину в начале каждой ступени, кроме первой
                stages.append(BasicBlock1d(prev, width, kernel_size, stride, block_dropout))
                prev = width
        self.stages = nn.Sequential(*stages)
        self.head = PoolHead(prev, num_classes, head_dropout)
        _init_weights(self)
        if zero_init_residual:  # блок стартует как тождество -> стабильнее первые шаги
            for m in self.modules():
                if isinstance(m, BasicBlock1d):
                    nn.init.zeros_(m.bn2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, 12, T) -> логиты (N, C)."""
        return self.head(self.stages(self.stem(x)))


def _init_weights(model: nn.Module) -> None:
    """Kaiming для сверток (ReLU), стандартная инициализация BatchNorm."""
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


MODELS: dict[str, type[nn.Module]] = {"cnn1d": CNN1d, "resnet1d": ResNet1d}


def build_model(name: str, in_channels: int, num_classes: int, **params: Any) -> nn.Module:
    """Создаёт модель по имени из конфига (`model.name`) и её параметрам (`model.params`)."""
    if name not in MODELS:
        raise ValueError(f"Неизвестная модель '{name}'. Доступны: {sorted(MODELS)}")
    return MODELS[name](in_channels=in_channels, num_classes=num_classes, **params)


def set_output_bias(model: nn.Module, prior: Sequence[float]) -> None:
    """Инициализирует bias выходного слоя как logit(доля положительных в train).

    Тогда в начале обучения sigmoid выдаёт частоту класса, а не 0.5, и первые эпохи
    не тратятся на подстройку «базового уровня».
    """
    p = torch.as_tensor(prior, dtype=torch.float32).clamp(1e-4, 1 - 1e-4)
    with torch.no_grad():
        model.head.fc.bias.copy_(torch.log(p / (1 - p)))  # type: ignore[union-attr]


def count_parameters(model: nn.Module) -> int:
    """Число обучаемых параметров."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
