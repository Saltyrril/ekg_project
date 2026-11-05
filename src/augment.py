"""Аугментации 12-канальной ЭКГ для батча на устройстве: (N, 12, T) -> (N, 12, T).

Применяются ПОСЛЕ z-нормализации (по статистикам train), поэтому амплитуды шума и дрейфа
заданы в единицах «std отведения», а не в мВ. Только для train: val/test не аугментируются.

У аугментатора собственный генератор случайных чисел: при `enabled: false` он не создаётся,
и обучение (порядок батчей, инициализация) идёт ровно так же, как без этого модуля.
"""
from __future__ import annotations

import math
from typing import Any

import torch

AUGMENTATIONS: tuple[str, ...] = ("time_shift", "amplitude", "baseline_wander", "noise")


class Augmenter:
    """Случайные искажения сигнала. Каждая аугментация применяется к записи независимо с вероятностью p.

    Порядок: сдвиг по времени -> масштаб амплитуды -> дрейф изолинии -> гауссов шум.
    Конфиг (все секции необязательны; p: 0 выключает аугментацию):
      time_shift:      {p, max_shift_sec}              сдвиг ±max_shift_sec, края дублируются
      amplitude:       {p, low, high, per_lead}        множитель ~U(low, high); per_lead — свой на отведение
      baseline_wander: {p, max_amp, freq_hz: [lo, hi]} синус с частотой ~U(lo, hi) Гц, амплитуда ~U(0, max_amp)
      noise:           {p, std}                        N(0, std²), независимо по отсчётам
    """

    def __init__(self, cfg: dict[str, Any], fs: int, device: torch.device, seed: int) -> None:
        unknown = set(cfg) - set(AUGMENTATIONS) - {"enabled"}
        if unknown:
            raise ValueError(f"Неизвестные аугментации в конфиге: {sorted(unknown)}. Доступны: {AUGMENTATIONS}")
        self.cfg, self.fs, self.device = cfg, fs, device
        self.gen = torch.Generator(device=device).manual_seed(seed)

    def _rand(self, *shape: int) -> torch.Tensor:
        """U[0, 1) нужной формы из собственного генератора."""
        return torch.rand(shape, generator=self.gen, device=self.device)

    def _apply_mask(self, n: int, p: float) -> torch.Tensor:
        """(N, 1, 1) bool: к каким записям батча применять аугментацию."""
        return self._rand(n, 1, 1) < p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Возвращает аугментированную копию батча (вход не меняется)."""
        n, n_leads, length = x.shape
        c = self.cfg

        if (s := c.get("time_shift")) and s["p"] > 0:
            max_shift = round(s["max_shift_sec"] * self.fs)
            shift = ((self._rand(n, 1, 1) * 2 - 1) * max_shift).round().long() * self._apply_mask(n, s["p"])
            idx = (torch.arange(length, device=x.device)[None, None, :] - shift).clamp(0, length - 1)
            x = x.gather(2, idx.expand(-1, n_leads, -1))  # сдвиг вправо при shift > 0, край повторяется

        if (s := c.get("amplitude")) and s["p"] > 0:
            width = n_leads if s.get("per_lead", False) else 1
            scale = s["low"] + (s["high"] - s["low"]) * self._rand(n, width, 1)
            x = x * torch.where(self._apply_mask(n, s["p"]), scale, torch.ones_like(scale))

        if (s := c.get("baseline_wander")) and s["p"] > 0:
            lo, hi = s["freq_hz"]
            freq = lo + (hi - lo) * self._rand(n, 1, 1)  # одна частота дыхания на запись
            phase = 2 * math.pi * self._rand(n, 1, 1)
            amp = s["max_amp"] * self._rand(n, n_leads, 1) * self._apply_mask(n, s["p"])  # амплитуда своя по отведениям
            t = torch.arange(length, device=x.device)[None, None, :] / self.fs
            x = x + amp * torch.sin(2 * math.pi * freq * t + phase)

        if (s := c.get("noise")) and s["p"] > 0:
            noise = torch.randn(x.shape, generator=self.gen, device=self.device, dtype=x.dtype)
            x = x + s["std"] * noise * self._apply_mask(n, s["p"])

        return x
