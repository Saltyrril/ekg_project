"""Инференс для Gradio-демо: предобработка записи (читатели форматов — readers.py), вероятности, Integrated Gradients.

Модуль не зависит от Gradio и от src/: всё нужное лежит в пакете app/ (его собирает
scripts/export_app_bundle.py). model/model_meta.json хранит статистики нормализации (train),
температуру и пороги (val, strat_fold 9).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.signal import resample_poly

from ecg_models import build_model
from readers import RawRecord, canon_lead, read_files

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "model"


@dataclass
class Bundle:
    """Модель в eval-режиме и всё, что нужно для предобработки и постобработки."""

    model: torch.nn.Module
    classes: list[str]
    leads: list[str]
    fs: int
    length: int
    mean: np.ndarray
    std: np.ndarray
    temperature: np.ndarray
    thresholds: np.ndarray
    ig_steps: int
    meta: dict[str, Any]


@dataclass
class Record:
    """Запись после чтения: сигнал (12, T) в мВ на частоте модели, имя и предупреждения."""

    signal: np.ndarray
    name: str
    fs_original: int
    warnings: list[str] = field(default_factory=list)
    fmt: str = "WFDB"


def load_bundle(model_dir: Path = MODEL_DIR) -> Bundle:
    """Читает model_meta.json и best.pt. Инференс на CPU в fp32 (так же будет на бесплатном Space)."""
    meta_path = model_dir / "model_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Нет {meta_path}. Соберите пакет: python scripts/export_app_bundle.py --config configs/app.yaml"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    model = build_model(meta["model"]["name"], len(meta["leads"]), len(meta["classes"]), **meta["model"]["params"])
    model.load_state_dict(torch.load(model_dir / "best.pt", map_location="cpu", weights_only=True))
    model.eval()
    return Bundle(
        model=model,
        classes=meta["classes"],
        leads=meta["leads"],
        fs=int(meta["sampling_rate"]),
        length=int(meta["length"]),
        mean=np.asarray(meta["norm_mean"], dtype=np.float32),
        std=np.asarray(meta["norm_std"], dtype=np.float32),
        temperature=np.asarray(meta["temperature"], dtype=np.float32),
        thresholds=np.asarray(meta["thresholds"], dtype=np.float32),
        ig_steps=int(meta["ig_steps"]),
        meta=meta,
    )


# ----------------------------------------------------------------------------- предобработка

UNIT_TO_MV: dict[str, float] = {"mv": 1.0, "uv": 1e-3, "nv": 1e-6, "v": 1e3}


def unit_scale(unit: str) -> float | None:
    """Множитель перевода единицы в мВ ('uV', 'µV', 'mV', 'V'); None, если единица неизвестна."""
    return UNIT_TO_MV.get(unit.strip().lower().replace("µ", "u").replace("μ", "u").replace("mkv", "uv"))


def prepare_record(raw: RawRecord, bundle: Bundle) -> Record:
    """Приводит запись к входу модели: порядок отведений, мВ, 100 Гц, 10 с."""
    warnings: list[str] = list(raw.notes)
    names, fs = raw.sig_names, raw.fs
    pos: dict[str, int] = {}
    for i, n in enumerate(names):
        pos.setdefault(canon_lead(n), i)
    missing = [ld for ld in bundle.leads if canon_lead(ld) not in pos]
    if missing:
        raise ValueError(f"В записи нет отведений {missing}. Ожидаются 12 стандартных: {bundle.leads}, "
                         f"в файле: {names}.")
    sig = raw.signal.astype(np.float32)

    scales = [unit_scale(u) if u.strip() else None for u in raw.units]
    unknown = sorted({u for u, k in zip(raw.units, scales) if u.strip() and k is None})
    if unknown:
        warnings.append(f"Неизвестные единицы {unknown}: считаю, что сигнал в мВ (так обучалась модель).")
    elif any(k is None for k in scales):
        warnings.append("Единицы сигнала не указаны: считаю, что сигнал в мВ (так обучалась модель).")
    if any(k is not None and k != 1.0 for k in scales):
        sig = sig * np.array([k if k is not None else 1.0 for k in scales], dtype=np.float32)[:, None]
        warnings.append("Сигнал переведён в мВ по единицам из файла.")
    sig = sig[[pos[canon_lead(ld)] for ld in bundle.leads]]

    if fs < bundle.fs:
        raise ValueError(f"Частота {fs} Гц ниже {bundle.fs} Гц: такую запись использовать нельзя.")
    if fs != bundle.fs:
        ratio = Fraction(bundle.fs, fs)
        sig = resample_poly(sig, up=ratio.numerator, down=ratio.denominator, axis=1).astype(np.float32)
        warnings.append(f"Запись {fs} Гц пересчитана в {bundle.fs} Гц (resample_poly). Модель обучена на готовой "
                        f"{bundle.fs}-Гц версии PTB-XL, поэтому результат может немного отличаться.")

    n = sig.shape[1]
    if n < bundle.length:
        raise ValueError(f"Запись короче 10 с: {n} отсчётов при {bundle.fs} Гц, нужно {bundle.length}.")
    if n > bundle.length:
        sig = sig[:, : bundle.length]
        warnings.append(f"Запись длиннее 10 с: взяты первые {bundle.length} отсчётов.")
    if not np.isfinite(sig).all():
        raise ValueError("В сигнале есть NaN/inf (пропуски), такую запись модель обработать не может.")
    return Record(signal=np.ascontiguousarray(sig), name=raw.name, fs_original=fs, warnings=warnings, fmt=raw.fmt)


def load_record(paths: list[str | Path], bundle: Bundle, fs: float | None = None, unit: str | None = None) -> Record:
    """Чтение + предобработка одной записи (WFDB, EDF или CSV). `fs` и `unit` нужны только для CSV."""
    return prepare_record(read_files(paths, fs=fs, unit=unit), bundle=bundle)


def normalize(signal: np.ndarray, bundle: Bundle) -> np.ndarray:
    """z-нормализация по отведениям статистиками train (те же, что при обучении)."""
    return ((signal - bundle.mean[:, None]) / bundle.std[:, None]).astype(np.float32)


# ----------------------------------------------------------------------------- модель

@torch.no_grad()
def predict_logits(x: np.ndarray, bundle: Bundle) -> np.ndarray:
    """Логиты (N, C) в fp32. x: (N, 12, T) нормализованный."""
    return bundle.model(torch.from_numpy(x)).float().numpy()


def calibrated_probs(logits: np.ndarray, bundle: Bundle) -> np.ndarray:
    """Вероятности после температурного шкалирования (T подобрана на val)."""
    return 1.0 / (1.0 + np.exp(-logits / bundle.temperature))


def integrated_gradients(x: np.ndarray, target: int, bundle: Bundle) -> np.ndarray:
    """IG по логиту класса `target` для одной записи x (12, T), нулевой baseline (= среднее train), midpoint.

    Совпадает с src/explain.py: attr = x * mean_alpha(d logit(alpha * x) / dx), fp32, без autocast.
    """
    steps = bundle.ig_steps
    xb = torch.from_numpy(x)[None]
    alphas = ((torch.arange(steps, dtype=torch.float32) + 0.5) / steps).view(steps, 1, 1)
    pts = (alphas * xb).detach().requires_grad_(True)
    grad = torch.autograd.grad(bundle.model(pts)[:, target].sum(), pts)[0].mean(dim=0)
    return (xb[0] * grad).detach().numpy()
