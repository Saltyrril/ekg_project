"""Инференс для Gradio-демо: чтение записи WFDB, предобработка, вероятности, Integrated Gradients.

Модуль не зависит от Gradio и от src/: всё нужное лежит в пакете app/ (его собирает
scripts/export_app_bundle.py). model/model_meta.json хранит статистики нормализации (train),
температуру и пороги (val, strat_fold 9).
"""
from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wfdb
from scipy.signal import resample_poly

from ecg_models import build_model

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


# ----------------------------------------------------------------------------- чтение и предобработка

def read_wfdb(paths: list[str | Path]) -> tuple[np.ndarray, list[str], int, str, list[str]]:
    """Читает пару .hea + .dat (одну запись). Возвращает (сигнал (C, T), отведения, fs, имя, единицы).

    Gradio кладёт каждый загруженный файл в свою временную папку, а .hea ссылается на .dat по имени,
    поэтому файлы сначала копируются в одну папку с исходными именами.
    """
    files = [Path(p) for p in paths]
    heads = [f for f in files if f.suffix.lower() == ".hea"]
    if len(heads) != 1:
        raise ValueError(f"Нужен ровно один файл .hea (заголовок записи WFDB), получено: {len(heads)}.")
    dats = [f for f in files if f.suffix.lower() == ".dat"]
    if not dats:
        raise ValueError("Нет файла .dat с сигналом. Загрузите оба файла записи: <имя>.hea и <имя>.dat.")
    if heads[0].stem not in {f.stem for f in dats}:
        raise ValueError(f"Файлы от разных записей: {heads[0].name} и {', '.join(f.name for f in dats)}. "
                         f"Нужна пара {heads[0].stem}.hea + {heads[0].stem}.dat.")
    with tempfile.TemporaryDirectory() as tmp:
        for f in files:
            shutil.copy(f, Path(tmp) / f.name)
        try:
            sig, fields = wfdb.rdsamp(str(Path(tmp) / heads[0].stem))
        except Exception as e:  # wfdb бросает разные исключения при битом/несовпадающем файле
            raise ValueError(f"Не удалось прочитать запись WFDB ({type(e).__name__}: {e}). "
                             f"Проверьте, что .hea и .dat от одной записи.") from e
    return (np.ascontiguousarray(sig.T, dtype=np.float32), list(fields["sig_name"]), int(round(fields["fs"])),
            heads[0].stem, list(fields.get("units") or []))


def prepare_record(signal: np.ndarray, sig_names: list[str], fs: int, name: str, units: list[str],
                   bundle: Bundle) -> Record:
    """Приводит запись к входу модели: порядок отведений, 100 Гц, 10 с. Сигнал остаётся в мВ."""
    warnings: list[str] = []
    pos = {n.upper(): i for i, n in enumerate(sig_names)}
    missing = [ld for ld in bundle.leads if ld.upper() not in pos]
    if missing:
        raise ValueError(f"В записи нет отведений {missing}. Ожидаются 12 стандартных: {bundle.leads}, "
                         f"в файле: {sig_names}.")
    sig = signal[[pos[ld.upper()] for ld in bundle.leads]]
    if any(u and u.lower() != "mv" for u in units):
        warnings.append(f"Единицы в заголовке: {sorted(set(units))}, модель обучена на мВ (как в PTB-XL).")

    if fs == 500 and bundle.fs == 100:
        sig = resample_poly(sig, up=1, down=5, axis=1).astype(np.float32)
        warnings.append("Запись 500 Гц пересчитана в 100 Гц (resample_poly). Модель обучена на готовой "
                        "100-Гц версии PTB-XL (records100), поэтому результат может немного отличаться.")
    elif fs != bundle.fs:
        raise ValueError(f"Частота {fs} Гц не поддерживается: нужна {bundle.fs} Гц (records100) или 500 Гц (records500).")

    n = sig.shape[1]
    if n < bundle.length:
        raise ValueError(f"Запись короче 10 с: {n} отсчётов при {bundle.fs} Гц, нужно {bundle.length}.")
    if n > bundle.length:
        sig = sig[:, : bundle.length]
        warnings.append(f"Запись длиннее 10 с: взяты первые {bundle.length} отсчётов.")
    if not np.isfinite(sig).all():
        raise ValueError("В сигнале есть NaN/inf (пропуски), такую запись модель обработать не может.")
    return Record(signal=np.ascontiguousarray(sig), name=name, fs_original=fs, warnings=warnings)


def load_record(paths: list[str | Path], bundle: Bundle) -> Record:
    """Чтение + предобработка одной записи из загруженных файлов."""
    return prepare_record(*read_wfdb(paths), bundle=bundle)


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
