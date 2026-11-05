"""Простые признаки ЭКГ для небейронного бейзлайна.

1. Статистики сигнала по каждому из 12 отведений (векторизованно, без обучаемых параметров).
2. Ритм и интервалы (ЧСС, RR, PR, QRS, QT, QTc) через neurokit2 по одному отведению.
   Значения вне физиологичных границ и записи, где детекция упала, превращаются в NaN
   (LightGBM работает с NaN нативно). Доля валидных значений считается отдельно
   (`coverage`), чтобы решить, можно ли доверять признакам.

Признаки считаются независимо по каждой записи и ничего не оценивают по выборке,
поэтому утечки между train/val/test быть не может.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from scipy import stats as sps

logger = logging.getLogger(__name__)

FEATURES_VERSION = 1
STAT_NAMES: tuple[str, ...] = (
    "mean", "std", "min", "max", "p5", "p25", "p50", "p75", "p95", "skew", "kurt", "mad_diff",
)
RHYTHM_NAMES: tuple[str, ...] = (
    "n_beats", "hr_bpm", "rr_mean_s", "rr_std_s", "rr_rmssd_s", "rr_min_s", "rr_max_s",
)
INTERVAL_NAMES: tuple[str, ...] = ("pr_ms", "qrs_ms", "qt_ms", "qtc_ms", "p_frac")


# ----------------------------------------------------------------------------- статистики

def signal_stats(X: np.ndarray) -> np.ndarray:
    """Статистики по каждому отведению: (N, 12, T) -> (N, 12 * len(STAT_NAMES)) float32.

    Порядок столбцов: отведение-мажорный (I_mean, I_std, ..., II_mean, ...).
    """
    x = X.astype(np.float64, copy=False)
    q = np.percentile(x, [5, 25, 50, 75, 95], axis=-1)  # (5, N, 12)
    cols = [
        x.mean(-1), x.std(-1), x.min(-1), x.max(-1),
        q[0], q[1], q[2], q[3], q[4],
        sps.skew(x, axis=-1), sps.kurtosis(x, axis=-1),
        np.abs(np.diff(x, axis=-1)).mean(-1),  # средняя абс. первая разность ~ «шумность»/крутизна
    ]
    out = np.stack(cols, axis=-1)  # (N, 12, S)
    out = np.nan_to_num(out, nan=0.0)  # константный сигнал даёт NaN в skew/kurt
    return out.reshape(len(X), -1).astype(np.float32)


def stat_feature_names(leads: list[str]) -> list[str]:
    """Имена столбцов для `signal_stats` (в том же порядке)."""
    return [f"{lead}_{s}" for lead in leads for s in STAT_NAMES]


# ----------------------------------------------------------------------------- ритм и интервалы

def _nan_rhythm() -> dict[str, float]:
    return {k: np.nan for k in (*RHYTHM_NAMES, *INTERVAL_NAMES)}


def _median_valid(values_ms: np.ndarray, lo: float, hi: float, min_beats: int) -> float:
    """Медиана по ударам, учитывая только значения в [lo, hi] мс; мало валидных -> NaN."""
    ok = values_ms[np.isfinite(values_ms) & (values_ms >= lo) & (values_ms <= hi)]
    return float(np.median(ok)) if len(ok) >= min_beats else np.nan


def rhythm_features(sig: np.ndarray, fs: int, cfg: dict[str, Any]) -> dict[str, float]:
    """ЧСС, статистики RR и интервалы PR/QRS/QT/QTc по одному отведению.

    Любая ошибка neurokit2 (плоский сигнал, мало ударов) -> все признаки NaN.
    Интервалы при 100 Гц имеют разрешение 10 мс — это грубые оценки.
    """
    import neurokit2 as nk  # импорт внутри: в joblib-воркерах он нужен лишь здесь

    out = _nan_rhythm()
    lim = cfg["limits"]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clean = nk.ecg_clean(sig.astype(np.float64), sampling_rate=fs, method="neurokit")
            _, info = nk.ecg_peaks(clean, sampling_rate=fs, method="neurokit")
            r = np.asarray(info["ECG_R_Peaks"], dtype=np.int64)
            out["n_beats"] = float(len(r))
            if len(r) < 3:
                return out
            rr = np.diff(r) / fs  # секунды
            rr = rr[(rr >= lim["rr_s"][0]) & (rr <= lim["rr_s"][1])]
            if len(rr) < 2:
                return out
            out["rr_mean_s"] = float(rr.mean())
            out["hr_bpm"] = float(60.0 / rr.mean())
            out["rr_std_s"] = float(rr.std())
            out["rr_rmssd_s"] = float(np.sqrt(np.mean(np.diff(rr) ** 2))) if len(rr) >= 3 else np.nan
            out["rr_min_s"], out["rr_max_s"] = float(rr.min()), float(rr.max())

            _, w = nk.ecg_delineate(clean, r, sampling_rate=fs, method="dwt")
            get = lambda k: np.asarray(w[k], dtype=np.float64)  # noqa: E731
            to_ms = 1000.0 / fs
            mb = cfg["min_valid_beats"]
            out["pr_ms"] = _median_valid((get("ECG_R_Onsets") - get("ECG_P_Onsets")) * to_ms, *lim["pr_ms"], mb)
            out["qrs_ms"] = _median_valid((get("ECG_R_Offsets") - get("ECG_R_Onsets")) * to_ms, *lim["qrs_ms"], mb)
            qt = _median_valid((get("ECG_T_Offsets") - get("ECG_R_Onsets")) * to_ms, *lim["qt_ms"], mb)
            out["qt_ms"] = qt
            out["qtc_ms"] = float(qt / np.sqrt(rr.mean())) if np.isfinite(qt) else np.nan  # Bazett
            out["p_frac"] = float(np.isfinite(get("ECG_P_Peaks")).mean())
    except Exception as e:  # noqa: BLE001 — устойчивость важнее: сбой одной записи не валит прогон
        logger.debug("neurokit2 не справился: %s: %s", type(e).__name__, e)
    return out


def _rhythm_chunk(sigs: np.ndarray, fs: int, cfg: dict[str, Any]) -> np.ndarray:
    """Ритм-признаки для пачки сигналов (n, T) -> (n, len(RHYTHM)+len(INTERVAL))."""
    names = (*RHYTHM_NAMES, *INTERVAL_NAMES)
    rows = []
    for s in sigs:
        feats = rhythm_features(s, fs, cfg)  # один вызов на запись
        rows.append([feats[k] for k in names])
    return np.array(rows, dtype=np.float32)


def rhythm_matrix(
    X: np.ndarray, leads: list[str], fs: int, cfg: dict[str, Any], n_jobs: int
) -> tuple[np.ndarray, list[str]]:
    """Ритм/интервалы по отведению `cfg['lead']` для всех записей, параллельно по процессам."""
    idx = leads.index(cfg["lead"])
    chunk = cfg["chunk_size"]
    parts = Parallel(n_jobs=n_jobs, return_as="generator")(
        delayed(_rhythm_chunk)(np.ascontiguousarray(X[i:i + chunk, idx]), fs, cfg)
        for i in range(0, len(X), chunk)
    )
    blocks = []
    t0 = time.perf_counter()
    for k, part in enumerate(parts, 1):
        blocks.append(part)
        if k % 10 == 0:
            logger.info("  neurokit2: %d/%d записей (%.0f с)", min(k * chunk, len(X)), len(X), time.perf_counter() - t0)
    names = [f"{cfg['lead']}_{n}" for n in (*RHYTHM_NAMES, *INTERVAL_NAMES)]
    return np.concatenate(blocks, axis=0), names


# ----------------------------------------------------------------------------- сборка и кэш

def coverage(F: np.ndarray, names: list[str]) -> dict[str, float]:
    """Доля записей с не-NaN значением по каждому признаку."""
    return {n: float(np.isfinite(F[:, j]).mean()) for j, n in enumerate(names)}


def _fingerprint(cfg: dict[str, Any], data_fingerprint: str) -> str:
    payload = json.dumps({"cfg": cfg, "data": data_fingerprint, "v": FEATURES_VERSION}, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def compute_features(
    X: np.ndarray, leads: list[str], fs: int, cfg: dict[str, Any], n_jobs: int
) -> tuple[np.ndarray, list[str]]:
    """Все признаки одной выборки: статистики (всегда) + ритм/интервалы neurokit2 (если включены)."""
    mats = [signal_stats(X)]
    names = stat_feature_names(leads)
    if cfg["rhythm"]["enabled"]:
        R, rnames = rhythm_matrix(X, leads, fs, cfg["rhythm"], n_jobs)
        mats.append(R)
        names += rnames
    return np.concatenate(mats, axis=1), names


def get_features(
    split: str,
    X: np.ndarray,
    leads: list[str],
    fs: int,
    cfg: dict[str, Any],
    cache_dir: Path,
    data_fingerprint: str,
    n_jobs: int,
) -> tuple[np.ndarray, list[str]]:
    """Признаки выборки с кэшем в `cache_dir` (сбрасывается при смене конфига/данных/версии)."""
    fp = _fingerprint(cfg, data_fingerprint)
    f_path, m_path = cache_dir / f"F_{split}.npy", cache_dir / f"F_{split}.json"
    if f_path.exists() and m_path.exists():
        meta = json.loads(m_path.read_text(encoding="utf-8"))
        if meta["fingerprint"] == fp:
            logger.info("%s: признаки из кэша", split)
            return np.load(f_path), meta["names"]
    t0 = time.perf_counter()
    F, names = compute_features(X, leads, fs, cfg, n_jobs)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(f_path, F)
    m_path.write_text(json.dumps({"fingerprint": fp, "names": names}, ensure_ascii=False), encoding="utf-8")
    logger.info("%s: признаки посчитаны за %.0f с, форма %s", split, time.perf_counter() - t0, F.shape)
    return F, names
