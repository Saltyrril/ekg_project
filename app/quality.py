"""Эвристическая проверка качества 12-канального сигнала ЭКГ (100 Гц, мВ) перед подачей в модель.

Проверки грубые и объяснимые, без обучения: плоские отведения и обрывы, ВЧ-шум, дрейф изолинии, наличие QRS и
правдоподобный пульс, правдоподобная амплитуда (признак неверных единиц), соотношения Эйнтховена между
отведениями (признак перепутанных электродов или порядка отведений), дубликаты отведений.

Это НЕ медицинская оценка качества: пороги подобраны по фолду 9 PTB-XL (scripts/check_quality.py, результаты в
results/) и могут не подходить для других приборов. Функция assess_quality работает со всем сигналом
(12, T) и не зависит от torch и Gradio.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

# Пороги. Значения подобраны по фолду 9 PTB-XL (см. results/*_quality_check), на test не смотрели.
THRESHOLDS: dict[str, float] = {
    "flat_p2p_mv": 0.05,        # размах (0.5–99.5 перцентили) меньше -> отведение «плоское»
    "flat_run_warn": 0.15,      # доля отсчётов в неподвижных участках >= 0.3 с (обрыв, насыщение)
    "flat_run_poor": 0.40,
    "hf_warn": 0.10,            # std(ВЧ > 30 Гц) / медианный размах отведений; чистый val: p99 0.10
    "hf_poor": 0.12,            # искусственный шум 0.2 размаха (метрика ~0.16) роняет macro AUROC на 0.11
    "bw_warn": 1.50,            # размах дрейфа < 0.5 Гц / медианный размах отведений; чистый val: p95 2.2 (дрейф у ~10% записей)
    "bw_poor": 2.50,
    "min_qrs_contrast": 3.0,    # медиана энергии в пиках QRS / медиана энергии: val >= 4.6, шум <= 1.9
    "max_p2p_mv": 12.0,         # размах больше -> нереально для ЭКГ (неверные единицы?)
    "median_p2p_mv": 0.15,      # медианный размах меньше -> слишком мало (неверные единицы?)
    "min_beats": 2,
    "hr_min": 30.0,
    "hr_max": 220.0,
    "einthoven_warn": 0.30,    # rms невязки соотношений отведений / rms сигнала
    "poor_noisy_leads": 6,      # столько отведений с «poor»-шумом/дрейфом -> общая оценка poor
}
FLAT_RUN_SECONDS = 0.3

STATUS_RU: dict[str, str] = {"ok": "хорошее", "warning": "есть замечания", "poor": "низкое"}


@dataclass
class QualityIssue:
    """Одно замечание: код, серьёзность ('warning' | 'poor'), текст, затронутые отведения."""

    code: str
    severity: str
    message: str
    leads: list[str] = field(default_factory=list)


@dataclass
class QualityReport:
    """Итог: status ('ok' | 'warning' | 'poor'), замечания и числовые метрики (для отчёта и калибровки порогов)."""

    status: str
    issues: list[QualityIssue]
    metrics: dict[str, Any]

    @property
    def status_ru(self) -> str:
        return STATUS_RU[self.status]


def _sos(kind: str, cutoff: float | list[float], fs: int) -> np.ndarray:
    return butter(2, cutoff, btype=kind, fs=fs, output="sos")


def _p2p(x: np.ndarray) -> np.ndarray:
    """Робастный размах по последней оси: перцентили 0.5 и 99.5."""
    lo, hi = np.percentile(x, [0.5, 99.5], axis=-1)
    return hi - lo


def _flat_run_fraction(x: np.ndarray, min_len: int) -> np.ndarray:
    """Доля отсчётов каждого отведения в участках, где значение не меняется >= min_len отсчётов подряд."""
    out = np.zeros(x.shape[0])
    for k, row in enumerate(x):
        same = np.concatenate(([False], np.diff(row) == 0))
        run = 0
        flat = 0
        for s in np.concatenate((same, [False])):
            if s:
                run += 1
            else:
                if run + 1 >= min_len:
                    flat += run + 1
                run = 0
        out[k] = flat / row.size
    return out


def detect_beats(x: np.ndarray, fs: int) -> tuple[np.ndarray, float]:
    """Индексы QRS и «контраст» по суммарной энергии отведений в полосе 5–20 Гц (упрощённый Pan–Tompkins).

    Контраст = медиана энергии в пиках / медиана энергии по записи. У ЭКГ пики QRS резко выделяются, у шума нет.
    """
    band = sosfiltfilt(_sos("bandpass", [5.0, 20.0], fs), x, axis=1)
    energy = (band ** 2).sum(axis=0)
    win = max(1, int(0.1 * fs))
    smooth = np.convolve(energy, np.ones(win) / win, mode="same")
    if smooth.max() <= 0:
        return np.array([], dtype=int), 0.0
    peaks, _ = find_peaks(smooth, height=0.25 * np.percentile(smooth, 99), distance=int(0.25 * fs))
    if len(peaks) == 0:
        return peaks, 0.0
    return peaks, float(np.median(smooth[peaks]) / max(np.median(smooth), 1e-12))


def _einthoven_residual(x: np.ndarray, leads: list[str]) -> float | None:
    """Наибольшая относительная невязка соотношений III=II−I, aVR=−(I+II)/2, aVL=I−II/2, aVF=II−I/2."""
    idx = {ld.upper(): i for i, ld in enumerate(leads)}
    if not all(k in idx for k in ("I", "II", "III", "AVR", "AVL", "AVF")):
        return None
    i, ii, iii, avr, avl, avf = (x[idx[k]] for k in ("I", "II", "III", "AVR", "AVL", "AVF"))
    rel = []
    for got, expect in ((iii, ii - i), (avr, -(i + ii) / 2), (avl, i - ii / 2), (avf, ii - i / 2)):
        scale = max(float(np.sqrt(np.mean(np.square(got - got.mean())))), float(np.sqrt(np.mean(np.square(expect - expect.mean())))), 1e-6)
        rel.append(float(np.sqrt(np.mean(np.square((got - got.mean()) - (expect - expect.mean())))) / scale))
    return max(rel)


def lead_metrics(x: np.ndarray, fs: int) -> dict[str, np.ndarray]:
    """Метрики по каждому отведению: размах, доля неподвижных участков, ВЧ-шум, дрейф изолинии."""
    p2p = _p2p(x)
    hf = sosfiltfilt(_sos("highpass", 30.0, fs), x, axis=1)
    bw = sosfiltfilt(_sos("lowpass", 0.5, fs), x, axis=1)
    # знаменатель — медианный по отведениям размах «чистой» части (0.5–30 Гц), без самого дрейфа и шума (иначе метрики
    # насыщаются). Единый для записи, чтобы малоамплитудные отведения (III, aVR, aVL) не давали ложных срабатываний.
    safe = max(float(np.median(_p2p(x - bw - hf))), 1e-6)
    return {
        "p2p_mv": p2p,
        "flat_run": _flat_run_fraction(x, int(round(FLAT_RUN_SECONDS * fs))),
        "hf_noise": hf.std(axis=1) / safe,
        "baseline_wander": _p2p(bw) / safe,
    }


def assess_quality(signal: np.ndarray, leads: list[str], fs: int, thr: dict[str, float] | None = None) -> QualityReport:
    """Оценка качества записи (12, T) в мВ. Возвращает статус, замечания и метрики."""
    t = {**THRESHOLDS, **(thr or {})}
    x = np.asarray(signal, dtype=np.float64)
    names = [("a" + ld[1:]) if ld.upper().startswith("AV") else ld for ld in leads]
    m = lead_metrics(x, fs)
    issues: list[QualityIssue] = []

    def pick(mask: np.ndarray) -> list[str]:
        return [names[k] for k in np.flatnonzero(mask)]

    flat = m["p2p_mv"] < t["flat_p2p_mv"]
    if flat.any():
        issues.append(QualityIssue("flat_lead", "poor", "Нет сигнала (плоская линия, отведение отключено?).", pick(flat)))
    stuck_poor = (m["flat_run"] >= t["flat_run_poor"]) & ~flat
    stuck_warn = (m["flat_run"] >= t["flat_run_warn"]) & ~stuck_poor & ~flat
    if stuck_poor.any():
        issues.append(QualityIssue("dropout", "poor", "Большие неподвижные участки (обрыв или насыщение сигнала).", pick(stuck_poor)))
    if stuck_warn.any():
        issues.append(QualityIssue("dropout", "warning", "Есть неподвижные участки (возможен обрыв или насыщение).", pick(stuck_warn)))

    noisy_poor = (m["hf_noise"] >= t["hf_poor"]) & ~flat
    noisy_warn = (m["hf_noise"] >= t["hf_warn"]) & ~noisy_poor & ~flat
    if noisy_poor.any():
        issues.append(QualityIssue("hf_noise", "warning", "Сильный высокочастотный шум (мышечный наводок, плохой контакт).", pick(noisy_poor)))
    if noisy_warn.any():
        issues.append(QualityIssue("hf_noise", "warning", "Заметный высокочастотный шум.", pick(noisy_warn)))
    wander_poor = (m["baseline_wander"] >= t["bw_poor"]) & ~flat
    wander_warn = (m["baseline_wander"] >= t["bw_warn"]) & ~wander_poor & ~flat
    if wander_poor.any():
        issues.append(QualityIssue("baseline_wander", "warning", "Сильный дрейф изолинии.", pick(wander_poor)))
    if wander_warn.any():
        issues.append(QualityIssue("baseline_wander", "warning", "Заметный дрейф изолинии.", pick(wander_warn)))
    n_bad_noise = int((noisy_poor | wander_poor).sum())
    if n_bad_noise >= t["poor_noisy_leads"]:
        issues.append(QualityIssue("widespread_noise", "poor", f"Сильный шум или дрейф в {n_bad_noise} из {len(names)} отведений.", []))

    active = ~flat
    p2p_active = m["p2p_mv"][active]
    if p2p_active.size and float(m["p2p_mv"].max()) > t["max_p2p_mv"]:
        issues.append(QualityIssue("amplitude_high", "poor", f"Нереально большой размах (до {m['p2p_mv'].max():.1f} мВ): возможно, "
                                   "сигнал не в мВ или это артефакт.", pick(m["p2p_mv"] > t["max_p2p_mv"])))
    if p2p_active.size and float(np.median(p2p_active)) < t["median_p2p_mv"]:
        issues.append(QualityIssue("amplitude_low", "warning", f"Очень малый размах (медиана {np.median(p2p_active):.2f} мВ): "
                                   "возможно, единицы указаны неверно.", []))

    beats, contrast = detect_beats(x, fs)
    duration = x.shape[1] / fs
    rr = np.diff(beats) / fs
    hr = float(60.0 / np.median(rr)) if len(rr) else float("nan")
    if len(beats) < t["min_beats"] or contrast < t["min_qrs_contrast"]:
        issues.append(QualityIssue("no_qrs", "poor", f"Комплексы QRS не выделяются (найдено {len(beats)} за {duration:.0f} с, "
                                   f"контраст {contrast:.1f}). Запись похожа на шум, не содержит ЭКГ или ритм слишком редкий.", []))
    elif not (t["hr_min"] <= hr <= t["hr_max"]):
        issues.append(QualityIssue("heart_rate", "warning", f"Нереалистичный пульс по детектору: {hr:.0f} уд/мин.", []))

    einthoven = _einthoven_residual(x, leads)
    if einthoven is not None and einthoven >= t["einthoven_warn"]:
        issues.append(QualityIssue("einthoven", "warning", f"Отведения не согласованы друг с другом (невязка {einthoven:.2f}): "
                                   "возможно, перепутаны электроды, порядок или полярность отведений, либо в отдельных отведениях сильный шум.", []))

    dup = [f"{names[a]}={names[b]}" for a in range(len(names)) for b in range(a + 1, len(names))
           if m["p2p_mv"][a] >= t["flat_p2p_mv"] and np.abs(x[a] - x[b]).max() < 1e-6]
    if dup:
        issues.append(QualityIssue("duplicate_leads", "warning", "Одинаковые отведения: " + ", ".join(dup) + ".", []))

    status = "poor" if any(i.severity == "poor" for i in issues) else ("warning" if issues else "ok")
    metrics = {
        "per_lead": {names[k]: {key: float(v[k]) for key, v in m.items()} for k in range(len(names))},
        "n_beats": int(len(beats)),
        "qrs_contrast": round(contrast, 2),
        "heart_rate_bpm": None if np.isnan(hr) else round(hr, 1),
        "einthoven_residual": None if einthoven is None else round(einthoven, 4),
    }
    return QualityReport(status=status, issues=issues, metrics=metrics)
