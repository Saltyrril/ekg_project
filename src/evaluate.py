"""Оценка предсказаний: метрики, пороги по классам, температурная калибровка, графики.

Всё считается ТОЛЬКО на val (strat_fold 9) по сохранённым `val_probs.npy` обученной модели.
Test (фолд 10) здесь не читается: скрипт принимает только val. Подобранные на val параметры
(температура T и пороги) сохраняются в `postprocess.json`; на test их потом только применяют
(`apply_postprocess`) — подбирать что-либо на test нельзя.

Запуск:  python -m src.evaluate --run results/2026-11-06_cnn1d_aug [--run ...]
Пишет:   <run>/eval_val.json, <run>/postprocess.json, results/figures/<run>/*.png
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.optimize import minimize_scalar
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from src.data import PROJECT_ROOT, _cache_dir, _resolve, load_config

logger = logging.getLogger(__name__)

EVAL_SPLIT = "val"  # единственная выборка, на которой этот модуль подбирает параметры


# ----------------------------------------------------------------------------- базовые метрики

def multilabel_metrics(y_true: np.ndarray, y_score: np.ndarray, classes: list[str]) -> dict[str, object]:
    """AUROC и PR-AUC по классам + macro-среднее.

    y_true: (N, C) {0,1}; y_score: (N, C) вероятности/скоры. Оба метрики не зависят от порога.
    """
    auroc = {c: float(roc_auc_score(y_true[:, i], y_score[:, i])) for i, c in enumerate(classes)}
    pr_auc = {c: float(average_precision_score(y_true[:, i], y_score[:, i])) for i, c in enumerate(classes)}
    return {
        "macro_auroc": float(np.mean(list(auroc.values()))),
        "auroc": auroc,
        "macro_pr_auc": float(np.mean(list(pr_auc.values()))),
        "pr_auc": pr_auc,
    }


# ----------------------------------------------------------------------------- логиты и температура

def to_logits(probs: np.ndarray, eps: float) -> np.ndarray:
    """logit(p) в float64; p клипуется в [eps, 1-eps], иначе float32-sigmoid даёт бесконечности."""
    p = np.clip(probs.astype(np.float64), eps, 1.0 - eps)
    return np.log(p) - np.log1p(-p)


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Устойчивая сигмоида."""
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def bce_from_logits(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Поэлементная бинарная кросс-энтропия по логитам (устойчивая форма)."""
    return np.logaddexp(0.0, z) - y * z


def fit_temperature(logits: np.ndarray, y: np.ndarray, per_class: bool, bounds: tuple[float, float]) -> np.ndarray:
    """Температура T (форма (C,)) минимизирует NLL(sigmoid(logits / T)) на переданной выборке.

    T > 1 смягчает уверенность модели, T < 1 заостряет. Оптимизируем log T, чтобы T оставалась > 0.
    global: один T на все классы (повторён C раз); per_class: независимо по каждому классу.
    """
    lo, hi = np.log(bounds[0]), np.log(bounds[1])

    def argmin(z: np.ndarray, t: np.ndarray) -> float:
        res = minimize_scalar(lambda s: bce_from_logits(z / np.exp(s), t).mean(),
                              bounds=(lo, hi), method="bounded", options={"xatol": 1e-6})
        return float(np.exp(res.x))

    n_classes = logits.shape[1]
    if per_class:
        return np.array([argmin(logits[:, i], y[:, i]) for i in range(n_classes)])
    return np.full(n_classes, argmin(logits, y))


# ----------------------------------------------------------------------------- калибровка: ECE и др.

def reliability_bins(y: np.ndarray, p: np.ndarray, n_bins: int) -> dict[str, np.ndarray]:
    """Равноширинные бины по p (бинарный случай): число записей, средняя уверенность, доля положительных."""
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    count = np.bincount(idx, minlength=n_bins)
    conf_sum = np.bincount(idx, weights=p, minlength=n_bins)
    pos_sum = np.bincount(idx, weights=y, minlength=n_bins)
    safe = np.maximum(count, 1)
    return {"count": count, "confidence": conf_sum / safe, "accuracy": pos_sum / safe}


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int) -> float:
    """ECE = sum_b (n_b / N) * |доля положительных в бине - средняя вероятность в бине|."""
    b = reliability_bins(y, p, n_bins)
    return float(np.sum(b["count"] / len(y) * np.abs(b["accuracy"] - b["confidence"])))


def calibration_metrics(y: np.ndarray, p: np.ndarray, classes: list[str], n_bins: int, eps: float) -> dict[str, Any]:
    """ECE, Brier и NLL по классам + macro-среднее. y, p: (N, C)."""
    per: dict[str, dict[str, float]] = {}
    for i, c in enumerate(classes):
        pi = np.clip(p[:, i], eps, 1 - eps)
        per[c] = {
            "ece": expected_calibration_error(y[:, i], p[:, i], n_bins),
            "brier": float(np.mean((p[:, i] - y[:, i]) ** 2)),
            "nll": float(np.mean(-(y[:, i] * np.log(pi) + (1 - y[:, i]) * np.log1p(-pi)))),
        }
    macro = {k: float(np.mean([v[k] for v in per.values()])) for k in ("ece", "brier", "nll")}
    return {"per_class": per, "macro": macro, "n_bins": n_bins}


# ----------------------------------------------------------------------------- пороги и F1

def select_thresholds(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Порог каждого класса — значение скора, максимизирующее F1 (точный перебор по PR-кривой).

    Предсказание: p >= порог. Если в классе нет положительных, порог 0.5 (с предупреждением).
    """
    thr = np.full(y.shape[1], 0.5)
    for i in range(y.shape[1]):
        if y[:, i].sum() == 0:
            logger.warning("Класс #%d: нет положительных записей, порог 0.5", i)
            continue
        prec, rec, ts = precision_recall_curve(y[:, i], p[:, i])
        denom = prec[:-1] + rec[:-1]  # у последней точки (prec=1, rec=0) порога нет
        f1 = np.divide(2 * prec[:-1] * rec[:-1], denom, out=np.zeros_like(denom), where=denom > 0)
        thr[i] = ts[int(np.argmax(f1))]
    return thr


def threshold_metrics(y: np.ndarray, p: np.ndarray, thr: np.ndarray, classes: list[str]) -> dict[str, Any]:
    """Матрица ошибок 2x2, precision/recall/specificity/F1 по классам + macro/micro F1."""
    pred = p >= thr[None, :]
    yb = y.astype(bool)
    per: dict[str, dict[str, float]] = {}
    for i, c in enumerate(classes):
        tp, fp = int((pred[:, i] & yb[:, i]).sum()), int((pred[:, i] & ~yb[:, i]).sum())
        fn, tn = int((~pred[:, i] & yb[:, i]).sum()), int((~pred[:, i] & ~yb[:, i]).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        per[c] = {
            "threshold": float(thr[i]), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "specificity": tn / (tn + fp) if tn + fp else 0.0,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        }
    tp_all = sum(v["tp"] for v in per.values())
    fp_all, fn_all = sum(v["fp"] for v in per.values()), sum(v["fn"] for v in per.values())
    micro = 2 * tp_all / (2 * tp_all + fp_all + fn_all) if tp_all else 0.0
    return {"per_class": per, "macro_f1": float(np.mean([v["f1"] for v in per.values()])), "micro_f1": float(micro)}


# ----------------------------------------------------------------------------- постобработка (T + пороги)

def fit_postprocess(probs: np.ndarray, y: np.ndarray, cfg: dict[str, Any]) -> dict[str, Any]:
    """Подбирает T (NLL) и затем пороги (max F1 по откалиброванным вероятностям). Только по переданной выборке.

    Монотонная калибровка не меняет ранжирование, поэтому AUROC/PR-AUC и достижимый F1 те же, что у сырых
    вероятностей; выигрыш — в интерпретируемости порогов и вероятностей.
    """
    z = to_logits(probs, cfg["probs_eps"])
    temp = fit_temperature(z, y, cfg["temperature"]["mode"] == "per_class", tuple(cfg["temperature"]["bounds"]))
    thr = select_thresholds(y, sigmoid(z / temp[None, :]))
    return {"temperature": temp, "thresholds": thr}


def apply_postprocess(probs: np.ndarray, params: dict[str, Any], eps: float = 1e-7) -> tuple[np.ndarray, np.ndarray]:
    """Применяет готовые T и пороги (например, с val) к новым вероятностям -> (калибр. вероятности, 0/1 предсказания)."""
    temp = np.asarray(params["temperature"], dtype=np.float64)
    thr = np.asarray(params["thresholds"], dtype=np.float64)
    p_cal = sigmoid(to_logits(probs, eps) / temp[None, :])
    return p_cal, (p_cal >= thr[None, :]).astype(np.uint8)


def crossfit_estimate(
    probs: np.ndarray, y: np.ndarray, groups: np.ndarray, classes: list[str], cfg: dict[str, Any]
) -> dict[str, Any]:
    """Несмещённая внутри-val оценка F1 и ECE: T и пороги подбираются на K-1 частях, считаются на оставшейся.

    Части формируются по patient_id. Предсказания всех удержанных частей склеиваются и оцениваются разом.
    """
    k = cfg["crossfit"]["n_folds"]
    rng = np.random.default_rng(cfg["seed"])
    uniq, inv = np.unique(groups, return_inverse=True)
    fold = (rng.permutation(len(uniq)) % k)[inv]
    p_cal = np.zeros_like(probs, dtype=np.float64)
    pred = np.zeros(probs.shape, dtype=bool)
    thr_all: list[np.ndarray] = []
    temp_all: list[np.ndarray] = []
    for f in range(k):
        tr, te = fold != f, fold == f
        params = fit_postprocess(probs[tr], y[tr], cfg)
        p_cal[te], pred[te] = apply_postprocess(probs[te], params, cfg["probs_eps"])
        thr_all.append(params["thresholds"])
        temp_all.append(params["temperature"])
    # Склеенные предсказания удержанных частей оцениваем теми же функциями (порог 0.5 к бинарным 0/1 -> pred)
    f1_cf = threshold_metrics(y, pred.astype(np.float64), np.full(y.shape[1], 0.5), classes)
    f1_05 = threshold_metrics(y, probs, np.full(y.shape[1], 0.5), classes)  # без подбора: порог 0.5, сырые вероятности
    return {
        "n_folds": k,
        "macro_f1": f1_cf["macro_f1"],
        "micro_f1": f1_cf["micro_f1"],
        "f1_per_class": {c: v["f1"] for c, v in f1_cf["per_class"].items()},
        "macro_f1_at_0.5_raw": f1_05["macro_f1"],
        "ece_calibrated": calibration_metrics(y, p_cal, classes, cfg["ece"]["n_bins"], cfg["probs_eps"]),
        "ece_raw": calibration_metrics(y, probs.astype(np.float64), classes, cfg["ece"]["n_bins"], cfg["probs_eps"]),
        "threshold_spread_over_folds": {
            c: {"min": float(np.min([t[i] for t in thr_all])), "max": float(np.max([t[i] for t in thr_all]))}
            for i, c in enumerate(classes)
        },
        "temperature_over_folds": [t.tolist() for t in temp_all],
    }


# ----------------------------------------------------------------------------- графики

# Категориальные слоты 1-5 референсной палитры (светлая тема), порядок фиксирован: класс -> цвет.
CLASS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE_RAMP = ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # sequential, light -> dark


def _plt() -> Any:
    """matplotlib с неинтерактивным бэкендом и спокойным оформлением (импорт отложен: train.py не нуждается в нём)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 11, "axes.titleweight": "regular",
        "axes.titlecolor": INK, "legend.frameon": False, "legend.fontsize": 9, "font.size": 10,
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
    })
    return plt


def _style_legend(ax: Any) -> None:
    """Текст легенды — нейтральными чернилами (цвет несёт только маркер линии)."""
    for t in ax.get_legend().get_texts():
        t.set_color(INK2)


def plot_roc(y: np.ndarray, p: np.ndarray, ops: dict[str, Any], classes: list[str], title: str, path: Path, dpi: int) -> None:
    """ROC-кривые по классам; кружок — рабочая точка (порог из подбора по F1)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1.2, ls=(0, (4, 4)), zorder=1)
    for i, c in enumerate(classes):
        fpr, tpr, _ = roc_curve(y[:, i], p[:, i])
        auc = roc_auc_score(y[:, i], p[:, i])
        ax.plot(fpr, tpr, color=CLASS_COLORS[i], label=f"{c}  AUC {auc:.3f}", zorder=2)
        o = ops["per_class"][c]
        ax.plot(1 - o["specificity"], o["recall"], "o", ms=8, color=CLASS_COLORS[i], mec=SURFACE, mew=2, zorder=3)
    ax.set(xlabel="FPR (1 − специфичность)", ylabel="TPR (полнота)", xlim=(0, 1), ylim=(0, 1.01), title=title)
    ax.set_aspect("equal")
    ax.legend(loc="lower right")
    _style_legend(ax)
    fig.text(0.01, 0.01, "кружок — рабочая точка при пороге, подобранном по F1 на val", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_pr(y: np.ndarray, p: np.ndarray, ops: dict[str, Any], classes: list[str], title: str, path: Path, dpi: int) -> None:
    """PR-кривые по классам; кружок — рабочая точка; в легенде AP и доля класса (уровень «случайного» классификатора)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    for i, c in enumerate(classes):
        prec, rec, _ = precision_recall_curve(y[:, i], p[:, i])
        ap = average_precision_score(y[:, i], p[:, i])
        ax.plot(rec, prec, color=CLASS_COLORS[i], label=f"{c}  AP {ap:.3f} · доля {y[:, i].mean():.0%}", zorder=2)
        o = ops["per_class"][c]
        ax.plot(o["recall"], o["precision"], "o", ms=8, color=CLASS_COLORS[i], mec=SURFACE, mew=2, zorder=3)
    ax.set(xlabel="Recall (полнота)", ylabel="Precision (точность)", xlim=(0, 1), ylim=(0, 1.02), title=title)
    ax.legend(loc="lower left")
    _style_legend(ax)
    fig.text(0.01, 0.01, "кружок — рабочая точка при пороге, подобранном по F1 на val", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_confusion(ops: dict[str, Any], classes: list[str], title: str, path: Path, dpi: int) -> None:
    """Матрицы ошибок 2x2 по каждому классу (one-vs-rest). Цвет — доля в строке, в ячейке — число и %."""
    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)
    fig, axes = plt.subplots(1, len(classes), figsize=(2.9 * len(classes), 3.6))
    for ax, c in zip(np.atleast_1d(axes), classes):
        o = ops["per_class"][c]
        cm = np.array([[o["tn"], o["fp"]], [o["fn"], o["tp"]]])
        frac = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(frac, cmap=cmap, vmin=0, vmax=1.15)
        ax.grid(False)
        for r in range(2):
            for k in range(2):
                ax.text(k, r, f"{cm[r, k]}\n{frac[r, k]:.0%}", ha="center", va="center", fontsize=11,
                        color="#ffffff" if frac[r, k] > 0.5 else INK)
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["нет", "да"], yticklabels=["нет", "да"],
               xlabel="предсказано", title=f"{c}  (порог {o['threshold']:.2f})")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    np.atleast_1d(axes)[0].set_ylabel("истинно")
    fig.suptitle(title, fontsize=11, x=0.01, ha="left")
    fig.text(0.01, 0.01, "доля — от числа записей в строке (истинного класса)", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_calibration(
    y: np.ndarray, p_raw: np.ndarray, p_cal: np.ndarray, calib_raw: dict[str, Any], calib_cal: dict[str, Any],
    classes: list[str], temp: np.ndarray, n_bins: int, min_count: int, title: str, path: Path, dpi: int,
) -> None:
    """Кривые надёжности до/после температурной калибровки + гистограмма вероятностей (лог-шкала)."""
    plt = _plt()
    from matplotlib.ticker import NullFormatter
    n = len(classes)
    fig, axes = plt.subplots(2, n, figsize=(2.9 * n, 5.0), gridspec_kw={"height_ratios": [3, 1]}, sharex="col")
    edges = np.linspace(0, 1, n_bins + 1)
    for i, c in enumerate(classes):
        ax, hx = axes[0, i], axes[1, i]
        ax.plot([0, 1], [0, 1], color=AXIS, lw=1.2, ls=(0, (4, 4)), zorder=1)
        for p, color, label in ((p_raw, MUTED, "до"), (p_cal, CLASS_COLORS[0], "после T")):
            b = reliability_bins(y[:, i], p[:, i], n_bins)
            keep = b["count"] >= min_count
            ax.plot(b["confidence"][keep], b["accuracy"][keep], "-o", ms=6, color=color, mec=SURFACE, mew=1.5,
                    label=label, zorder=2)
        ax.set(xlim=(0, 1), ylim=(0, 1.02), title=f"{c}: ECE {calib_raw['per_class'][c]['ece']:.3f} → "
                                                  f"{calib_cal['per_class'][c]['ece']:.3f}")
        ax.set_aspect("equal", adjustable="box")
        if i == 0:
            ax.set_ylabel("доля положительных")
            ax.legend(loc="upper left")
            _style_legend(ax)
        hx.hist(p_raw[:, i], bins=edges, color=MUTED, alpha=0.85)
        hx.hist(p_cal[:, i], bins=edges, histtype="step", color=CLASS_COLORS[0], lw=1.6)
        hx.set_yscale("log")
        hx.yaxis.set_minor_formatter(NullFormatter())  # без налезающих друг на друга подписей минорных делений
        hx.set_xlabel("предсказанная вероятность")
        if i == 0:
            hx.set_ylabel("записей")
    t = np.unique(np.round(temp, 3))
    fig.suptitle(f"{title} · T = {', '.join(str(x) for x in t)}", fontsize=11, x=0.01, ha="left")
    fig.text(0.01, 0.005, f"{n_bins} равноширинных бинов; бины с < {min_count} записей на кривой не показаны",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# ----------------------------------------------------------------------------- запуск на одном эксперименте

def _git_state() -> dict[str, Any]:
    """Хэш коммита и признак грязного дерева (копия из train.py: тот тянет torch, а здесь он не нужен)."""
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    try:
        return {"commit": run("rev-parse", "--short", "HEAD"), "dirty": bool(run("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def load_val_inputs(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    """Читает val_probs.npy запуска и val-метки/пациентов из кэша; проверяет согласованность. Только val."""
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    if metrics["train_setup"].get("limit_records"):
        raise ValueError(f"{run_dir.name}: запуск с --limit (проверочный), оценивать нельзя")
    data_cfg = yaml.safe_load((run_dir / "data_config.yaml").read_text(encoding="utf-8"))["data"]
    cache = _cache_dir(data_cfg)
    meta = json.loads((cache / "meta.json").read_text(encoding="utf-8"))
    fp_run = metrics.get("env", {}).get("cache_fingerprint")
    if fp_run and fp_run != meta["fingerprint"]:
        raise ValueError(f"{run_dir.name}: кэш данных изменился с момента обучения ({fp_run} != {meta['fingerprint']})")
    classes: list[str] = metrics["classes"]
    if classes != meta["classes"]:
        raise ValueError(f"Классы запуска {classes} != классы кэша {meta['classes']}")
    probs = np.load(run_dir / "val_probs.npy")
    y = np.load(cache / f"y_{EVAL_SPLIT}.npy")
    groups = np.load(cache / f"patient_id_{EVAL_SPLIT}.npy")
    if probs.shape != y.shape:
        raise ValueError(f"{run_dir.name}: val_probs {probs.shape} != y_val {y.shape}")
    return probs, y.astype(np.uint8), groups, classes, metrics


def _jsonable(o: Any) -> Any:
    """numpy -> обычные типы Python для json."""
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    return o


def evaluate_run(run_dir: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """Полная оценка запуска на val: метрики, T, пороги, честная cross-fit оценка, графики, логи."""
    t0 = time.perf_counter()
    probs, y, groups, classes, train_metrics = load_val_inputs(run_dir)
    eps, n_bins = cfg["probs_eps"], cfg["ece"]["n_bins"]
    probs64 = probs.astype(np.float64)

    params = fit_postprocess(probs, y, cfg)
    p_cal, _ = apply_postprocess(probs, params, eps)
    thr, temp = params["thresholds"], params["temperature"]

    ranking = multilabel_metrics(y, p_cal, classes)  # AUROC/PR-AUC: T не меняет ранжирование
    ops = threshold_metrics(y, p_cal, thr, classes)
    ops_05 = threshold_metrics(y, probs64, np.full(len(classes), 0.5), classes)
    calib_raw = calibration_metrics(y, probs64, classes, n_bins, eps)
    calib_cal = calibration_metrics(y, p_cal, classes, n_bins, eps)
    cross = crossfit_estimate(probs, y, groups, classes, cfg)

    fig_dir = _resolve(cfg["plots"]["figures_dir"]) / run_dir.name
    fig_dir.mkdir(parents=True, exist_ok=True)
    dpi, label = cfg["plots"]["dpi"], f"{run_dir.name} · val (фолд 9)"
    plot_roc(y, p_cal, ops, classes, f"ROC · {label}", fig_dir / "roc.png", dpi)
    plot_pr(y, p_cal, ops, classes, f"Precision–Recall · {label}", fig_dir / "pr.png", dpi)
    plot_confusion(ops, classes, f"Матрицы ошибок (пороги по F1 на val) · {label}", fig_dir / "confusion.png", dpi)
    plot_calibration(y, probs64, p_cal, calib_raw, calib_cal, classes, temp, n_bins, cfg["plots"]["min_bin_count"],
                     f"Калибровка · {label}", fig_dir / "calibration.png", dpi)

    result = {
        "run": run_dir.name,
        "evaluated_on": "val (strat_fold 9); test не использовался",
        "n_val": int(len(y)),
        "ranking_metrics": ranking,
        "postprocess": {
            "temperature_mode": cfg["temperature"]["mode"], "temperature": temp,
            "thresholds": dict(zip(classes, thr)),
        },
        "at_tuned_thresholds_in_sample": ops,
        "at_0.5_raw_probs": {"macro_f1": ops_05["macro_f1"], "micro_f1": ops_05["micro_f1"],
                             "f1_per_class": {c: v["f1"] for c, v in ops_05["per_class"].items()}},
        "calibration_raw": calib_raw,
        "calibration_after_temperature_in_sample": calib_cal,
        "cross_fitted_estimate": cross,
        "figures_dir": str(fig_dir.relative_to(PROJECT_ROOT)) if fig_dir.is_relative_to(PROJECT_ROOT) else str(fig_dir),
        "note": (
            "T и пороги подобраны на val и применены к тому же val (блок *_in_sample) — это оптимистичная оценка. "
            "Честнее блок cross_fitted_estimate (подбор на части val, оценка на остальной). "
            "Несмещённая оценка F1/ECE — только на test, где эти T и пороги применяются без пересчёта."
        ),
        "config": cfg,
        "train_run_val_macro_auroc": train_metrics["val"]["macro_auroc"],
        "time_sec": round(time.perf_counter() - t0, 1),
        "env": {"python": platform.python_version(), "numpy": np.__version__, "git": _git_state()},
    }
    (run_dir / "eval_val.json").write_text(json.dumps(_jsonable(result), ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "postprocess.json").write_text(json.dumps(_jsonable({
        "classes": classes, "temperature_mode": cfg["temperature"]["mode"], "temperature": temp, "thresholds": thr,
        "fitted_on": "val (strat_fold 9)", "probs_eps": eps,
    }), ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def format_report(r: dict[str, Any], classes: list[str]) -> str:
    """Табличка результатов для консоли."""
    ops, cr, cc = r["at_tuned_thresholds_in_sample"], r["calibration_raw"], r["calibration_after_temperature_in_sample"]
    head = f"{'класс':<6}{'порог':>7}{'P':>7}{'R':>7}{'F1':>7}{'AUROC':>8}{'PR-AUC':>8}{'ECE до':>8}{'ECE после':>11}"
    lines = [f"== {r['run']} ==", head, "-" * len(head)]
    for c in classes:
        o = ops["per_class"][c]
        lines.append(f"{c:<6}{o['threshold']:>7.3f}{o['precision']:>7.3f}{o['recall']:>7.3f}{o['f1']:>7.3f}"
                     f"{r['ranking_metrics']['auroc'][c]:>8.4f}{r['ranking_metrics']['pr_auc'][c]:>8.4f}"
                     f"{cr['per_class'][c]['ece']:>8.4f}{cc['per_class'][c]['ece']:>11.4f}")
    cf = r["cross_fitted_estimate"]
    lines += [
        f"macro AUROC {r['ranking_metrics']['macro_auroc']:.4f} | macro PR-AUC {r['ranking_metrics']['macro_pr_auc']:.4f}",
        f"macro F1: порог 0.5 {r['at_0.5_raw_probs']['macro_f1']:.4f} | подобранные пороги, in-sample "
        f"{ops['macro_f1']:.4f} (оптимистично) | cross-fit {cf['macro_f1']:.4f} (честно)",
        f"macro ECE: до {cr['macro']['ece']:.4f} -> после T {cc['macro']['ece']:.4f} (in-sample) / "
        f"{cf['ece_calibrated']['macro']['ece']:.4f} (cross-fit; до: {cf['ece_raw']['macro']['ece']:.4f}) | "
        f"T = {np.round(r['postprocess']['temperature'], 3).tolist()}",
    ]
    return "\n".join(lines)


def main() -> None:
    """CLI: оценить один или несколько запусков на val."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/evaluate.yaml")
    parser.add_argument("--run", action="append", required=True, help="папка запуска с val_probs.npy (можно несколько раз)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    for run in args.run:
        run_dir = _resolve(run)
        result = evaluate_run(run_dir, cfg)
        print(format_report(result, list(result["at_tuned_thresholds_in_sample"]["per_class"])))
        print(f"Графики: {result['figures_dir']}\n")


if __name__ == "__main__":
    main()
