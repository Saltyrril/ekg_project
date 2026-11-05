"""Карты значимости (Integrated Gradients) обученной модели на val (strat_fold 9).

Что делает:
  1. Загружает модель запуска (best.pt) и проверяет, что она воспроизводит сохранённые val_probs.npy.
  2. Считает IG по логиту класса: (N, 12, T) вклад каждого отсчёта каждого отведения в ответ модели.
  3. Рисует по `examples.per_class` случайных верно найденных одно-меточных записей на класс.
  4. Количественно проверяет, куда смотрит модель, на ВСЕХ верно найденных записях класса: доля |IG| в окнах
     QRS / ST–T / P–PR / остальное (по R-зубцам отведения II), доли по отведениям, тест стирания участков
     и два контроля (та же архитектура со случайными весами; тривиальная карта |x|).

Ограничения (важно при чтении результатов): IG — это атрибуция к конкретному baseline, а не «объяснение» причин;
окна от R-зубца грубые (100 Гц); стирание участков выводит сигнал за пределы обучающего распределения.
Test (фолд 10) не читается: загружается только val.

Запуск:  python -m src.explain --config configs/explain.yaml
Проверка: python -m src.explain --config configs/explain.yaml --limit 15 --out-dir tmp/explain_smoke
Пишет:   <out-dir>/*.png и <out-dir>/explain_val.json (по умолчанию results/figures/<run>/explain)
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from scipy.ndimage import uniform_filter1d
from scipy.stats import spearmanr

from src.data import PROJECT_ROOT, _resolve, lead_indices, load_config, load_metadata, load_splits, normalize_
from src.error_analysis import cluster_bootstrap, format_codes, percentile_ci
from src.evaluate import apply_postprocess, load_val_inputs
from src.models import build_model

logger = logging.getLogger(__name__)

WINDOWS: tuple[str, ...] = ("qrs", "st_t", "p", "other")
WINDOW_LABELS: dict[str, str] = {
    "qrs": "QRS (R ± 60 мс)", "st_t": "ST–T (60–400 мс после R)", "p": "P/PR (250–60 мс до R)", "other": "остальное",
}
WINDOW_COLORS: dict[str, str] = {"qrs": "#e69f00", "st_t": "#009e73", "p": "#cc79a7", "other": "#bbbbbb"}
METHODS: tuple[str, ...] = ("ig", "rand", "amp")
METHOD_LABELS: dict[str, str] = {
    "ig": "IG, обученная модель", "rand": "IG, случайные веса", "amp": "контроль |x| (амплитуда)",
}
METHOD_COLORS: dict[str, str] = {"ig": "#0072b2", "rand": "#999999", "amp": "#d55e00"}


# ----------------------------------------------------------------------------- модель

def load_run_model(
    run_dir: Path, device: torch.device, all_leads: list[str]
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Собирает модель запуска и грузит best.pt. Возвращает (model в eval-режиме, info).

    У запусков до эксперимента с отведениями поля `leads` в metrics.json нет — тогда использовались все 12.
    """
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    leads: list[str] = metrics.get("leads") or all_leads
    classes: list[str] = metrics["classes"]
    model = build_model(cfg["model"]["name"], len(leads), len(classes), **cfg["model"]["params"])
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True))
    model.to(device).eval()
    return model, {"model_cfg": cfg["model"], "leads": leads, "classes": classes, "metrics": metrics}


def random_model_like(model_cfg: dict[str, Any], n_leads: int, n_classes: int, seed: int, device: torch.device) -> torch.nn.Module:
    """Та же архитектура со случайными весами (контроль Adebayo et al.: карта не должна быть такой же)."""
    torch.manual_seed(seed)
    model = build_model(model_cfg["name"], n_leads, n_classes, **model_cfg["params"])
    return model.to(device).eval()


@torch.no_grad()
def predict_logits(model: torch.nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    """Логиты (N, C) в fp32, без autocast."""
    out = [model(torch.from_numpy(X[i:i + batch_size]).to(device)).float().cpu() for i in range(0, len(X), batch_size)]
    return torch.cat(out).numpy()


# ----------------------------------------------------------------------------- Integrated Gradients

def integrated_gradients(
    model: torch.nn.Module, X: np.ndarray, target: int, steps: int, max_points: int, device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """IG по логиту класса `target` с нулевым baseline, правило средних точек.

    attr = x * mean_alpha( d logit(alpha * x) / dx ). X: (N, C, T) z-нормализованные сигналы.
    Возвращает (attr (N, C, T), delta (N,), f(x)-f(0) (N,)); delta = sum(attr) - (f(x) - f(0)) —
    ошибка полноты (completeness): у точного интеграла она равна 0.
    Модель в eval-режиме (BatchNorm по накопленным статистикам), поэтому записи в батче не влияют друг на друга.
    """
    n, c, t = X.shape
    per = max(1, max_points // steps)
    alphas = ((torch.arange(steps, device=device, dtype=torch.float32) + 0.5) / steps).view(1, steps, 1, 1)
    with torch.no_grad():
        f0 = float(model(torch.zeros(1, c, t, device=device))[0, target])
    attrs, deltas, diffs = [], [], []
    for s in range(0, n, per):
        xb = torch.from_numpy(X[s:s + per]).to(device)
        b = xb.shape[0]
        pts = (alphas * xb.unsqueeze(1)).reshape(b * steps, c, t).detach().requires_grad_(True)
        grad = torch.autograd.grad(model(pts)[:, target].sum(), pts)[0].view(b, steps, c, t).mean(dim=1)
        a = xb * grad
        with torch.no_grad():
            diff = model(xb)[:, target] - f0
        attrs.append(a.detach().cpu())
        deltas.append((a.sum(dim=(1, 2)) - diff).detach().cpu())
        diffs.append(diff.cpu())
    return torch.cat(attrs).numpy(), torch.cat(deltas).numpy(), torch.cat(diffs).numpy()


def deletion_drop(
    model: torch.nn.Module, X: np.ndarray, attr: np.ndarray, target: int, block: int, k: int, n_random: int,
    rng: np.random.Generator, device: torch.device,
) -> dict[str, np.ndarray]:
    """Падение логита класса при стирании k блоков (ставим baseline = 0): самые значимые / случайные / наименее значимые.

    Значимость блока — сумма |attr| по всем отведениям и отсчётам блока. Блоки целые и по всем отведениям сразу,
    чтобы не плодить одиночные «иголки» в сигнале. Возвращает {"top","random","bottom"} -> (N,) падения (>0 = логит упал).
    """
    n, _, t = X.shape
    if t % block:
        raise ValueError(f"Длина {t} не делится на block_samples={block}")
    nb = t // block
    score = np.abs(attr).sum(axis=1).reshape(n, nb, block).sum(axis=2)
    order = np.argsort(score, axis=1)
    base = predict_logits(model, X, device)[:, target]

    def drop(blocks: np.ndarray) -> np.ndarray:
        sel = np.zeros((n, nb), dtype=bool)
        np.put_along_axis(sel, blocks, True, axis=1)
        mask = np.repeat(sel, block, axis=1)[:, None, :]
        return base - predict_logits(model, np.where(mask, 0.0, X).astype(np.float32), device)[:, target]

    rand = np.mean([drop(rng.permuted(np.tile(np.arange(nb), (n, 1)), axis=1)[:, :k]) for _ in range(n_random)], axis=0)
    return {"top": drop(order[:, -k:]), "random": rand, "bottom": drop(order[:, :k])}


# ----------------------------------------------------------------------------- окна вокруг R-зубцов

def detect_r_peaks(sig: np.ndarray, fs: int, cfg: dict[str, Any]) -> np.ndarray | None:
    """R-зубцы по одному отведению (neurokit2). None, если ударов мало или медианный RR неправдоподобен."""
    import neurokit2 as nk

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clean = nk.ecg_clean(sig.astype(np.float64), sampling_rate=fs, method="neurokit")
            _, info = nk.ecg_peaks(clean, sampling_rate=fs, method="neurokit")
        r = np.asarray(info["ECG_R_Peaks"], dtype=np.int64)
    except Exception as e:  # noqa: BLE001 — сбой одной записи не должен валить весь прогон
        logger.debug("neurokit2: %s: %s", type(e).__name__, e)
        return None
    if len(r) < cfg["min_beats"]:
        return None
    lo, hi = cfg["rr_s"]
    return r if lo <= float(np.median(np.diff(r))) / fs <= hi else None


def window_masks(r: np.ndarray, length: int, fs: int, w: dict[str, float]) -> dict[str, np.ndarray]:
    """Булевы маски длины T: QRS, ST–T, P/PR, остальное (взаимоисключающие; приоритет QRS > ST–T > P/PR)."""
    hw, st_end, p_start = (round(w[k] * fs) for k in ("qrs_half_width_s", "st_t_end_s", "p_start_s"))
    qrs, st, p = (np.zeros(length, dtype=bool) for _ in range(3))
    for ri in r:
        qrs[max(0, ri - hw):ri + hw + 1] = True
        st[ri + hw + 1:ri + st_end + 1] = True
        p[max(0, ri - p_start):max(0, ri - hw)] = True
    st &= ~qrs
    p &= ~(qrs | st)
    return {"qrs": qrs, "st_t": st, "p": p, "other": ~(qrs | st | p)}


def window_shares(profile: np.ndarray, masks: dict[str, np.ndarray]) -> np.ndarray:
    """Доля суммы профиля (T,) в каждом окне, порядок WINDOWS."""
    total = profile.sum()
    return np.array([profile[masks[k]].sum() / total if total > 0 else np.nan for k in WINDOWS])


# ----------------------------------------------------------------------------- графики

def _plt() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_example(
    sig: np.ndarray, attr: np.ndarray, leads: list[str], r: np.ndarray | None, masks: dict[str, np.ndarray] | None,
    title: str, fs: int, dpi: int, smooth: int = 5,
) -> Any:
    """Верх: профиль sum_leads |IG| по времени (окна и R-зубцы). Ниже: отведения; фон — знаковый IG.

    Фон ТОЛЬКО для показа сглажен скользящей суммой в `smooth` отсчётов (5 = 50 мс): поотсчётный знак IG
    скачет красный/синий и нечитаем. Сумма (не среднее) сохраняет суммарный вклад окна. Расчёты и статистики
    идут по несглаженному IG.

    Красный фон — отсчёт повышает логит класса («за»), синий — понижает («против»). Шкала цвета симметрична и
    ОДНА на все отведения записи (99.5-й перцентиль |IG|), поэтому яркость отведений сравнима между собой.
    """
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Patch
    from matplotlib.ticker import MultipleLocator

    plt = _plt()
    n_leads, length = sig.shape
    n_rows = (n_leads + 1) // 2
    t = np.arange(length) / fs
    centered = sig - np.median(sig, axis=1, keepdims=True)
    lim = float(np.clip(np.ceil(2 * 1.1 * np.abs(centered).max()) / 2, 1.0, 4.0))
    shown = uniform_filter1d(attr, size=smooth, axis=1, mode="nearest") * smooth
    vmax = float(np.percentile(np.abs(shown), 99.5)) or 1.0
    lead_share = np.abs(attr).sum(axis=1) / max(np.abs(attr).sum(), 1e-12)

    fig = plt.figure(figsize=(12, 9.6), dpi=dpi)
    gs = GridSpec(1 + n_rows, 2, figure=fig, height_ratios=[1.7] + [1] * n_rows, hspace=0.22, wspace=0.1)
    top = fig.add_subplot(gs[0, :])
    profile = np.abs(attr).sum(axis=0)
    if masks is not None:
        for k in ("qrs", "st_t", "p"):
            top.fill_between(t, 0, 1, where=masks[k], transform=top.get_xaxis_transform(), color=WINDOW_COLORS[k],
                             alpha=0.28, lw=0)
    top.plot(t, profile, color="#111111", lw=1.0)
    if r is not None:
        top.plot(r / fs, profile[r], "v", color="#111111", ms=4)
    top.set_xlim(0, t[-1])
    top.set_ylim(bottom=0)
    top.set_ylabel("Σ|IG| по отведениям", fontsize=9)
    top.tick_params(labelsize=8)
    handles = [Patch(color=WINDOW_COLORS[k], alpha=0.5, label=WINDOW_LABELS[k]) for k in ("qrs", "st_t", "p")]
    if masks is not None:
        top.legend(handles=handles, loc="upper right", fontsize=7, ncol=3, frameon=False)
    else:
        top.text(0.99, 0.95, "R-зубцы не найдены: окна не показаны", transform=top.transAxes, ha="right", va="top",
                 fontsize=7, color="#898781")

    for k, lead in enumerate(leads):
        ax = fig.add_subplot(gs[1 + k % n_rows, k // n_rows])
        ax.imshow(shown[k][None, :], aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest",
                  extent=(0, t[-1] + 1 / fs, -lim, lim), alpha=0.6, zorder=0)
        ax.plot(t, centered[k], color="#0b0b0b", lw=0.8, zorder=2)
        ax.set_xlim(0, t[-1])
        ax.set_ylim(-lim, lim)
        ax.xaxis.set_major_locator(MultipleLocator(1.0))
        ax.xaxis.set_minor_locator(MultipleLocator(0.2))
        ax.grid(which="minor", axis="x", color="#ffffff", lw=0.3, alpha=0.6)
        ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
        ax.set_ylabel(f"{lead}\n{lead_share[k]:.0%}", rotation=0, ha="right", va="center", fontsize=8)
        for s in ax.spines.values():
            s.set_visible(False)
        if k % n_rows == n_rows - 1:
            ax.tick_params(axis="x", labelbottom=True, labelsize=8)
    fig.suptitle(title, fontsize=9.5, x=0.01, ha="left")
    fig.text(0.99, 0.004,
             "красный — отсчёт повышает логит класса («за»), синий — понижает; подпись у отведения — доля его |IG|; "
             f"по вертикали ±{lim:g} мВ; фон сглажен на {1000 * smooth // fs} мс; отведения сдвинуты на медиану", fontsize=6.5, color="#898781", ha="right")
    fig.subplots_adjust(left=0.07, right=0.99, top=0.93, bottom=0.04)
    return fig


def plot_windows_summary(stats: dict[str, Any], classes: list[str], dpi: int) -> Any:
    """Доля |IG| в каждом окне по классам: обученная модель, случайные веса, |x|; линия — доля времени окна."""
    plt = _plt()
    fig, axes = plt.subplots(1, len(WINDOWS), figsize=(14, 3.8), dpi=dpi, sharey=True)
    x = np.arange(len(classes))
    wd = 0.26
    for ax, wname in zip(axes, WINDOWS):
        for m_i, m in enumerate(METHODS):
            val = np.array([stats[c]["windows"][wname][m]["mean"] for c in classes])
            lo = np.array([stats[c]["windows"][wname][m]["ci"][0] for c in classes])
            hi = np.array([stats[c]["windows"][wname][m]["ci"][1] for c in classes])
            err = np.vstack([np.nan_to_num(val - lo), np.nan_to_num(hi - val)])
            ax.bar(x + (m_i - 1) * wd, val, wd, yerr=err, color=METHOD_COLORS[m], error_kw={"lw": 0.8, "capsize": 2},
                   label=METHOD_LABELS[m])
        for i, c in enumerate(classes):  # доля времени = ожидание при равномерной карте
            ts = stats[c]["windows"][wname]["time_share"]
            ax.hlines(ts, i - 0.42, i + 0.42, color="#111111", ls="--", lw=1.2)
        ax.set_xticks(x, classes, fontsize=9)
        ax.set_title(WINDOW_LABELS[wname], fontsize=9.5)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", color="#e5e5e5", lw=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("доля Σ|attr|", fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([0], [0], color="#111111", ls="--", lw=1.2))
    labels.append("доля времени окна (равномерная карта)")
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle("Куда приходится |attribution|: окна вокруг R-зубцов, верно найденные записи val (95% ДИ, бутстреп по пациентам)",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    return fig


def plot_leads_heatmap(stats: dict[str, Any], classes: list[str], leads: list[str], dpi: int) -> Any:
    """Слева: средняя доля |IG| по отведениям. Справа: она же минус доля |x| (контроль «просто амплитуда»).

    Звёздочка справа — 95% ДИ (бутстреп по пациентам) разности не включает 0.
    """
    plt = _plt()
    share = np.array([[stats[c]["lead_share"][ld]["mean"] for ld in leads] for c in classes])
    diff = np.array([[stats[c]["lead_share_minus_abs_x"][ld]["mean"] for ld in leads] for c in classes])
    sig = np.array([[np.prod(np.sign(stats[c]["lead_share_minus_abs_x"][ld]["ci"])) > 0 for ld in leads] for c in classes])
    fig, axes = plt.subplots(1, 2, figsize=(15, 3.4), dpi=dpi)
    im0 = axes[0].imshow(share, cmap="Blues", vmin=0, vmax=max(float(share.max()), 2.0 / len(leads)), aspect="auto")
    lim = float(np.abs(diff).max())
    im1 = axes[1].imshow(diff, cmap="PuOr", vmin=-lim, vmax=lim, aspect="auto")
    for ax, im, mat in ((axes[0], im0, share), (axes[1], im1, diff)):
        ax.set_xticks(range(len(leads)), leads, fontsize=9)
        ax.set_yticks(range(len(classes)), classes, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        for i in range(len(classes)):
            for j in range(len(leads)):
                dark = abs(mat[i, j]) > 0.6 * np.abs(mat).max()
                txt = f"{mat[i, j]:.0%}" if mat is share else f"{100 * mat[i, j]:+.1f}{'*' if sig[i, j] else ''}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="white" if dark else "#111111",
                        fontweight="bold" if mat is diff and sig[i, j] else "normal")
    axes[0].set_title(f"Средняя доля |IG| по отведениям (равномерно ≈ {1 / len(leads):.0%})", fontsize=10, loc="left")
    axes[1].set_title("Доля |IG| минус доля |x|, п.п. (* — 95% ДИ не включает 0)", fontsize=10, loc="left")
    fig.tight_layout()
    return fig


def plot_faithfulness(stats: dict[str, Any], classes: list[str], dpi: int) -> Any:
    """Падение логита класса при стирании 10% записи: самые значимые / случайные / наименее значимые блоки."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9.5, 4.0), dpi=dpi)
    x = np.arange(len(classes))
    wd = 0.17
    spec = (("top", "топ по IG (обученная модель)", "#0072b2"), ("top_amp", "топ по |x| (амплитуда)", "#d55e00"),
            ("top_randmodel", "топ по IG случайной модели", "#7f7f7f"), ("random", "случайные блоки", "#cccccc"),
            ("bottom", "наименее значимые по IG", "#e69f00"))
    for i, (key, label, color) in enumerate(spec):
        val = np.array([stats[c]["deletion_drop_logit"][key]["mean"] for c in classes])
        lo = np.array([stats[c]["deletion_drop_logit"][key]["ci"][0] for c in classes])
        hi = np.array([stats[c]["deletion_drop_logit"][key]["ci"][1] for c in classes])
        err = np.vstack([np.nan_to_num(val - lo), np.nan_to_num(hi - val)])
        ax.bar(x + (i - 2) * wd, val, wd, yerr=err, color=color, error_kw={"lw": 0.8, "capsize": 2}, label=label)
    ax.axhline(0, color="#111111", lw=0.8)
    ax.set_xticks(x, classes)
    ax.set_ylabel("падение логита класса", fontsize=9)
    ax.set_title("Стирание 10% записи (5 блоков по 0.2 с) по карте IG", fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", color="#e5e5e5", lw=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------- агрегаты по классу

def _nanmean(a: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=0)


def _summ(col: int, mean: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict[str, Any]:
    return {"mean": float(mean[col]), "ci": [float(lo[col]), float(hi[col])]}


def class_stats(
    rows: np.ndarray, names: list[str], groups: np.ndarray, bcfg: dict[str, Any], seed: int,
) -> dict[str, dict[str, Any]]:
    """Среднее по записям и перцентильный 95% ДИ (бутстреп по пациентам) для каждого столбца `rows` (n, k)."""
    mean = _nanmean(rows)
    boot = cluster_bootstrap(lambda idx: _nanmean(rows[idx]), groups, bcfg["n_boot"], seed)
    lo, hi = percentile_ci(boot, bcfg["ci_level"], bcfg["min_valid_share"])
    return {n: _summ(i, mean, lo, hi) for i, n in enumerate(names)}


# ----------------------------------------------------------------------------- пайплайн

def _git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    try:
        return {"commit": run("rev-parse", "--short", "HEAD"), "dirty": bool(run("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def run_explain(cfg: dict[str, Any], out_dir: Path, limit: int | None) -> dict[str, Any]:
    """Весь анализ; возвращает словарь, который сохраняется в explain_val.json."""
    t_start = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = _resolve(cfg["run"])
    rng = np.random.default_rng(cfg["seed"])
    ig_cfg, fs = cfg["ig"], 100

    # --- данные: только val; сырые сигналы (для графиков и R-зубцов) и z-нормализованные (для модели)
    data_cfg = load_config(cfg["data_config"])["data"]
    fs = data_cfg["sampling_rate"]
    splits, cache_meta = load_splits(data_cfg, splits=("val",), normalize=False)
    val = splits["val"]
    raw = val.X
    mean, std = np.asarray(cache_meta["norm_mean"], np.float32), np.asarray(cache_meta["norm_std"], np.float32)
    model, info = load_run_model(run_dir, device, cache_meta["leads"])
    classes, leads = info["classes"], info["leads"]
    lead_idx = lead_indices(cache_meta["leads"], leads)
    Xn = np.ascontiguousarray(normalize_(raw.copy(), mean, std)[:, lead_idx])
    r_lead = cache_meta["leads"].index(cfg["rpeaks"]["lead"])

    # --- проверка: модель воспроизводит сохранённые val_probs (иначе карты были бы про другую модель)
    probs_saved, y, groups, classes_chk, _ = load_val_inputs(run_dir)
    if classes_chk != classes or not np.array_equal(y, val.y):
        raise ValueError("Классы/метки val в запуске и в кэше не совпали")
    probs_now = 1.0 / (1.0 + np.exp(-predict_logits(model, Xn, device)))
    diff = np.abs(probs_now - probs_saved)
    logger.info("Воспроизведение val_probs: max |Δp| = %.4f, mean = %.5f", diff.max(), diff.mean())
    if diff.max() > cfg["repro_max_abs_diff"]:
        raise ValueError(f"Модель не воспроизводит val_probs.npy: max |Δp| = {diff.max():.4f}")
    pp = json.loads((run_dir / "postprocess.json").read_text(encoding="utf-8"))
    p_cal, pred = apply_postprocess(probs_saved, pp, pp["probs_eps"])

    rand_model = random_model_like(info["model_cfg"], len(leads), len(classes), cfg["controls"]["random_model_seed"], device)
    db, _ = load_metadata(_resolve(data_cfg["root"]))
    meta_val = db.loc[val.ecg_id]

    # --- R-зубцы и окна по запросу (кэш по индексу записи)
    mask_cache: dict[int, tuple[np.ndarray, dict[str, np.ndarray]] | None] = {}

    def get_masks(i: int) -> tuple[np.ndarray, dict[str, np.ndarray]] | None:
        if i not in mask_cache:
            r = detect_r_peaks(raw[i, r_lead], fs, cfg["rpeaks"])
            mask_cache[i] = None if r is None else (r, window_masks(r, raw.shape[2], fs, cfg["windows"]))
        return mask_cache[i]

    # --- примеры: случайные верно найденные одно-меточные записи
    n_ex = cfg["examples"]["per_class"]
    examples: list[tuple[int, int]] = []  # (класс, индекс записи)
    for c in range(len(classes)):
        cand = np.flatnonzero((y.sum(axis=1) == 1) & (y[:, c] == 1) & (pred[:, c] == 1))
        if len(cand) < n_ex:
            logger.warning("%s: одно-меточных верно найденных записей %d < %d", classes[c], len(cand), n_ex)
        examples += [(c, int(i)) for i in rng.choice(cand, size=min(n_ex, len(cand)), replace=False)]

    plt = _plt()
    out_dir.mkdir(parents=True, exist_ok=True)
    dpi = cfg["plots"]["dpi"]
    n_boot_seed = cfg["seed"]
    result: dict[str, Any] = {"classes": {}, "examples": []}
    ig_errors: list[np.ndarray] = []
    thr = np.asarray(pp["thresholds"])

    # --- примеры: IG + рисунки
    for c, i in examples:
        a, d, df = integrated_gradients(model, Xn[i:i + 1], c, ig_cfg["steps"], ig_cfg["max_points_per_batch"], device)
        mk = get_masks(i)
        codes = format_codes(meta_val["scp_codes"].iloc[i])
        title = (f"{classes[c]} | ecg_id {val.ecg_id[i]} | p(калибр.) = {p_cal[i, c]:.2f} (порог {thr[c]:.2f}) | "
                 f"коды SCP: {codes}")
        fig = plot_example(raw[i][lead_idx], a[0], leads, mk[0] if mk else None, mk[1] if mk else None, title, fs, dpi, cfg["plots"]["smooth_samples"])
        path = out_dir / f"example_{classes[c]}_{val.ecg_id[i]}.png"
        fig.savefig(path)
        plt.close(fig)
        prof = np.abs(a[0]).sum(axis=0)
        result["examples"].append({
            "class": classes[c], "ecg_id": int(val.ecg_id[i]), "p_cal": float(p_cal[i, c]), "scp_codes": codes,
            "ig_rel_error": float(abs(d[0]) / max(abs(df[0]), 1e-6)),
            "window_shares": dict(zip(WINDOWS, window_shares(prof, mk[1]).tolist())) if mk else None,
            "lead_share": dict(zip(leads, (np.abs(a[0]).sum(axis=1) / np.abs(a[0]).sum()).tolist())),
            "figure": path.name,
        })

    # --- количественный анализ: все верно найденные записи класса
    fcfg = cfg["faithfulness"]
    for c, cname in enumerate(classes):
        idx = np.flatnonzero((y[:, c] == 1) & (pred[:, c] == 1))
        if limit:
            idx = idx[:limit]
        if len(idx) < 5:
            logger.warning("%s: верно найденных записей %d — статистика не считается", cname, len(idx))
            continue
        t0 = time.perf_counter()
        X_c = Xn[idx]
        a_ig, d_ig, df_ig = integrated_gradients(model, X_c, c, ig_cfg["steps"], ig_cfg["max_points_per_batch"], device)
        a_rd, _, _ = integrated_gradients(rand_model, X_c, c, ig_cfg["steps"], ig_cfg["max_points_per_batch"], device)
        rel = np.abs(d_ig) / np.maximum(np.abs(df_ig), 1e-6)
        ig_errors.append(rel)
        if np.median(rel) > ig_cfg["max_rel_error_warn"]:
            logger.warning("%s: медианная ошибка полноты IG %.1f%% — увеличьте ig.steps", cname, 100 * np.median(rel))
        dkw = (c, fcfg["block_samples"], fcfg["delete_blocks"], fcfg["n_random_draws"], rng, device)
        dele = deletion_drop(model, X_c, a_ig, *dkw)
        # контроли: те же 5 блоков, но выбранные по |x| и по IG случайной модели (это «top» из их вызова)
        dele["top_amp"] = deletion_drop(model, X_c, X_c, *dkw)["top"]
        dele["top_randmodel"] = deletion_drop(model, X_c, a_rd, *dkw)["top"]

        prof = {"ig": np.abs(a_ig).sum(axis=1), "rand": np.abs(a_rd).sum(axis=1), "amp": np.abs(X_c).sum(axis=1)}
        cols: list[str] = []
        rows: list[np.ndarray] = []
        nan_w = np.full(len(WINDOWS), np.nan)
        shares = {m: [] for m in METHODS}
        time_share = []
        for j, i in enumerate(idx):
            mk = get_masks(int(i))
            for m in METHODS:
                shares[m].append(window_shares(prof[m][j], mk[1]) if mk else nan_w)
            time_share.append(np.array([mk[1][k].mean() for k in WINDOWS]) if mk else nan_w)
        for m in METHODS:
            sh = np.array(shares[m])
            rows += [sh[:, k] for k in range(len(WINDOWS))]
            cols += [f"win:{m}:{w}" for w in WINDOWS]
        ts = np.array(time_share)
        rows += [ts[:, k] for k in range(len(WINDOWS))]
        cols += [f"time:{w}" for w in WINDOWS]
        lead_sh = np.abs(a_ig).sum(axis=2) / np.maximum(np.abs(a_ig).sum(axis=(1, 2))[:, None], 1e-12)
        lead_amp = np.abs(X_c).sum(axis=2) / np.abs(X_c).sum(axis=(1, 2))[:, None]  # контроль: доли |x| по отведениям
        lead_rd = np.abs(a_rd).sum(axis=2) / np.maximum(np.abs(a_rd).sum(axis=(1, 2))[:, None], 1e-12)
        for tag, arr in (("lead", lead_sh), ("leadamp", lead_amp), ("leadrand", lead_rd), ("leaddiff", lead_sh - lead_amp)):
            rows += [arr[:, k] for k in range(len(leads))]
            cols += [f"{tag}:{ld}" for ld in leads]
        for key in ("top", "top_amp", "top_randmodel", "random", "bottom"):
            rows.append(dele[key])
            cols.append(f"del:{key}")
        rows.append(dele["top"] - dele["random"])
        cols.append("del:top_minus_random")
        for m, other in (("rand", "ig"), ("amp", "ig")):  # согласие профилей по времени (Спирмен)
            rows.append(np.array([spearmanr(prof["ig"][j], prof[m][j])[0] for j in range(len(idx))]))
            cols.append(f"spearman:ig_vs_{m}")
        M = np.vstack(rows).T
        cs = class_stats(M, cols, groups[idx], cfg["bootstrap"], n_boot_seed)
        n_valid = int(np.isfinite(M[:, cols.index("win:ig:qrs")]).sum())
        result["classes"][cname] = {
            "n_records": int(len(idx)), "n_with_rpeaks": n_valid,
            "windows": {w: {**{m: cs[f"win:{m}:{w}"] for m in METHODS}, "time_share": cs[f"time:{w}"]["mean"]}
                        for w in WINDOWS},
            "lead_share": {ld: cs[f"lead:{ld}"] for ld in leads},
            "lead_share_abs_x": {ld: cs[f"leadamp:{ld}"] for ld in leads},
            "lead_share_random_model": {ld: cs[f"leadrand:{ld}"] for ld in leads},
            "lead_share_minus_abs_x": {ld: cs[f"leaddiff:{ld}"] for ld in leads},
            "deletion_drop_logit": {k: cs[f"del:{k}"] for k in ("top", "top_amp", "top_randmodel", "random", "bottom", "top_minus_random")},
            "spearman_time_profile": {"ig_vs_random_model": cs["spearman:ig_vs_rand"], "ig_vs_abs_x": cs["spearman:ig_vs_amp"]},
            "ig_rel_error_median": float(np.median(rel)),
        }
        logger.info("%s: n=%d (R-зубцы найдены у %d), %.0f с", cname, len(idx), n_valid, time.perf_counter() - t0)

    all_err = np.concatenate(ig_errors) if ig_errors else np.array([np.nan])
    result["ig_completeness"] = {
        "rel_error_median": float(np.median(all_err)), "rel_error_p95": float(np.percentile(all_err, 95)),
        "note": "|sum(attr) - (f(x)-f(0))| / |f(x)-f(0)| по всем записям количественного анализа",
    }
    done = [c for c in classes if c in result["classes"]]
    for name, fig in (
        ("summary_windows.png", plot_windows_summary(result["classes"], done, dpi)),
        ("summary_leads.png", plot_leads_heatmap(result["classes"], done, leads, dpi)),
        ("summary_faithfulness.png", plot_faithfulness(result["classes"], done, dpi)),
    ):
        fig.savefig(out_dir / name)
        plt.close(fig)

    result |= {
        "run": run_dir.name, "evaluated_on": "val (strat_fold 9); test не использовался", "method": "Integrated Gradients",
        "target": "логит класса (до sigmoid)", "baseline": "ноль в z-пространстве (среднее train по отведению)",
        "repro_val_probs": {"max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean())},
        "limit": limit, "config": cfg, "seed": cfg["seed"], "time_sec": round(time.perf_counter() - t_start, 1),
        "env": {"python": platform.python_version(), "torch": torch.__version__, "device": str(device),
                "git": _git_state(), "cache_fingerprint": cache_meta["fingerprint"]},
    }
    (out_dir / "explain_val.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def format_report(r: dict[str, Any]) -> str:
    """Короткая сводка в консоль."""
    lines = [f"Запуск {r['run']}; IG, ошибка полноты: медиана {r['ig_completeness']['rel_error_median']:.2%}, "
             f"p95 {r['ig_completeness']['rel_error_p95']:.2%}", ""]
    lines.append(f"{'класс':<6}{'n':>5}  " + "  ".join(f"{w:^23}" for w in ("QRS", "ST–T", "P/PR")))
    lines.append(f"{'':<11}  " + "  ".join(f"{'IG / случ. / |x| / время':^23}" for _ in range(3)))
    for c, s in r["classes"].items():
        cells = []
        for w in ("qrs", "st_t", "p"):
            v = s["windows"][w]
            cells.append(f"{v['ig']['mean']:.2f}/{v['rand']['mean']:.2f}/{v['amp']['mean']:.2f}/{v['time_share']:.2f}")
        lines.append(f"{c:<6}{s['n_records']:>5}  " + "  ".join(f"{x:^23}" for x in cells))
    lines += ["", "Стирание 10% записи: падение логита (топ IG / топ |x| / топ IG случ.модели / случайные / наименее значимые)"]
    for c, s in r["classes"].items():
        d = s["deletion_drop_logit"]
        vals = " / ".join(f"{d[k]['mean']:5.2f}" for k in ("top", "top_amp", "top_randmodel", "random", "bottom"))
        lines.append(f"  {c:<5} {vals}   "
                     f"(top-random {d['top_minus_random']['mean']:+.2f}, 95% ДИ {d['top_minus_random']['ci'][0]:+.2f}..{d['top_minus_random']['ci'][1]:+.2f})")
    return "\n".join(lines)


def main() -> None:
    """CLI."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/explain.yaml")
    parser.add_argument("--limit", type=int, help="взять первые N верно найденных записей на класс (для проверки)")
    parser.add_argument("--out-dir", help="куда писать (по умолчанию results/figures/<run>/explain)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    out_dir = _resolve(args.out_dir) if args.out_dir else _resolve("results/figures") / Path(cfg["run"]).name / "explain"
    r = run_explain(cfg, out_dir, args.limit)
    print(format_report(r))
    print(f"\nГотово за {r['time_sec']} с: {out_dir}")


if __name__ == "__main__":
    main()
