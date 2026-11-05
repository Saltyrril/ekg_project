"""Анализ ошибок обученной модели на val (strat_fold 9): путаница классов, срезы, уверенные ошибки.

Здесь лежат функции для notebooks/02_error_analysis.ipynb (по правилам проекта код — в src/, ноутбук только
показывает результаты). Читает только val: `val_probs.npy` запуска и val-часть кэша данных. Test не используется.
Пороги и температура берутся из `postprocess.json` запуска — они подобраны на том же val, поэтому метрики
по порогам здесь in-sample (для разбора ошибок это допустимо, но как оценку качества их использовать нельзя).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data import _cache_dir, _resolve, load_config, load_metadata, load_splits
from src.evaluate import apply_postprocess, load_val_inputs

METRICS: tuple[str, ...] = ("auroc", "recall", "fpr")  # порядок оси метрик в бутстрепе


# ----------------------------------------------------------------------------- загрузка

@dataclass
class ValBundle:
    """Всё нужное для разбора ошибок, строго выровненное по записям val."""

    run: str
    classes: list[str]
    leads: list[str]
    X: np.ndarray            # (N, 12, T) сырые сигналы, мВ
    y: np.ndarray            # (N, C) uint8
    probs: np.ndarray        # (N, C) вероятности модели без калибровки
    p_cal: np.ndarray        # (N, C) после температуры
    pred: np.ndarray         # (N, C) uint8, p_cal >= порог класса
    temperature: np.ndarray  # (C,)
    thresholds: np.ndarray   # (C,)
    ecg_id: np.ndarray
    patient_id: np.ndarray
    meta: pd.DataFrame       # строки ptbxl_database.csv в том же порядке (индекс ecg_id)
    scp: pd.DataFrame        # диагностические строки scp_statements.csv


def load_bundle(cfg: dict[str, Any]) -> ValBundle:
    """Собирает ValBundle для запуска из конфига; проверяет согласованность меток, пациентов и фолда."""
    run_dir = _resolve(cfg["run"])
    probs, y, groups, classes, _ = load_val_inputs(run_dir)
    data_cfg = load_config(cfg["data_config"])["data"]
    data, meta_cache = load_splits(data_cfg, splits=("val",), normalize=False)  # test не грузим
    val = data["val"]
    if not (np.array_equal(val.y, y) and np.array_equal(val.patient_id, groups)):
        raise ValueError("Метки/пациенты val в кэше не совпали с теми, по которым оценивался запуск")
    pp = json.loads((run_dir / "postprocess.json").read_text(encoding="utf-8"))
    if pp["classes"] != classes:
        raise ValueError(f"Порядок классов в postprocess.json {pp['classes']} != {classes}")
    p_cal, pred = apply_postprocess(probs, pp, pp["probs_eps"])
    db, scp = load_metadata(_resolve(data_cfg["root"]))
    meta = db.loc[val.ecg_id].copy()
    if not (meta["strat_fold"] == 9).all() or not np.array_equal(meta["patient_id"].to_numpy(), val.patient_id):
        raise ValueError("Метаданные не соответствуют val (фолд 9) по записям/пациентам")
    return ValBundle(
        run=run_dir.name, classes=classes, leads=meta_cache["leads"], X=val.X, y=y, probs=probs, p_cal=p_cal,
        pred=pred, temperature=np.asarray(pp["temperature"]), thresholds=np.asarray(pp["thresholds"]),
        ecg_id=val.ecg_id, patient_id=val.patient_id, meta=meta, scp=scp,
    )


# ----------------------------------------------------------------------------- типы ошибок и путаница

def combo_names(y: np.ndarray, classes: list[str], empty: str = "(пусто)") -> np.ndarray:
    """Строки вида 'MI+CD' для каждой записи по multi-hot матрице."""
    return np.array(["+".join(c for c, v in zip(classes, row) if v) or empty for row in y])


def record_error_types(y: np.ndarray, pred: np.ndarray) -> pd.Series:
    """Тип ошибки по записи: точное совпадение / только пропуски / только лишние / и то и другое."""
    missed = ((y == 1) & (pred == 0)).any(axis=1)
    extra = ((y == 0) & (pred == 1)).any(axis=1)
    out = np.select([~missed & ~extra, missed & ~extra, ~missed & extra], ["верно", "только пропуск", "только лишний"],
                    default="пропуск и лишний")
    return pd.Series(out, name="тип").astype("category")


def single_label_confusion(y: np.ndarray, pred: np.ndarray, classes: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Для записей ровно с одной истинной меткой: доля записей, где предсказан каждый класс (+ 'ничего').

    Строки — истинный класс, столбцы — предсказанный; предсказаний может быть несколько, поэтому строка в сумме
    не обязана давать 100%. Диагональ — полнота на «чистых» одно-метковых записях.
    """
    one = y.sum(axis=1) == 1
    true_cls = y[one].argmax(axis=1)
    pr = pred[one]
    rows = {}
    for i, c in enumerate(classes):
        m = true_cls == i
        rows[c] = [pr[m][:, j].mean() for j in range(len(classes))] + [(pr[m].sum(axis=1) == 0).mean()]
    mat = pd.DataFrame(rows, index=classes + ["ничего"]).T
    return mat, pd.Series({c: int((true_cls == i).sum()) for i, c in enumerate(classes)}, name="n")


def error_sources(y: np.ndarray, pred: np.ndarray, classes: list[str], j: int, kind: str, top: int) -> pd.DataFrame:
    """Откуда берутся ошибки класса j.

    kind='fp': ложные срабатывания — какие истинные наборы меток у этих записей.
    kind='fn': пропуски — что модель предсказала вместо класса j.
    """
    if kind == "fp":
        sel = (pred[:, j] == 1) & (y[:, j] == 0)
        names = combo_names(y[sel], classes)
    elif kind == "fn":
        sel = (pred[:, j] == 0) & (y[:, j] == 1)
        names = combo_names(pred[sel], classes, empty="ничего")
    else:
        raise ValueError(kind)
    counts = pd.Series(names).value_counts()
    return pd.DataFrame({"n": counts.head(top), "доля": (counts / max(sel.sum(), 1)).head(top)})


# ----------------------------------------------------------------------------- бутстреп по пациентам и срезы

def safe_auroc(y: np.ndarray, p: np.ndarray) -> float:
    """AUROC или NaN, если в выборке один класс."""
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")


def cluster_bootstrap(stat_fn: Callable[[np.ndarray], np.ndarray], groups: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """Бутстреп с повторной выборкой пациентов (все записи пациента идут вместе).

    stat_fn получает индексы записей и возвращает вектор статистик; результат — (n_boot, k).
    """
    rng = np.random.default_rng(seed)
    _, inv = np.unique(groups, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    members = np.split(order, np.cumsum(np.bincount(inv))[:-1])
    n_groups = len(members)
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, n_groups, n_groups)
        out.append(stat_fn(np.concatenate([members[g] for g in pick])))
    return np.asarray(out, dtype=float)


def percentile_ci(boot: np.ndarray, level: float, min_valid_share: float) -> tuple[np.ndarray, np.ndarray]:
    """Перцентильный интервал по оси 0; NaN, если метрика определена в слишком малой доле реплик."""
    a = (1 - level) / 2 * 100
    valid = np.isfinite(boot).mean(axis=0)
    with np.errstate(all="ignore"):
        lo, hi = np.nanpercentile(boot, [a, 100 - a], axis=0)
    bad = valid < min_valid_share
    return np.where(bad, np.nan, lo), np.where(bad, np.nan, hi)


def _group_metrics(y: np.ndarray, p: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """(3, C+1): AUROC, полнота, FPR по классам + macro-среднее в последнем столбце (NaN, если хоть один NaN)."""
    n_cls = y.shape[1]
    out = np.full((len(METRICS), n_cls + 1), np.nan)
    for j in range(n_cls):
        pos, neg = y[:, j] == 1, y[:, j] == 0
        out[0, j] = safe_auroc(y[:, j], p[:, j])
        out[1, j] = pred[pos, j].mean() if pos.any() else np.nan
        out[2, j] = pred[neg, j].mean() if neg.any() else np.nan
    out[:, n_cls] = out[:, :n_cls].mean(axis=1)
    return out


def subgroup_bootstrap(
    y: np.ndarray, p: np.ndarray, pred: np.ndarray, masks: list[np.ndarray], groups: np.ndarray,
    n_boot: int, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Метрики по срезам: точечные (M, 3, C+1) и бутстреп-реплики (n_boot, M, 3, C+1).

    Бутстрепятся пациенты всей выборки, срез внутри реплики берётся заново, так что размер среза тоже случаен.
    """
    def stat(idx: np.ndarray) -> np.ndarray:
        res = []
        for m in masks:
            sel = idx[m[idx]]
            res.append(_group_metrics(y[sel], p[sel], pred[sel]) if len(sel) else np.full((len(METRICS), y.shape[1] + 1), np.nan))
        return np.stack(res).ravel()

    all_idx = np.arange(len(y))
    shape = (len(masks), len(METRICS), y.shape[1] + 1)
    point = stat(all_idx).reshape(shape)
    boot = cluster_bootstrap(stat, groups, n_boot, seed).reshape((n_boot, *shape))
    return point, boot


def subgroup_frame(
    point: np.ndarray, boot: np.ndarray, names: list[str], masks: list[np.ndarray], y: np.ndarray,
    classes: list[str], level: float, min_valid_share: float, min_pos: int,
) -> pd.DataFrame:
    """Длинная таблица: срез × класс × метрика с интервалом.

    Скрываем (NaN) то, что считается по слишком малому числу записей: AUROC — если положительных или
    отрицательных < min_pos, полнота — если положительных < min_pos, FPR — если отрицательных < min_pos.
    Для macro метрика скрыта, если скрыта хотя бы у одного класса.
    """
    lo, hi = percentile_ci(boot, level, min_valid_share)
    n_cls = len(classes)
    rows = []
    for m_i, (name, mask) in enumerate(zip(names, masks)):
        n_pos_all = y[mask].sum(axis=0)
        n_neg_all = mask.sum() - n_pos_all
        small = {"auroc": (n_pos_all < min_pos) | (n_neg_all < min_pos), "recall": n_pos_all < min_pos,
                 "fpr": n_neg_all < min_pos}
        for j, cls in enumerate(classes + ["macro"]):
            for k_i, metric in enumerate(METRICS):
                hide = small[metric].any() if j == n_cls else small[metric][j]
                rows.append({"группа": name, "n": int(mask.sum()), "класс": cls, "метрика": metric,
                             "n_pos": int(n_pos_all[j]) if j < n_cls else int(mask.sum()),
                             "значение": np.nan if hide else point[m_i, k_i, j],
                             "lo": np.nan if hide else lo[m_i, k_i, j], "hi": np.nan if hide else hi[m_i, k_i, j]})
    return pd.DataFrame(rows)


def difference_ci(point: np.ndarray, boot: np.ndarray, a: int, b: int, level: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Разность метрик срезов a − b: точка, нижняя и верхняя границы (по оси (3, C+1))."""
    d_point = point[a] - point[b]
    d_boot = boot[:, a] - boot[:, b]
    q = (1 - level) / 2 * 100
    with np.errstate(all="ignore"):
        lo, hi = np.nanpercentile(d_boot, [q, 100 - q], axis=0)
    return d_point, lo, hi


# ----------------------------------------------------------------------------- уверенные ошибки

def confident_errors(y: np.ndarray, p_cal: np.ndarray, j: int, kind: str) -> np.ndarray:
    """Индексы записей по убыванию «уверенности» ошибки в классе j.

    fp: класса нет в метках, сортировка по p убыванию; fn: класс есть, сортировка по p возрастанию.
    """
    if kind == "fp":
        idx = np.flatnonzero(y[:, j] == 0)
        return idx[np.argsort(-p_cal[idx, j], kind="stable")]
    if kind == "fn":
        idx = np.flatnonzero(y[:, j] == 1)
        return idx[np.argsort(p_cal[idx, j], kind="stable")]
    raise ValueError(kind)


def label_support(codes: pd.Series, scp: pd.DataFrame, classes: list[str]) -> np.ndarray:
    """(N, C): максимальный likelihood среди кодов записи, относящихся к классу; NaN, если кодов класса нет."""
    code_to_class = scp["diagnostic_class"].dropna().to_dict()
    out = np.full((len(codes), len(classes)), np.nan)
    for r, d in enumerate(codes):
        for code, lik in d.items():
            cls = code_to_class.get(code)
            if cls in classes:
                j = classes.index(cls)
                out[r, j] = lik if np.isnan(out[r, j]) else max(out[r, j], lik)
    return out


def format_codes(codes: dict[str, float]) -> str:
    """'IMI:35, ABQRS:0' — коды SCP записи с likelihood в исходном виде."""
    return ", ".join(f"{k}:{v:g}" for k, v in codes.items())


def noise_flags(meta: pd.DataFrame) -> pd.Series:
    """True, если у записи есть хотя бы одна пометка о шуме/дрейфе/плохих электродах/пейсмекере в метаданных PTB-XL."""
    cols = ["baseline_drift", "static_noise", "burst_noise", "electrodes_problems", "pacemaker"]
    return meta[cols].notna().any(axis=1)


# ----------------------------------------------------------------------------- график ЭКГ

def plot_ecg(sig: np.ndarray, leads: list[str], title: str, sample_rate: int = 100, dpi: int = 100) -> Any:
    """12 отведений в 2 столбца (конечностные | грудные) на сетке: 0.2 с и 0.5 мВ (крупная), 0.04 с и 0.1 мВ (мелкая).

    Каждое отведение сдвинуто на свою медиану (только для показа), масштаб по мВ общий на всю запись.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    centered = sig - np.median(sig, axis=1, keepdims=True)
    lim = float(np.clip(np.ceil(2 * 1.1 * np.abs(centered).max()) / 2, 1.0, 4.0))
    t = np.arange(sig.shape[1]) / sample_rate
    fig, axes = plt.subplots(6, 2, figsize=(12, 7.2), sharex=True, dpi=dpi)
    for k, lead in enumerate(leads):
        ax = axes[k % 6, k // 6]
        ax.plot(t, centered[k], color="#0b0b0b", lw=0.9)
        ax.set_ylim(-lim, lim)
        ax.set_xlim(0, t[-1])
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.xaxis.set_minor_locator(MultipleLocator(0.04))
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.grid(which="major", color="#e8a0a0", lw=0.6)
        ax.grid(which="minor", color="#f6d6d6", lw=0.3)
        ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
        ax.set_ylabel(lead, rotation=0, ha="right", va="center", fontsize=9)
        for s in ax.spines.values():
            s.set_visible(False)
    for ax in axes[-1]:
        ax.tick_params(axis="x", labelbottom=True, labelsize=8)
        ax.set_xticks(np.arange(0, t[-1] + 0.01, 1.0))
        ax.set_xlabel("с", fontsize=8)
    fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    fig.text(0.99, 0.005, f"по вертикали ±{lim:g} мВ (сетка 0.5 мВ), по горизонтали 0.2 с; отведения сдвинуты на медиану",
             fontsize=7, color="#898781", ha="right")
    fig.tight_layout(rect=(0, 0.015, 1, 0.97))
    return fig


# ----------------------------------------------------------------------------- таблицы и графики для ноутбука

def noise_detail(meta: pd.DataFrame) -> pd.Series:
    """Какие именно пометки о шуме/дрейфе/электродах/пейсмекере есть у записи (пустая строка — нет пометок)."""
    cols = ["baseline_drift", "static_noise", "burst_noise", "electrodes_problems", "pacemaker"]
    return meta[cols].apply(lambda r: ", ".join(c for c in cols if pd.notna(r[c])), axis=1)


def age_label(age: float) -> str:
    """Возраст для показа: 300 в PTB-XL означает «старше 89» (анонимизация)."""
    return "90+" if age > 89 else f"{age:g}"


def error_table(b: ValBundle, idx: np.ndarray, j: int, max_report: int = 110) -> pd.DataFrame:
    """Таблица записей с метаданными и кодами SCP; p — откалиброванная вероятность класса j и всех классов."""
    m = b.meta.iloc[idx]
    true_c, pred_c = combo_names(b.y[idx], b.classes), combo_names(b.pred[idx], b.classes, empty="ничего")
    p_all = [" ".join(f"{c} {p:.2f}" for c, p in zip(b.classes, row)) for row in b.p_cal[idx]]
    return pd.DataFrame({
        "ecg_id": b.ecg_id[idx], "возраст": [age_label(a) for a in m["age"]], "пол": np.where(m["sex"] == 0, "М", "Ж"),
        "истинно": true_c, "предсказано": pred_c, f"p({b.classes[j]})": b.p_cal[idx, j].round(3), "p по классам": p_all,
        "коды SCP (likelihood)": [format_codes(d) for d in m["scp_codes"]],
        "пометки о шуме": noise_detail(m).to_numpy(),
        "отчёт (исходный текст)": [str(s)[:max_report] for s in m["report"]],
    }).reset_index(drop=True)


def plot_metric_by_group(
    frame: pd.DataFrame, metric: str, order: list[str], classes: list[str], title: str, ylabel: str,
    colors: list[str], ylim: tuple[float, float] | None = None, connect: bool = True, note: str = "",
) -> Any:
    """Метрика с доверительным интервалом по срезам: по панели на класс и macro. Скрытые (NaN) точки не рисуются."""
    import matplotlib.pyplot as plt

    names = classes + ["macro"]
    ncols = 3
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.1 * nrows + 0.6), sharey=True)
    x = np.arange(len(order))
    for ax, name in zip(np.ravel(axes), names):
        sub = frame[(frame["класс"] == name) & (frame["метрика"] == metric)].set_index("группа").loc[order]
        col = colors[names.index(name)] if name != "macro" else "#0b0b0b"
        v, lo, hi = sub["значение"].to_numpy(), sub["lo"].to_numpy(), sub["hi"].to_numpy()
        ax.errorbar(x, v, yerr=[v - lo, hi - v], fmt="-o" if connect else "o", color=col, ms=5, capsize=3, lw=1.6)
        ax.set_xticks(x, [f"{g}\nn={n}" for g, n in zip(order, sub["n"])], fontsize=8)
        ax.set_title(name)
        ax.set_xlim(-0.5, len(order) - 0.5)
        if ylim:
            ax.set_ylim(*ylim)
    for ax in np.ravel(axes)[len(names):]:
        ax.set_visible(False)
    for ax in np.ravel(axes)[::ncols]:
        ax.set_ylabel(ylabel)
    fig.suptitle(title, fontsize=11, x=0.01, ha="left")
    if note:
        fig.text(0.01, 0.005, note, fontsize=8, color="#898781")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    return fig
