"""Подготовка PTB-XL: сигналы через wfdb, multi-hot метки по суперклассам,
разбиение по strat_fold, нормализация по train, кэш в .npy.

Запуск:  python -m src.data --config configs/data.yaml [--rebuild]
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wfdb
import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS: tuple[str, ...] = ("train", "val", "test")
# Разбиение по правилам проекта; отклонение от него — повод для предупреждения.
CANONICAL_FOLDS: dict[str, list[int]] = {"train": list(range(1, 9)), "val": [9], "test": [10]}
CACHE_VERSION = 1


@dataclass
class SplitData:
    """Одна выборка: X (N, 12, T) float32, y (N, C) uint8 multi-hot, идентификаторы записей."""

    X: np.ndarray
    y: np.ndarray
    ecg_id: np.ndarray
    patient_id: np.ndarray


# ----------------------------------------------------------------------------- конфиг

def load_config(path: str | Path) -> dict[str, Any]:
    """Читает YAML-конфиг."""
    with open(_resolve(path), encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve(path: str | Path) -> Path:
    """Относительные пути считаются от корня проекта, а не от текущей папки."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _validate_folds(folds: dict[str, list[int]]) -> None:
    """Проверяет, что фолды не пересекаются; предупреждает об отличии от правил проекта."""
    seen: dict[int, str] = {}
    for split in SPLITS:
        for fold in folds[split]:
            if fold in seen:
                raise ValueError(f"Фолд {fold} одновременно в '{seen[fold]}' и '{split}'")
            seen[fold] = split
    if {k: sorted(v) for k, v in folds.items()} != CANONICAL_FOLDS:
        logger.warning("Разбиение отличается от правил проекта (train 1-8, val 9, test 10): %s", folds)


def _fingerprint(cfg: dict[str, Any]) -> str:
    """Хэш параметров, от которых зависит содержимое кэша."""
    keys = ("sampling_rate", "classes", "min_likelihood", "folds", "normalization")
    payload = json.dumps({k: cfg[k] for k in keys} | {"v": CACHE_VERSION}, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def _cache_dir(cfg: dict[str, Any]) -> Path:
    return _resolve(cfg["cache_dir"]) / f"ptbxl_{cfg['sampling_rate']}hz"


# ----------------------------------------------------------------------------- метки

def load_metadata(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Читает ptbxl_database.csv и диагностические строки scp_statements.csv."""
    db_path = root / "ptbxl_database.csv"
    if not db_path.exists():
        raise FileNotFoundError(
            f"Не найден {db_path}. Скачайте PTB-XL с PhysioNet и распакуйте так, "
            f"чтобы ptbxl_database.csv лежал в data.root из конфига."
        )
    db = pd.read_csv(db_path, index_col="ecg_id")
    db["scp_codes"] = db["scp_codes"].apply(ast.literal_eval)  # строка "{'NORM': 100.0}" -> dict
    scp = pd.read_csv(root / "scp_statements.csv", index_col=0)
    scp = scp[scp["diagnostic"] == 1]  # ритм/форма не имеют diagnostic_class
    return db, scp


def build_labels(
    db: pd.DataFrame, scp: pd.DataFrame, classes: list[str], min_likelihood: float
) -> np.ndarray:
    """Multi-hot матрица (N, C): класс = 1, если хоть один код записи относится к нему."""
    code_to_class: dict[str, str] = scp["diagnostic_class"].dropna().to_dict()
    unknown = set(classes) - set(code_to_class.values())
    if unknown:
        raise ValueError(f"Классов {unknown} нет в scp_statements.diagnostic_class")
    class_idx = {c: i for i, c in enumerate(classes)}
    y = np.zeros((len(db), len(classes)), dtype=np.uint8)
    for row, codes in enumerate(db["scp_codes"]):
        for code, likelihood in codes.items():
            cls = code_to_class.get(code)
            if cls in class_idx and likelihood >= min_likelihood:
                y[row, class_idx[cls]] = 1
    return y


# ----------------------------------------------------------------------------- сигналы

def _read_record(path: Path) -> tuple[np.ndarray, list[str]]:
    """Читает одну запись wfdb -> (12, T) float32 и имена отведений."""
    signal, fields = wfdb.rdsamp(str(path))
    return np.ascontiguousarray(signal.T, dtype=np.float32), fields["sig_name"]


def load_signals(root: Path, filenames: list[str], num_workers: int) -> tuple[np.ndarray, list[str]]:
    """Загружает записи в массив (N, 12, T); проверяет одинаковую форму и порядок отведений."""
    out: np.ndarray | None = None
    leads: list[str] = []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        for i, (sig, names) in enumerate(pool.map(_read_record, (root / f for f in filenames))):
            if out is None:
                out = np.empty((len(filenames), *sig.shape), dtype=np.float32)
                leads = names
            if sig.shape != out.shape[1:] or names != leads:
                raise ValueError(f"{filenames[i]}: форма {sig.shape} / отведения {names} не совпадают")
            out[i] = sig
            if (i + 1) % 2000 == 0:
                logger.info("  прочитано %d/%d (%.0f с)", i + 1, len(filenames), time.perf_counter() - t0)
    assert out is not None, "пустой список файлов"
    if not np.isfinite(out).all():
        raise ValueError("В сигналах есть NaN/inf")
    return out, leads


# ----------------------------------------------------------------------------- нормализация

def compute_norm_stats(X_train: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Mean/std формы (12,) только по train. Считаем по отведению, чтобы не плодить копию 1 ГБ."""
    n_leads = X_train.shape[1]
    if mode == "per_lead":
        mean = np.array([X_train[:, c].mean(dtype=np.float64) for c in range(n_leads)])
        std = np.array([X_train[:, c].std(dtype=np.float64) for c in range(n_leads)])
    elif mode == "global":
        mean = np.full(n_leads, X_train.mean(dtype=np.float64))
        std = np.full(n_leads, X_train.std(dtype=np.float64))
    else:
        raise ValueError(f"Неизвестный режим нормализации: {mode}")
    return mean.astype(np.float32), np.maximum(std, 1e-6).astype(np.float32)


def normalize_(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Нормализация на месте: (X - mean) / std по оси отведений."""
    X -= mean[None, :, None]
    X /= std[None, :, None]
    return X


# ----------------------------------------------------------------------------- статистика

def compute_stats(
    labels: dict[str, np.ndarray],
    classes: list[str],
    excluded: dict[str, int],
    n_total: int,
    patient_overlap: dict[str, int],
) -> dict[str, Any]:
    """Размеры выборок, число примеров каждого класса, исключённые записи."""
    splits: dict[str, Any] = {}
    for split, y in labels.items():
        n_labels = y.sum(axis=1)
        splits[split] = {
            "n": int(len(y)),
            "excluded_no_superclass": int(excluded[split]),
            "class_counts": {c: int(v) for c, v in zip(classes, y.sum(axis=0))},
            "labels_per_record": {k: int((n_labels == k).sum()) for k in range(1, int(n_labels.max()) + 1)},
        }
    return {
        "n_total_in_csv": n_total,
        "n_excluded_no_superclass": int(sum(excluded.values())),
        "n_kept": int(sum(len(y) for y in labels.values())),
        "patient_overlap": patient_overlap,
        "splits": splits,
    }


def format_stats(stats: dict[str, Any], classes: list[str]) -> str:
    """Табличка для вывода в консоль."""
    head = f"{'split':<6}{'n':>7}" + "".join(f"{c:>14}" for c in classes) + f"{'исключено':>11}"
    lines = [head, "-" * len(head)]
    for split, s in stats["splits"].items():
        cells = "".join(
            f"{s['class_counts'][c]:>7} ({100 * s['class_counts'][c] / s['n']:4.1f}%)" for c in classes
        )
        lines.append(f"{split:<6}{s['n']:>7}{cells}{s['excluded_no_superclass']:>11}")
    lines.append("")
    lines.append(
        f"Всего в CSV: {stats['n_total_in_csv']}, оставлено: {stats['n_kept']}, "
        f"исключено без суперкласса: {stats['n_excluded_no_superclass']}"
    )
    for split, s in stats["splits"].items():
        dist = ", ".join(f"{k} метк.: {v}" for k, v in s["labels_per_record"].items())
        lines.append(f"  {split}: {dist}")
    overlap = ", ".join(f"{k}: {v}" for k, v in stats["patient_overlap"].items())
    lines.append(f"Общих пациентов между выборками: {overlap}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- пайплайн

def build_cache(cfg: dict[str, Any]) -> dict[str, Any]:
    """Строит кэш: метки, разбиение, сигналы по выборкам, статистики нормализации."""
    root = _resolve(cfg["root"])
    classes: list[str] = cfg["classes"]
    folds: dict[str, list[int]] = cfg["folds"]
    _validate_folds(folds)
    file_col = {100: "filename_lr", 500: "filename_hr"}[cfg["sampling_rate"]]

    db, scp = load_metadata(root)
    y_all = build_labels(db, scp, classes, cfg["min_likelihood"])
    has_label = y_all.any(axis=1)

    fold_to_split = {f: s for s, fs in folds.items() for f in fs}
    split_col = db["strat_fold"].map(fold_to_split)
    if split_col.isna().any():
        logger.warning("Записей вне всех выборок (фолд не указан в конфиге): %d", int(split_col.isna().sum()))

    out_dir = _cache_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels: dict[str, np.ndarray] = {}
    patients: dict[str, set[int]] = {}
    excluded: dict[str, int] = {}
    leads: list[str] = []
    mean = std = np.empty(0)

    for split in SPLITS:  # train первым: по нему считаются статистики нормализации
        in_split = (split_col == split).to_numpy()
        excluded[split] = int((in_split & ~has_label).sum())
        mask = in_split & has_label
        sub = db[mask]
        logger.info("%s: %d записей (исключено без суперкласса: %d), читаю сигналы...",
                    split, len(sub), excluded[split])
        X, leads = load_signals(root, sub[file_col].tolist(), cfg["num_workers"])
        if split == "train":
            mean, std = compute_norm_stats(X, cfg["normalization"])
        labels[split] = y_all[mask]
        patients[split] = set(sub["patient_id"].astype(int))
        np.save(out_dir / f"X_{split}.npy", X)  # сырые сигналы (мВ); нормализация при загрузке
        np.save(out_dir / f"y_{split}.npy", labels[split])
        np.save(out_dir / f"ecg_id_{split}.npy", sub.index.to_numpy(dtype=np.int64))
        np.save(out_dir / f"patient_id_{split}.npy", sub["patient_id"].to_numpy(dtype=np.int64))
        del X

    overlap = {
        f"{a}-{b}": len(patients[a] & patients[b])
        for i, a in enumerate(SPLITS) for b in SPLITS[i + 1:]
    }
    if any(overlap.values()):
        raise RuntimeError(f"Пациенты пересекаются между выборками: {overlap}")

    stats = compute_stats(labels, classes, excluded, len(db), overlap)
    meta = {
        "fingerprint": _fingerprint(cfg),
        "config": cfg,
        "classes": classes,
        "leads": leads,
        "norm_mean": mean.tolist(),
        "norm_std": std.tolist(),
        "stats": stats,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    # meta.json пишется последним: его наличие = кэш полный
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def prepare_data(cfg: dict[str, Any], rebuild: bool = False) -> dict[str, Any]:
    """Возвращает meta кэша, при необходимости (нет / устарел / rebuild) пересобирает его."""
    meta_path = _cache_dir(cfg) / "meta.json"
    if not rebuild and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("fingerprint") == _fingerprint(cfg):
            logger.info("Кэш актуален: %s", meta_path.parent)
            return meta
        logger.info("Параметры конфига изменились — пересобираю кэш")
    t0 = time.perf_counter()
    meta = build_cache(cfg)
    logger.info("Кэш собран за %.0f с: %s", time.perf_counter() - t0, meta_path.parent)
    return meta


def load_splits(
    cfg: dict[str, Any],
    splits: tuple[str, ...] = ("train", "val"),
    normalize: bool = True,
) -> tuple[dict[str, SplitData], dict[str, Any]]:
    """Загружает выборки из кэша. Test не грузится по умолчанию: только для финальной оценки."""
    meta = prepare_data(cfg)
    d = _cache_dir(cfg)
    mean = np.asarray(meta["norm_mean"], dtype=np.float32)
    std = np.asarray(meta["norm_std"], dtype=np.float32)
    data: dict[str, SplitData] = {}
    for split in splits:
        X = np.load(d / f"X_{split}.npy")
        if normalize:
            normalize_(X, mean, std)  # всегда статистики train, в том числе для val/test
        data[split] = SplitData(
            X=X,
            y=np.load(d / f"y_{split}.npy"),
            ecg_id=np.load(d / f"ecg_id_{split}.npy"),
            patient_id=np.load(d / f"patient_id_{split}.npy"),
        )
    return data, meta


def main() -> None:
    """CLI: собрать кэш и напечатать статистику."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--rebuild", action="store_true", help="пересобрать кэш принудительно")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)["data"]
    meta = prepare_data(cfg, rebuild=args.rebuild)
    print()
    print(format_stats(meta["stats"], meta["classes"]))
    print(f"Нормализация ({cfg['normalization']}, по train): "
          f"mean={np.round(meta['norm_mean'], 4).tolist()}, std={np.round(meta['norm_std'], 4).tolist()}")

    stats_path = _resolve(cfg["stats_path"])
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(meta["stats"], ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Статистика сохранена: %s", stats_path)


if __name__ == "__main__":
    main()
