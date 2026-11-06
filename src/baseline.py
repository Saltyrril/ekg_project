"""Бейзлайн без нейросетей: признаки ЭКГ + LightGBM one-vs-rest (CPU).

Обучение на train (strat_fold 1-8), метрики и ранняя остановка на val (фолд 9).
Test не загружается. Результаты пишутся в output_dir из конфига (metrics.json и др.).

Запуск:  python -m src.baseline --config configs/baseline.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import neurokit2 as nk
import numpy as np
import sklearn
import yaml

from src.data import PROJECT_ROOT, _resolve, load_config, load_splits
from src.evaluate import multilabel_metrics
from src.features import STAT_NAMES, coverage, get_features

logger = logging.getLogger(__name__)


def _git_state() -> dict[str, Any]:
    """Короткий хэш коммита и признак незакоммиченных изменений (для воспроизводимости)."""
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    try:
        return {"commit": run("rev-parse", "--short", "HEAD"), "dirty": bool(run("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def fit_ovr(
    X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, y_va: np.ndarray,
    classes: list[str], names: list[str], lgb_cfg: dict[str, Any], seed: int,
) -> tuple[np.ndarray, dict[str, int], dict[str, list[tuple[str, float]]], list[lgb.LGBMClassifier]]:
    """Один бинарный LightGBM на класс. Возвращает скоры val, best_iteration, топ-признаки по gain и модели."""
    scores = np.zeros(y_va.shape, dtype=np.float64)
    best_iter: dict[str, int] = {}
    top: dict[str, list[tuple[str, float]]] = {}
    models: list[lgb.LGBMClassifier] = []
    for i, c in enumerate(classes):
        model = lgb.LGBMClassifier(**lgb_cfg["params"], random_state=seed, importance_type="gain")
        model.fit(
            X_tr, y_tr[:, i],
            eval_X=X_va, eval_y=y_va[:, i],
            callbacks=[lgb.early_stopping(lgb_cfg["early_stopping_rounds"], verbose=False)],
        )
        scores[:, i] = model.predict_proba(X_va)[:, 1]
        best_iter[c] = int(model.best_iteration_)
        order = np.argsort(model.feature_importances_)[::-1][:10]
        top[c] = [(names[j], round(float(model.feature_importances_[j]), 1)) for j in order]
        models.append(model)
    return scores, best_iter, top, models


def select_variants(
    cov_tr: dict[str, float], names: list[str], n_leads: int, min_coverage: float
) -> tuple[dict[str, list[int]], list[str]]:
    """Наборы столбцов: stats_only и (если есть устойчивые ритм-признаки) stats_plus_neurokit.

    Устойчивость ритм-признака — доля валидных значений на TRAIN не ниже min_coverage. Возвращает (варианты, отброшенные).
    """
    n_stat = n_leads * len(STAT_NAMES)
    stat_cols = list(range(n_stat))
    rhythm_cols = [j for j in range(n_stat, len(names)) if cov_tr[names[j]] >= min_coverage]
    dropped = [names[j] for j in range(n_stat, len(names)) if j not in rhythm_cols]
    variants: dict[str, list[int]] = {"stats_only": stat_cols}
    if rhythm_cols:
        variants["stats_plus_neurokit"] = stat_cols + rhythm_cols
    return variants, dropped


def main() -> None:
    """CLI: посчитать признаки, обучить LightGBM, записать metrics.json."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    seed: int = cfg["seed"]
    np.random.seed(seed)
    data_cfg = load_config(cfg["data_config"])["data"]
    out_dir = _resolve(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    # Только train и val; сырые сигналы (признаки считаются по мВ, обучаемой нормализации нет).
    splits, meta = load_splits(data_cfg, splits=("train", "val"), normalize=False)
    tr, va = splits["train"], splits["val"]
    classes: list[str] = meta["classes"]
    leads: list[str] = meta["leads"]
    fs: int = data_cfg["sampling_rate"]
    logger.info("train: %d, val: %d, отведения: %s", len(tr.X), len(va.X), leads)

    feat_cfg = {"rhythm": cfg["features"]["rhythm"]}  # только то, что влияет на значения признаков
    feat_dir = _resolve(cfg["features"]["cache_dir"])
    t0 = time.perf_counter()
    F_tr, names = get_features("train", tr.X, leads, fs, feat_cfg, feat_dir, meta["fingerprint"], cfg["n_jobs_features"])
    F_va, names_va = get_features("val", va.X, leads, fs, feat_cfg, feat_dir, meta["fingerprint"], cfg["n_jobs_features"])
    assert names == names_va
    t_features = time.perf_counter() - t0

    # Устойчивость: покрытие считаем по train (метки и val не участвуют в отборе).
    cov_tr, cov_va = coverage(F_tr, names), coverage(F_va, names)
    n_stat = len(leads) * len(STAT_NAMES)
    variants, dropped = select_variants(cov_tr, names, len(leads), cfg["features"]["min_coverage"])
    logger.info("ритм-признаков оставлено %d, отброшено (покрытие < %.0f%% на train): %s",
                len(names) - n_stat - len(dropped), 100 * cfg["features"]["min_coverage"], dropped)

    results: dict[str, Any] = {}
    t_fit0 = time.perf_counter()
    for vname, cols in variants.items():
        vnames = [names[j] for j in cols]
        logger.info("[%s] %d признаков, обучаю %d моделей...", vname, len(cols), len(classes))
        scores, best_iter, top, _ = fit_ovr(
            F_tr[:, cols], tr.y, F_va[:, cols], va.y, classes, vnames, cfg["lightgbm"], seed
        )
        m = multilabel_metrics(va.y, scores, classes)
        results[vname] = {"n_features": len(cols), "val": m, "best_iteration": best_iter, "top_features_gain": top}
        logger.info("[%s] val macro AUROC = %.4f | %s", vname, m["macro_auroc"],
                    ", ".join(f"{c} {v:.3f}" for c, v in m["auroc"].items()))
    t_fit = time.perf_counter() - t_fit0

    primary = "stats_plus_neurokit" if "stats_plus_neurokit" in results else "stats_only"
    metrics = {
        "experiment": "baseline_lightgbm_ovr",
        "evaluated_on": "val (strat_fold 9); test не использовался",
        "primary_variant": primary,
        "val": results[primary]["val"],
        "variants": results,
        "n_train": int(len(tr.X)),
        "n_val": int(len(va.X)),
        "excluded_no_superclass": {k: v["excluded_no_superclass"] for k, v in meta["stats"]["splits"].items() if k != "test"},
        "classes": classes,
        "rhythm_features": {
            "lead": cfg["features"]["rhythm"]["lead"],
            "coverage_min_required": cfg["features"]["min_coverage"],
            "coverage_train": {names[j]: round(cov_tr[names[j]], 4) for j in range(n_stat, len(names))},
            "coverage_val": {names[j]: round(cov_va[names[j]], 4) for j in range(n_stat, len(names))},
            "dropped_as_unstable": dropped,
        },
        "note": (
            "Ранняя остановка идёт по val, поэтому val-метрики слегка оптимистичны; "
            "несмещённая оценка — только на test (фолд 10) в финале."
        ),
        "seed": seed,
        "time_sec": {
            "features": round(t_features, 1), "lightgbm_fit": round(t_fit, 1),
            "total": round(time.perf_counter() - t_start, 1),
        },
        "env": {
            "python": platform.python_version(), "lightgbm": lgb.__version__, "neurokit2": nk.__version__,
            "numpy": np.__version__, "scikit-learn": sklearn.__version__, "git": _git_state(),
            "cache_fingerprint": meta["fingerprint"],
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy(_resolve(args.config), out_dir / "config.yaml")
    (out_dir / "data_config.yaml").write_text(yaml.safe_dump({"data": data_cfg}, allow_unicode=True), encoding="utf-8")
    logger.info("Готово. [%s] val macro AUROC = %.4f. metrics.json: %s", primary, metrics["val"]["macro_auroc"], out_dir)


if __name__ == "__main__":
    main()
