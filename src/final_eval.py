"""ФИНАЛЬНАЯ оценка лучшей модели и бейзлайна на test (strat_fold 10) с bootstrap-ДИ по пациентам.

Правила (CLAUDE.md): на test ничего не подбирается. T и пороги берутся готовыми с val:
  * лучшая модель — из `<run>/postprocess.json` (подобраны src/evaluate.py на фолде 9);
  * бейзлайн LightGBM (модель не сохранялась) переобучается на train тем же кодом и seed с ранней остановкой
    по val; воспроизведение проверяется по val macro AUROC; его T и пороги подбираются на val той же процедурой.
Все проверки делаются ДО чтения test; при любом расхождении скрипт падает, не открыв test.
Боевой прогон выполняется один раз: если `<output_dir>/metrics.json` уже есть, скрипт отказывается работать.

Проверка конвейера (test не читается, оценка на val):  python -m src.final_eval --dry-run
Единственный прогон на test:                          python -m src.final_eval
Пишет: metrics.json, table.md, config.yaml, baseline_postprocess.json, probs_*.npy, bootstrap_replicates.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import platform
import shutil
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import sklearn
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

from src.baseline import fit_ovr, select_variants
from src.data import PROJECT_ROOT, SplitData, _resolve, lead_indices, load_config, load_splits, normalize_
from src.error_analysis import cluster_bootstrap, percentile_ci
from src.evaluate import _git_state, _jsonable, apply_postprocess, calibration_metrics, fit_postprocess, threshold_metrics
from src.explain import load_run_model
from src.features import coverage, get_features
from src.train import AMP_DTYPES, predict, set_seed

logger = logging.getLogger(__name__)

MODEL_KEY, BASE_KEY = "model", "baseline"
HEADLINE: tuple[tuple[str, str], ...] = (
    ("macro_auroc", "macro AUROC"),
    ("macro_pr_auc", "macro PR-AUC"),
    ("macro_f1", "macro F1 (пороги с val)"),
    ("micro_f1", "micro F1 (пороги с val)"),
    ("macro_recall", "macro полнота"),
    ("macro_specificity", "macro специфичность"),
    ("macro_ece", "macro ECE (после T с val)"),
    ("macro_ece_raw", "macro ECE (без T)"),
    ("macro_brier", "macro Brier (после T)"),
)
# Значения столбца val, посчитанные на отложенных частях val (подбор T/порогов на других частях).
VAL_HELD_OUT_KEYS: frozenset[str] = frozenset({"macro_f1", "micro_f1", "macro_ece"})


# ----------------------------------------------------------------------------- метрики

def metric_vector(
    y: np.ndarray, p_cal: np.ndarray, p_raw: np.ndarray, thr: np.ndarray, classes: list[str], n_bins: int, eps: float,
) -> dict[str, float]:
    """Все метрики одной выборки в фиксированном порядке ключей (для точечной оценки и реплик бутстрепа).

    Пороги `thr` применяются к откалиброванным вероятностям `p_cal` (так они и подбирались на val).
    AUROC/PR-AUC класса = NaN, если в выборке нет обоих значений метки.
    """
    out: dict[str, float] = {}
    for name, fn in (("auroc", roc_auc_score), ("pr_auc", average_precision_score)):
        vals = []
        for i, c in enumerate(classes):
            ok = 0 < y[:, i].sum() < len(y)
            vals.append(float(fn(y[:, i], p_cal[:, i])) if ok else math.nan)
            out[f"{name}_{c}"] = vals[-1]
        out[f"macro_{name}"] = float(np.mean(vals))
    ops = threshold_metrics(y, p_cal, thr, classes)
    for k in ("f1", "precision", "recall", "specificity"):
        for c in classes:
            out[f"{k}_{c}"] = ops["per_class"][c][k]
        out[f"macro_{k}"] = float(np.mean([ops["per_class"][c][k] for c in classes]))
    out["micro_f1"] = ops["micro_f1"]
    cal = calibration_metrics(y, p_cal, classes, n_bins, eps)
    raw = calibration_metrics(y, p_raw, classes, n_bins, eps)
    for c in classes:
        out[f"ece_{c}"] = cal["per_class"][c]["ece"]
    out["macro_ece"] = cal["macro"]["ece"]
    out["macro_ece_raw"] = raw["macro"]["ece"]
    out["macro_brier"] = cal["macro"]["brier"]
    out["macro_nll"] = cal["macro"]["nll"]
    return out


def _nan_to_none(o: Any) -> Any:
    """NaN -> null, чтобы metrics.json был валидным JSON."""
    if isinstance(o, dict):
        return {k: _nan_to_none(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_nan_to_none(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def _sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()[:12]


# ----------------------------------------------------------------------------- модель и бейзлайн

def predict_run(run_dir: Path, X: np.ndarray, y: np.ndarray, all_leads: list[str], device: torch.device) -> np.ndarray:
    """Вероятности модели запуска на нормализованных сигналах X (N, 12, T), тем же AMP, что при обучении."""
    model, info = load_run_model(run_dir, device, all_leads)
    tc = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))["train"]
    amp = AMP_DTYPES[tc["amp_dtype"]] if device.type == "cuda" else None
    idx = lead_indices(all_leads, info["leads"])
    probs, _ = predict(
        model, torch.from_numpy(np.ascontiguousarray(X[:, idx])), torch.from_numpy(y).float(),
        batch_size=tc["eval_batch_size"], criterion=nn.BCEWithLogitsLoss(), device=device, amp_dtype=amp,
    )
    return probs


def baseline_features(bcfg: dict[str, Any], split: str, X: np.ndarray, meta: dict[str, Any], fs: int) -> tuple[np.ndarray, list[str]]:
    """Признаки бейзлайна по сырому сигналу (мВ), с тем же кэшем, что у src/baseline.py."""
    return get_features(
        split, X, meta["leads"], fs, {"rhythm": bcfg["features"]["rhythm"]},
        _resolve(bcfg["features"]["cache_dir"]), meta["fingerprint"], bcfg["n_jobs_features"],
    )


def fit_baseline(
    bcfg: dict[str, Any], tr: SplitData, va: SplitData, meta: dict[str, Any], fs: int,
) -> tuple[list[lgb.LGBMClassifier], list[int], np.ndarray, dict[str, Any]]:
    """Повторяет основной вариант src/baseline.py: train для обучения, val для ранней остановки. Test не нужен."""
    np.random.seed(bcfg["seed"])
    F_tr, names = baseline_features(bcfg, "train", tr.X, meta, fs)
    F_va, names_va = baseline_features(bcfg, "val", va.X, meta, fs)
    assert names == names_va
    variants, dropped = select_variants(coverage(F_tr, names), names, len(meta["leads"]), bcfg["features"]["min_coverage"])
    primary = "stats_plus_neurokit" if "stats_plus_neurokit" in variants else "stats_only"
    cols = variants[primary]
    scores, best_iter, _, models = fit_ovr(
        F_tr[:, cols], tr.y, F_va[:, cols], va.y, meta["classes"], [names[j] for j in cols], bcfg["lightgbm"], bcfg["seed"],
    )
    return models, cols, scores, {"variant": primary, "n_features": len(cols), "dropped": dropped,
                                  "best_iteration": best_iter, "feature_names": names}


def baseline_predict(models: list[lgb.LGBMClassifier], F: np.ndarray) -> np.ndarray:
    """Скоры one-vs-rest моделей -> (N, C)."""
    return np.stack([m.predict_proba(F)[:, 1] for m in models], axis=1)


# ----------------------------------------------------------------------------- отчёт

def _fmt(point: float | None, ci: list[float | None] | None) -> str:
    if point is None:
        return "—"
    if not ci or ci[0] is None:
        return f"{point:.3f}"
    return f"{point:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def format_table(res: dict[str, Any], classes: list[str]) -> str:
    """Markdown-таблицы: итоговые метрики с ДИ, разница моделей, метрики по классам."""
    m, b, d = res["models"][MODEL_KEY], res["models"][BASE_KEY], res["difference_model_minus_baseline"]
    split = res["evaluated_on"]
    lvl = int(round(100 * res["bootstrap"]["level"]))
    lines = [
        f"# Финальная оценка: {split}",
        "",
        f"Записей: {res['n_records']}, пациентов: {res['n_patients']}. ДИ {lvl}%: перцентильный bootstrap по пациентам, "
        f"{res['bootstrap']['n_boot']} реплик. Пороги и T подобраны на val (фолд 9) и применены без пересчёта.",
        "",
        f"| Метрика | {m['name']} | {b['name']} | разница (модель − бейзлайн) | {m['name']}, val (фолд 9) |",
        "|---|---|---|---|---|",
    ]
    for key, label in HEADLINE:
        val_ref = m["val_reference"].get(key)
        mark = "\\*" if key in VAL_HELD_OUT_KEYS else ""
        lines.append(
            f"| {label} | {_fmt(m['point'][key], m['ci'][key])} | {_fmt(b['point'][key], b['ci'][key])} | "
            f"{_fmt(d[key]['point'], d[key]['ci'])} | {'—' if val_ref is None else f'{val_ref:.3f}{mark}'} |"
        )
    lines += [
        "",
        f"## По классам ({res['eval_split']}, ДИ {lvl}%)",
        "",
        f"| Класс | доля | AUROC {m['name']} | AUROC {b['name']} | ΔAUROC | PR-AUC {m['name']} | F1 {m['name']} | F1 {b['name']} |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in classes:
        lines.append(
            f"| {c} | {res['prevalence'][c]:.3f} | {_fmt(m['point'][f'auroc_{c}'], m['ci'][f'auroc_{c}'])} | "
            f"{_fmt(b['point'][f'auroc_{c}'], b['ci'][f'auroc_{c}'])} | "
            f"{_fmt(d[f'auroc_{c}']['point'], d[f'auroc_{c}']['ci'])} | "
            f"{_fmt(m['point'][f'pr_auc_{c}'], m['ci'][f'pr_auc_{c}'])} | "
            f"{_fmt(m['point'][f'f1_{c}'], m['ci'][f'f1_{c}'])} | {_fmt(b['point'][f'f1_{c}'], b['ci'][f'f1_{c}'])} |"
        )
    lines += [
        "",
        "ECE, Brier: меньше — лучше. Разница считается на одних и тех же репликах для обеих моделей (парный bootstrap).",
        "ДИ для ECE ориентировочный: бинированный ECE на bootstrap-репликах смещён вверх (шум выборки добавляется "
        "к ошибке калибровки), поэтому интервал может не накрывать точечное значение. Для сравнения калибровки "
        "надёжнее Brier (собственная оценка, без бинов).",
        "Столбец «val» — точечные значения на фолде 9 (без ДИ), только для сравнения с test. На val выбирались модель, "
        "эпоха, T и пороги, поэтому эти значения слегка оптимистичны.",
        "\\* F1 и ECE на val посчитаны так, чтобы пороги и T не оценивались на тех же записях, на которых подобраны: "
        "пациенты val делятся на 5 частей, T и пороги подбираются на 4 частях и применяются к пятой, по кругу; "
        "метрика считается по всем отложенным предсказаниям (в коде это `cross_fitted_estimate`, src/evaluate.py). "
        "Пороги и T, применённые к test, подобраны на всём val.",
    ]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- основной сценарий

def run(cfg: dict[str, Any], config_path: Path, dry_run: bool) -> tuple[dict[str, Any], str, Path]:
    """Проверки на val -> чтение оценочной выборки -> предсказания -> метрики и bootstrap -> файлы."""
    t0 = time.perf_counter()
    eval_split = "val" if dry_run else "test"
    out_dir = _resolve(cfg["dry_run_dir"] if dry_run else cfg["output_dir"])
    if not dry_run and (out_dir / "metrics.json").exists():
        raise SystemExit(f"{out_dir / 'metrics.json'} уже существует: финальная оценка выполняется один раз. Остановка.")
    set_seed(cfg["seed"])
    ev_cfg = load_config(cfg["evaluate_config"])
    eps, n_bins = ev_cfg["probs_eps"], ev_cfg["ece"]["n_bins"]
    data_cfg = load_config(cfg["data_config"])["data"]
    fs = data_cfg["sampling_rate"]
    bcfg = load_config(cfg["baseline"]["config"])
    run_dir = _resolve(cfg["model"]["run"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Только train + val (сырые). Все проверки и подбор параметров бейзлайна — до test.
    splits, meta = load_splits(data_cfg, splits=("train", "val"), normalize=False)
    tr, va = splits["train"], splits["val"]
    classes: list[str] = meta["classes"]
    mean, std = np.asarray(meta["norm_mean"], np.float32), np.asarray(meta["norm_std"], np.float32)
    run_metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    ref = json.loads(_resolve(cfg["baseline"]["reference_metrics"]).read_text(encoding="utf-8"))
    for name, fp in (("модель", run_metrics["env"]["cache_fingerprint"]), ("бейзлайн", ref["env"]["cache_fingerprint"])):
        if fp != meta["fingerprint"]:
            raise RuntimeError(f"{name}: кэш данных изменился ({fp} != {meta['fingerprint']})")
    post = json.loads((run_dir / "postprocess.json").read_text(encoding="utf-8"))
    if post["classes"] != classes or "val" not in post["fitted_on"]:
        raise RuntimeError(f"postprocess.json не соответствует: {post['classes']}, {post['fitted_on']}")

    # 2. Модель воспроизводит свои val-вероятности.
    p_val_model = predict_run(run_dir, normalize_(va.X.copy(), mean, std), va.y, meta["leads"], device)
    model_diff = float(np.abs(p_val_model - np.load(run_dir / "val_probs.npy")).max())
    logger.info("модель: макс. |Δp| с сохранённым val_probs.npy = %.2e", model_diff)
    if model_diff > cfg["model"]["repro_max_abs_diff"]:
        raise RuntimeError(f"best.pt не воспроизводит val_probs.npy: {model_diff:.2e}")

    # 3. Бейзлайн: переобучение (train, ES по val), проверка по val, T и пороги на val.
    models, cols, p_val_base, binfo = fit_baseline(bcfg, tr, va, meta, fs)
    base_val_auc = float(np.mean([roc_auc_score(va.y[:, i], p_val_base[:, i]) for i in range(len(classes))]))
    base_diff = abs(base_val_auc - ref["val"]["macro_auroc"])
    logger.info("бейзлайн: val macro AUROC %.6f (записано %.6f)", base_val_auc, ref["val"]["macro_auroc"])
    if base_diff > cfg["baseline"]["repro_max_abs_diff_auroc"] or binfo["best_iteration"] != ref["variants"][binfo["variant"]]["best_iteration"]:
        raise RuntimeError(f"бейзлайн не воспроизвёлся: ΔAUROC {base_diff:.2e}, best_iteration {binfo['best_iteration']}")
    base_post = fit_postprocess(p_val_base, va.y, ev_cfg)  # только val
    model_val_ref = json.loads((run_dir / "eval_val.json").read_text(encoding="utf-8"))
    n_excl = {k: v["excluded_no_superclass"] for k, v in meta["stats"]["splits"].items()}
    del splits, tr

    # 4. Оценочная выборка: test (боевой прогон) или val (dry run). Дальше ничего не подбирается.
    logger.info("Читаю оценочную выборку: %s", eval_split)
    ev = va if dry_run else load_splits(data_cfg, splits=("test",), normalize=False)[0]["test"]
    y = ev.y.astype(np.uint8)
    p_model_raw = predict_run(run_dir, normalize_(ev.X.copy(), mean, std), ev.y, meta["leads"], device).astype(np.float64)
    F_ev, names_ev = baseline_features(bcfg, eval_split, ev.X, meta, fs)
    assert names_ev == binfo["feature_names"]
    p_base_raw = baseline_predict(models, F_ev[:, cols])

    p_model_cal, _ = apply_postprocess(p_model_raw, post, post["probs_eps"])
    p_base_cal, _ = apply_postprocess(p_base_raw, base_post, eps)
    thr_model = np.asarray(post["thresholds"], dtype=np.float64)
    thr_base = np.asarray(base_post["thresholds"], dtype=np.float64)
    preds = {MODEL_KEY: (p_model_cal, p_model_raw, thr_model), BASE_KEY: (p_base_cal, p_base_raw, thr_base)}

    point = {k: metric_vector(y, pc, pr, th, classes, n_bins, eps) for k, (pc, pr, th) in preds.items()}
    keys = list(point[MODEL_KEY])

    def stat(idx: np.ndarray) -> np.ndarray:
        return np.concatenate([
            list(metric_vector(y[idx], pc[idx], pr[idx], th, classes, n_bins, eps).values())
            for pc, pr, th in preds.values()
        ])

    bs = cfg["bootstrap"]
    t_boot = time.perf_counter()
    boot = cluster_bootstrap(stat, ev.patient_id, bs["n_boot"], cfg["seed"])  # (B, 2K): [модель | бейзлайн]
    t_boot = time.perf_counter() - t_boot
    k = len(keys)
    reps = {MODEL_KEY: boot[:, :k], BASE_KEY: boot[:, k:], "diff": boot[:, :k] - boot[:, k:]}
    ci = {name: percentile_ci(r, bs["level"], bs["min_valid_share"]) for name, r in reps.items()}

    def ci_dict(name: str) -> dict[str, list[float]]:
        lo, hi = ci[name]
        return {key: [float(lo[j]), float(hi[j])] for j, key in enumerate(keys)}

    cf = model_val_ref["cross_fitted_estimate"]
    val_reference = {
        "macro_auroc": model_val_ref["ranking_metrics"]["macro_auroc"],
        "macro_pr_auc": model_val_ref["ranking_metrics"]["macro_pr_auc"],
        "macro_f1": cf["macro_f1"], "micro_f1": cf["micro_f1"],
        "macro_ece": cf["ece_calibrated"]["macro"]["ece"], "macro_ece_raw": cf["ece_raw"]["macro"]["ece"],
    }
    lo_d, hi_d = ci["diff"]
    result: dict[str, Any] = {
        "evaluated_on": (f"val (strat_fold 9) — DRY RUN, проверка конвейера, не финальный результат" if dry_run
                         else "test (strat_fold 10) — единственный финальный прогон"),
        "dry_run": dry_run,
        "eval_split": eval_split,
        "n_records": int(len(y)),
        "n_patients": int(len(np.unique(ev.patient_id))),
        "excluded_no_superclass": n_excl,
        "prevalence": {c: float(y[:, i].mean()) for i, c in enumerate(classes)},
        "classes": classes,
        "models": {
            MODEL_KEY: {
                "name": run_dir.name.split("_", 1)[1], "source_run": cfg["model"]["run"],
                "postprocess": {"source": f"{cfg['model']['run']}/postprocess.json", "fitted_on": post["fitted_on"],
                                "temperature": post["temperature"], "thresholds": dict(zip(classes, post["thresholds"]))},
                "point": point[MODEL_KEY], "ci": ci_dict(MODEL_KEY), "val_reference": val_reference,
            },
            BASE_KEY: {
                "name": "lightgbm", "source": cfg["baseline"]["config"],
                "refit": {**{k2: v for k2, v in binfo.items() if k2 != "feature_names"},
                          "val_macro_auroc": base_val_auc, "val_macro_auroc_recorded": ref["val"]["macro_auroc"]},
                "postprocess": {"fitted_on": "val (strat_fold 9)", "temperature": base_post["temperature"],
                                "thresholds": dict(zip(classes, base_post["thresholds"]))},
                "point": point[BASE_KEY], "ci": ci_dict(BASE_KEY),
                "val_reference": {"macro_auroc": ref["val"]["macro_auroc"], "macro_pr_auc": ref["val"]["macro_pr_auc"]},
            },
        },
        "difference_model_minus_baseline": {
            key: {"point": point[MODEL_KEY][key] - point[BASE_KEY][key], "ci": [float(lo_d[j]), float(hi_d[j])],
                  "share_replicates_le_0": float(np.mean(reps["diff"][:, j] <= 0))}
            for j, key in enumerate(keys)
        },
        "bootstrap": {"unit": "patient_id", "n_boot": bs["n_boot"], "level": bs["level"], "method": "percentile",
                      "paired": True, "seed": cfg["seed"], "min_valid_share": bs["min_valid_share"],
                      "n_valid_min": int(np.isfinite(boot).sum(axis=0).min())},
        "checks": {"model_val_probs_max_abs_diff": model_diff, "baseline_val_macro_auroc_abs_diff": base_diff,
                   "cache_fingerprint": meta["fingerprint"]},
        "rules": "T и пороги — только с val (фолд 9); на оценочной выборке ничего не подбиралось и не перезапускалось.",
        "config": cfg,
        "time_sec": {"bootstrap": round(t_boot, 1), "total": round(time.perf_counter() - t0, 1)},
        "env": {
            "python": platform.python_version(), "torch": torch.__version__, "lightgbm": lgb.__version__,
            "scikit-learn": sklearn.__version__, "numpy": np.__version__, "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None, "git": _git_state(),
            "sha1": {p: _sha1(PROJECT_ROOT / p) for p in ("src/final_eval.py", "src/baseline.py", "src/evaluate.py",
                                                          str(config_path.relative_to(PROJECT_ROOT)).replace("\\", "/"))},
        },
    }
    result = _nan_to_none(_jsonable(result))

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"probs_{eval_split}_{MODEL_KEY}.npy", p_model_raw)
    np.save(out_dir / f"probs_{eval_split}_{BASE_KEY}.npy", p_base_raw)
    np.save(out_dir / f"ecg_id_{eval_split}.npy", ev.ecg_id)
    np.savez_compressed(out_dir / "bootstrap_replicates.npz", model=reps[MODEL_KEY], baseline=reps[BASE_KEY],
                        keys=np.array(keys))
    (out_dir / "baseline_postprocess.json").write_text(json.dumps(_jsonable({
        "classes": classes, "temperature_mode": ev_cfg["temperature"]["mode"], "temperature": base_post["temperature"],
        "thresholds": base_post["thresholds"], "fitted_on": "val (strat_fold 9)", "probs_eps": eps,
    }), ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy(config_path, out_dir / "config.yaml")
    table = format_table(result, classes)
    (out_dir / "table.md").write_text(table, encoding="utf-8")
    # metrics.json — последним: его наличие означает, что финальный прогон завершён
    (out_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result, table, out_dir


def main() -> None:
    """CLI: dry run на val или единственный финальный прогон на test."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/final_eval.yaml")
    parser.add_argument("--dry-run", action="store_true", help="проверить конвейер на val, test не читать")
    parser.add_argument("--table-only", action="store_true",
                        help="только перерисовать table.md из готового metrics.json (данные не читаются, ничего не считается)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config_path = _resolve(args.config)
    if args.table_only:
        cfg = load_config(config_path)
        out_dir = _resolve(cfg["dry_run_dir"] if args.dry_run else cfg["output_dir"])
        result = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
        result.setdefault("eval_split", "val" if result["dry_run"] else "test")
        table = format_table(result, result["classes"])
        (out_dir / "table.md").write_text(table, encoding="utf-8")
        print(table)
        return
    _, table, out_dir = run(load_config(config_path), config_path, args.dry_run)
    print(table)
    print(f"Сохранено: {out_dir}")


if __name__ == "__main__":
    main()
