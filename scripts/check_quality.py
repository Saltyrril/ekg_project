"""Проверка и калибровка порогов app/quality.py на валидационном фолде 9 PTB-XL. Тестовый фолд 10 НЕ читается.

Что считается:
  1. Чистые записи val: доля статусов ok/warning/poor и частота каждого замечания (ложные срабатывания, если
     считать val чистым), связь с шумовыми разметками PTB-XL (baseline_drift, static_noise, burst_noise).
  2. Искусственные искажения (белый шум, дрейф изолинии): доля помеченных записей и macro AUROC модели приложения
     на искажённых записях — чтобы пороги шума/дрейфа стояли там, где качество предсказаний начинает падать.
  3. Искусственные неисправности (плоское отведение, обрыв, неверные единицы, перепутанные отведения, чистый
     шум): доля обнаруженных.

Запуск:  python scripts/check_quality.py --config configs/quality_check.yaml
Результат: results/<дата>_quality_check/{config.yaml, metrics.json, summary.md}.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))
from inference import calibrated_probs, load_bundle, normalize, predict_logits  # noqa: E402
from quality import THRESHOLDS, assess_quality, lead_metrics  # noqa: E402
from src.data import _resolve, load_config, load_metadata, load_splits  # noqa: E402

FS = 100


def git_state() -> dict[str, Any]:
    """Короткий хэш коммита и признак «грязного» дерева."""
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    try:
        return {"commit": run("rev-parse", "--short", "HEAD"), "dirty": bool(run("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def lead_p2p(x: np.ndarray) -> np.ndarray:
    """Робастный размах каждого отведения: (N, 12) для x (N, 12, T)."""
    lo, hi = np.percentile(x, [0.5, 99.5], axis=-1)
    return hi - lo


def macro_auroc(model_bundle: Any, x_mv: np.ndarray, y: np.ndarray, batch: int) -> float:
    """macro AUROC модели приложения на сигналах (N, 12, T) в мВ."""
    logits = np.concatenate([predict_logits(normalize_batch(x_mv[i:i + batch], model_bundle), model_bundle)
                             for i in range(0, len(x_mv), batch)])
    return float(roc_auc_score(y, calibrated_probs(logits, model_bundle), average="macro"))


def normalize_batch(x: np.ndarray, bundle: Any) -> np.ndarray:
    """z-нормализация пачки (N, 12, T) статистиками train из модели приложения."""
    return ((x - bundle.mean[None, :, None]) / bundle.std[None, :, None]).astype(np.float32)


def assess_all(x: np.ndarray, leads: list[str]) -> list[Any]:
    """assess_quality для каждой записи."""
    return [assess_quality(rec, leads, FS) for rec in x]


def flag_summary(reports: list[Any]) -> dict[str, Any]:
    """Доли статусов и частота замечаний (по коду и серьёзности)."""
    n = len(reports)
    status = Counter(r.status for r in reports)
    codes = Counter(f"{i.code}:{i.severity}" for r in reports for i in {(i.code, i.severity): i for i in r.issues}.values())
    by_code = Counter(c for r in reports for c in {i.code for i in r.issues})
    return {"n": n, "status": {k: status.get(k, 0) / n for k in ("ok", "warning", "poor")},
            "flagged_any": 1 - status.get("ok", 0) / n, "issue_rates": {k: v / n for k, v in sorted(codes.items())},
            "code_rates": {k: v / n for k, v in sorted(by_code.items())}}


def metric_percentiles(x: np.ndarray, fs: int) -> dict[str, dict[str, float]]:
    """Распределение метрик по чистым записям: для каждой записи берётся максимум по отведениям."""
    keys = ("hf_noise", "baseline_wander", "flat_run")
    per = {k: [] for k in keys}
    for rec in x:
        m = lead_metrics(rec.astype(np.float64), fs)
        for k in keys:
            per[k].append(float(m[k].max()))
    return {k: {f"p{q}": float(np.percentile(v, q)) for q in (50, 95, 99, 99.9, 100)} for k, v in per.items()}


def make_faults(x: np.ndarray, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Искусственные неисправности чистых записей (N, 12, T) мВ."""
    out: dict[str, np.ndarray] = {}
    a = x.copy(); a[:, 8] = np.median(a[:, 8], axis=-1, keepdims=True); out["flat_lead_V3"] = a
    a = x.copy(); a[:, :, 500:] = a[:, :, 499:500]; out["dropout_last_5s_all_leads"] = a
    a = x.copy(); a[:, 3, 300:] = a[:, 3, 299:300]; out["dropout_aVR_last_7s"] = a
    out["units_uV_read_as_mV"] = x * 1000.0
    out["units_V_read_as_mV"] = x * 0.001
    a = x.copy(); a[:, [1, 2]] = a[:, [2, 1]]; out["swap_II_III"] = a
    a = x.copy(); a[:, [0, 1]] = a[:, [1, 0]]; out["swap_I_II"] = a
    a = x.copy(); a[:, 0] *= -1; out["invert_I"] = a
    out["pure_noise_no_ecg"] = rng.normal(0.0, 0.3, x.shape).astype(np.float32)
    a = x.copy(); a[:, 6] = a[:, 1]; out["duplicate_II_as_V1"] = a
    return out


def main() -> int:
    """Выполняет проверки, печатает таблицу и пишет results/<дата>_quality_check/."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/quality_check.yaml")
    args = parser.parse_args()
    t0 = time.time()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if cfg["fold"] != 9:
        raise ValueError("Подбор порогов допустим только на фолде 9 (val); test (фолд 10) не используется.")
    rng = np.random.default_rng(cfg["seed"])
    data_cfg = load_config(cfg["data_config"])["data"]
    if data_cfg["folds"]["val"] != [9]:
        raise ValueError("В конфиге данных val должен быть фолдом 9.")
    data, meta = load_splits(data_cfg, splits=("val",), normalize=False)  # test не загружается
    val = data["val"]
    bundle = load_bundle(ROOT / cfg["model_dir"])
    leads = bundle.leads
    if meta.get("leads", leads) != leads:
        raise ValueError("Порядок отведений кэша не совпадает с моделью приложения.")
    x = val.X.astype(np.float32)
    y = val.y
    batch = int(cfg["batch_size"])
    print(f"val: {x.shape}, модель {bundle.meta['run']}")

    # 1. чистый val
    clean = assess_all(x, leads)
    result: dict[str, Any] = {"n_records": int(len(x)), "thresholds": THRESHOLDS, "clean": flag_summary(clean),
                              "clean_metric_percentiles_max_over_leads": metric_percentiles(x, FS)}
    db, _ = load_metadata(_resolve(data_cfg["root"]))
    ann = db.loc[val.ecg_id]
    flagged = np.array([r.status != "ok" for r in clean])
    poor = np.array([r.status == "poor" for r in clean])
    result["vs_ptbxl_annotations"] = {}
    for col in ("baseline_drift", "static_noise", "burst_noise", "electrodes_problems"):
        has = (ann[col].notna() & (ann[col].astype(str).str.strip() != "")).to_numpy()
        result["vs_ptbxl_annotations"][col] = {
            "n_annotated": int(has.sum()),
            "flagged_if_annotated": float(flagged[has].mean()) if has.any() else None,
            "flagged_if_not_annotated": float(flagged[~has].mean()),
            "poor_if_annotated": float(poor[has].mean()) if has.any() else None,
        }
    result["auroc_clean"] = macro_auroc(bundle, x, y, batch)
    print("чистый val:", result["clean"]["status"], "AUROC", round(result["auroc_clean"], 4))

    # 2. шум и дрейф
    p2p = lead_p2p(x)[:, :, None]
    t = np.arange(x.shape[-1]) / FS
    sweeps: dict[str, list[dict[str, Any]]] = {"white_noise": [], "baseline_drift": []}
    for r in cfg["noise_levels"]:
        xm = x + rng.normal(0.0, 1.0, x.shape).astype(np.float32) * (r * p2p) if r else x
        s = flag_summary(assess_all(xm, leads))
        sweeps["white_noise"].append({"level": r, "macro_auroc": macro_auroc(bundle, xm, y, batch), **s})
    for d in cfg["drift_levels"]:
        phase = rng.uniform(0, 2 * np.pi, (len(x), 1, 1))
        xm = x + (d * p2p * np.sin(2 * np.pi * cfg["drift_freq_hz"] * t[None, None] + phase)).astype(np.float32) if d else x
        s = flag_summary(assess_all(xm, leads))
        sweeps["baseline_drift"].append({"level": d, "macro_auroc": macro_auroc(bundle, xm, y, batch), **s})
    result["sweeps"] = sweeps

    # 3. неисправности
    faults = {}
    for name, xf in make_faults(x, rng).items():
        s = flag_summary(assess_all(xf, leads))
        faults[name] = {"macro_auroc": macro_auroc(bundle, xf.astype(np.float32), y, batch), **s}
    result["faults"] = faults
    result["elapsed_s"] = round(time.time() - t0, 1)
    result["git"] = git_state()
    result["created"] = time.strftime("%Y-%m-%d %H:%M:%S")

    out = ROOT / cfg["out_dir"] / f"{time.strftime('%Y-%m-%d')}_quality_check"
    if out.exists():
        out = out.with_name(out.name + time.strftime("_%H%M%S"))
    out.mkdir(parents=True)
    shutil.copy(args.config, out / "config.yaml")
    (out / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "summary.md").write_text(summary_md(result), encoding="utf-8")
    print(summary_md(result))
    print(f"-> {out}")
    return 0


def summary_md(r: dict[str, Any]) -> str:
    """Короткая таблица результатов."""
    c = r["clean"]
    lines = [f"# Проверка качества сигнала (val, фолд 9), {r['n_records']} записей\n",
             f"Чистый val: ok {c['status']['ok']:.1%}, warning {c['status']['warning']:.1%}, poor {c['status']['poor']:.1%}; "
             f"macro AUROC модели {r['auroc_clean']:.4f}.\n",
             "Замечания на чистом val: " + (", ".join(f"{k} {v:.2%}" for k, v in c["issue_rates"].items()) or "нет") + "\n",
             "## Связь с разметкой PTB-XL\n", "| разметка | записей | помечено среди размеченных | среди остальных |", "|---|---:|---:|---:|"]
    for k, v in r["vs_ptbxl_annotations"].items():
        fa = "—" if v["flagged_if_annotated"] is None else f"{v['flagged_if_annotated']:.1%}"
        lines.append(f"| {k} | {v['n_annotated']} | {fa} | {v['flagged_if_not_annotated']:.1%} |")
    lines += ["\n## Распределение метрик на чистом val (максимум по отведениям)\n", "| метрика | p50 | p95 | p99 | p99.9 | max |",
              "|---|---:|---:|---:|---:|---:|"]
    lines += [f"| {k} | " + " | ".join(f"{v[q]:.3f}" for q in ("p50", "p95", "p99", "p99.9", "p100")) + " |"
              for k, v in r["clean_metric_percentiles_max_over_leads"].items()]
    cols = ("hf_noise", "baseline_wander", "einthoven", "widespread_noise")
    for name, rows in r["sweeps"].items():
        lines += [f"\n## {name}\n", "| уровень | macro AUROC | warning | poor | " + " | ".join(cols) + " |",
                  "|---:|---:|---:|---:|" + "---:|" * len(cols)]
        lines += [f"| {x['level']} | {x['macro_auroc']:.4f} | {x['status']['warning']:.1%} | {x['status']['poor']:.1%} | "
                  + " | ".join(f"{x['code_rates'].get(c, 0):.1%}" for c in cols) + " |" for x in rows]
    lines += ["\n## Неисправности\n", "| искажение | macro AUROC | помечено (любое) | poor |", "|---|---:|---:|---:|"]
    lines += [f"| {k} | {v['macro_auroc']:.4f} | {v['flagged_any']:.1%} | {v['status']['poor']:.1%} |" for k, v in r["faults"].items()]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
