"""Собирает самодостаточный пакет Gradio-демо в app/ (папка = корень Hugging Face Space).

Что пишет:
  app/model/best.pt          веса запуска (копия)
  app/model/model_meta.json  архитектура, отведения, нормализация (train), T и пороги (val)
  app/ecg_models.py          копия src/models.py: на Space нет папки src/
  app/examples/              случайные одно-меточные записи ВАЛИДАЦИОННОГО фолда 9 (.hea + .dat) и examples.json

Проверка: пайплайн приложения (чтение wfdb -> нормализация -> модель в fp32) на случайных записях val должен
воспроизводить val_probs.npy запуска. Test (фолд 10) не читается.

Запуск:  python scripts/export_app_bundle.py --config configs/app.yaml
"""
from __future__ import annotations

import argparse
import importlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.data import CANONICAL_FOLDS, _cache_dir, _resolve, load_metadata  # noqa: E402
from src.evaluate import load_val_inputs  # noqa: E402

VAL_FOLD = CANONICAL_FOLDS["val"][0]


def _git_commit() -> str:
    """Короткий хэш HEAD (или 'unknown')."""
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_meta(run_dir: Path, cache_meta: dict[str, Any], ig_steps: int) -> dict[str, Any]:
    """Всё, что приложению нужно знать о модели, без доступа к src/ и data/."""
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    post = json.loads((run_dir / "postprocess.json").read_text(encoding="utf-8"))
    if post["classes"] != metrics["classes"]:
        raise ValueError(f"Классы postprocess {post['classes']} != классы запуска {metrics['classes']}")
    leads: list[str] = metrics.get("leads") or cache_meta["leads"]
    if leads != cache_meta["leads"]:
        raise ValueError(f"Демо рассчитано на все 12 отведений в порядке кэша, у запуска: {leads}")
    rate = int(cache_meta["config"]["sampling_rate"])
    return {
        "run": run_dir.name,
        "model": cfg["model"],
        "classes": metrics["classes"],
        "leads": leads,
        "sampling_rate": rate,
        "length": 10 * rate,
        "normalization": cache_meta["config"]["normalization"],
        "norm_mean": cache_meta["norm_mean"],
        "norm_std": cache_meta["norm_std"],
        "norm_fitted_on": "train (strat_fold 1-8)",
        "temperature": post["temperature"],
        "thresholds": post["thresholds"],
        "postprocess_fitted_on": post["fitted_on"],
        "ig_steps": ig_steps,
        "cache_fingerprint": cache_meta["fingerprint"],
        "git_commit": _git_commit(),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def pick_examples(y: np.ndarray, classes: list[str], per_class: int, rng: np.random.Generator) -> list[int]:
    """Индексы случайных записей val с ровно одной меткой, по `per_class` на класс."""
    single = y.sum(axis=1) == 1
    out: list[int] = []
    for c in range(len(classes)):
        idx = np.flatnonzero(single & (y[:, c] == 1))
        out += sorted(rng.choice(idx, size=per_class, replace=False).tolist())
    return out


def main() -> int:
    """Собирает пакет, копирует примеры, проверяет воспроизведение val_probs."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/app.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(_resolve(args.config).read_text(encoding="utf-8"))
    run_dir, out_dir = _resolve(cfg["run"]), _resolve(cfg["out_dir"])
    rng = np.random.default_rng(cfg["seed"])

    val_probs, y_val, _, classes, _ = load_val_inputs(run_dir)  # проверяет отпечаток кэша и классы
    data_cfg = yaml.safe_load((run_dir / "data_config.yaml").read_text(encoding="utf-8"))["data"]
    cache = _cache_dir(data_cfg)
    cache_meta = json.loads((cache / "meta.json").read_text(encoding="utf-8"))
    ecg_val = np.load(cache / "ecg_id_val.npy")
    db, _ = load_metadata(_resolve(data_cfg["root"]))
    file_col = {100: "filename_lr", 500: "filename_hr"}[int(data_cfg["sampling_rate"])]
    folds = db.loc[ecg_val, "strat_fold"].to_numpy()
    if not (folds == VAL_FOLD).all():
        raise RuntimeError(f"В ecg_id_val.npy есть записи не из фолда {VAL_FOLD}: {sorted(set(folds.tolist()))}")

    # 1. модель и параметры
    model_dir = out_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_dir / "best.pt", model_dir / "best.pt")
    meta = build_meta(run_dir, cache_meta, int(cfg["ig_steps"]))
    (model_dir / "model_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    src_models = (PROJECT_ROOT / "src" / "models.py").read_text(encoding="utf-8")
    header = "# Сгенерировано scripts/export_app_bundle.py: копия src/models.py для Space. Не редактировать.\n"
    (out_dir / "ecg_models.py").write_text(header + src_models, encoding="utf-8")

    # 2. примеры (только val)
    ex_dir = out_dir / "examples"
    if ex_dir.exists():
        shutil.rmtree(ex_dir)
    ex_dir.mkdir(parents=True)
    root = _resolve(data_cfg["root"])
    examples = []
    for i in pick_examples(y_val, classes, int(cfg["examples"]["per_class"]), rng):
        eid = int(ecg_val[i])
        rel = Path(db.loc[eid, file_col])
        for ext in (".hea", ".dat"):
            shutil.copy2(root / rel.with_suffix(ext), ex_dir / (rel.name + ext))
        examples.append({"record": rel.name, "files": [rel.name + ".hea", rel.name + ".dat"], "ecg_id": eid,
                         "strat_fold": int(db.loc[eid, "strat_fold"]),
                         "labels": [c for c, v in zip(classes, y_val[i]) if v]})
    (ex_dir / "examples.json").write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")

    # 3. проверка: приложение воспроизводит val_probs (сырые sigmoid, до температуры)
    sys.path.insert(0, str(out_dir))
    inference = importlib.import_module("inference")
    bundle = inference.load_bundle(model_dir)
    check_idx = sorted(set(rng.choice(len(ecg_val), size=int(cfg["check"]["n_records"]), replace=False).tolist())
                       | {int(np.flatnonzero(ecg_val == e["ecg_id"])[0]) for e in examples})
    diffs = []
    for i in check_idx:
        rel = root / db.loc[int(ecg_val[i]), file_col]
        rec = inference.load_record([rel.with_suffix(".hea"), rel.with_suffix(".dat")], bundle)
        logits = inference.predict_logits(inference.normalize(rec.signal, bundle)[None], bundle)[0]
        diffs.append(np.abs(1 / (1 + np.exp(-logits)) - val_probs[i]).max())
    max_diff = float(np.max(diffs))
    ok = max_diff <= float(cfg["check"]["max_abs_diff"])
    report = {"n_checked": len(check_idx), "max_abs_diff": max_diff, "threshold": cfg["check"]["max_abs_diff"],
              "passed": ok, "split": f"val (strat_fold {VAL_FOLD})"}
    (model_dir / "repro_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Пакет собран в {out_dir}: модель {meta['run']}, примеров {len(examples)} "
          f"({', '.join(e['record'] + ':' + '+'.join(e['labels']) for e in examples)})")
    print(f"Воспроизведение val_probs на {len(check_idx)} записях val: max |Δp| = {max_diff:.2e} "
          f"({'OK' if ok else 'ПРОВАЛ'}, порог {cfg['check']['max_abs_diff']})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
