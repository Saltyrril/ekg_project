"""Прямая сверка кэша data/cache/ptbxl_100hz с исходными CSV PTB-XL. Только чтение кэша.

Проверки:
  1. strat_fold каждой записи кэша (по ecg_id) совпадает с CSV и с правилами (train 1-8, val 9, test 10);
     patient_id в кэше совпадает с CSV; набор записей полон (все записи фолда с суперклассом есть, лишних нет);
     метки y совпадают с пересчитанными из scp_codes.
  2. Нет общих patient_id между выборками (по кэшу и по CSV).
  3. norm_mean / norm_std из meta.json пересчитываются из X_train.npy.
  Доп.: сигналы случайных записей кэша совпадают с файлами wfdb.

Запуск:  python scripts/verify_cache.py [--out results/final/cache_check.json]
Код выхода 0 — все проверки пройдены, 1 — есть расхождение.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import (  # noqa: E402
    CANONICAL_FOLDS, SPLITS, _cache_dir, _fingerprint, _read_record, _resolve, build_labels, compute_norm_stats,
    load_config, load_metadata,
)


def main() -> int:
    """Выполняет проверки, печатает итог, пишет JSON-отчёт."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--out", default="results/final/cache_check.json")
    parser.add_argument("--n-signal-samples", type=int, default=30, help="записей на выборку для сверки с wfdb")
    args = parser.parse_args()
    cfg = load_config(args.config)
    data_cfg, seed = cfg["data"], cfg["seed"]
    cache = _cache_dir(data_cfg)
    meta = json.loads((cache / "meta.json").read_text(encoding="utf-8"))
    root = _resolve(data_cfg["root"])
    db, scp = load_metadata(root)
    y_csv = build_labels(db, scp, data_cfg["classes"], data_cfg["min_likelihood"])
    has_label = y_csv.any(axis=1)
    row_of = {eid: i for i, eid in enumerate(db.index)}
    report: dict[str, Any] = {"cache_dir": str(cache), "fingerprint_meta": meta["fingerprint"],
                              "fingerprint_config": _fingerprint(data_cfg), "splits": {}}
    ok = report["fingerprint_meta"] == report["fingerprint_config"]
    patients: dict[str, set[int]] = {}
    rng = np.random.default_rng(seed)
    file_col = {100: "filename_lr", 500: "filename_hr"}[data_cfg["sampling_rate"]]

    for split in SPLITS:
        ecg = np.load(cache / f"ecg_id_{split}.npy")
        pid = np.load(cache / f"patient_id_{split}.npy")
        y = np.load(cache / f"y_{split}.npy")
        X = np.load(cache / f"X_{split}.npy", mmap_mode="r")  # только для формы и выборочной сверки сигналов
        rows = np.array([row_of[e] for e in ecg])
        sub = db.iloc[rows]
        folds_found = sorted(set(sub["strat_fold"].astype(int)))
        expected = set(db.index[db["strat_fold"].isin(CANONICAL_FOLDS[split]).to_numpy() & has_label])
        sample = rng.choice(len(ecg), size=min(args.n_signal_samples, len(ecg)), replace=False)
        sig_diff = max(float(np.abs(_read_record(root / sub[file_col].iloc[i])[0] - X[i]).max()) for i in sample)
        s = {
            "n": int(len(ecg)),
            "strat_folds_in_csv": folds_found,
            "strat_fold_matches_rules": folds_found == sorted(CANONICAL_FOLDS[split]),
            "unique_ecg_id": bool(len(set(ecg.tolist())) == len(ecg)),
            "record_set_equals_csv": set(ecg.tolist()) == expected,
            "n_missing_vs_csv": len(expected - set(ecg.tolist())),
            "n_extra_vs_csv": len(set(ecg.tolist()) - expected),
            "patient_id_matches_csv": bool(np.array_equal(sub["patient_id"].to_numpy(dtype=np.int64), pid)),
            "labels_match_csv": bool(np.array_equal(y_csv[rows], y)),
            "shapes_consistent": bool(len(X) == len(y) == len(pid) == len(ecg)),
            "excluded_no_superclass_csv": int((db["strat_fold"].isin(CANONICAL_FOLDS[split]).to_numpy() & ~has_label).sum()),
            "excluded_no_superclass_meta": meta["stats"]["splits"][split]["excluded_no_superclass"],
            "signal_samples_checked": int(len(sample)),
            "signal_max_abs_diff_vs_wfdb": sig_diff,
        }
        s["excluded_matches_meta"] = s["excluded_no_superclass_csv"] == s["excluded_no_superclass_meta"]
        s["signals_match_wfdb"] = sig_diff == 0.0
        s["passed"] = all(s[k] for k in ("strat_fold_matches_rules", "unique_ecg_id", "record_set_equals_csv",
                                          "patient_id_matches_csv", "labels_match_csv", "shapes_consistent",
                                          "excluded_matches_meta", "signals_match_wfdb"))
        ok &= s["passed"]
        report["splits"][split] = s
        patients[split] = set(pid.tolist())
        del X

    # 2. Пересечение пациентов: по кэшу и независимо по CSV (по strat_fold)
    overlap_cache = {f"{a}-{b}": len(patients[a] & patients[b]) for i, a in enumerate(SPLITS) for b in SPLITS[i + 1:]}
    csv_pat = {s: set(db.loc[db["strat_fold"].isin(CANONICAL_FOLDS[s]), "patient_id"].astype(int)) for s in SPLITS}
    overlap_csv = {f"{a}-{b}": len(csv_pat[a] & csv_pat[b]) for i, a in enumerate(SPLITS) for b in SPLITS[i + 1:]}
    report["patient_overlap"] = {"cache": overlap_cache, "csv_all_records": overlap_csv,
                                 "passed": not any(overlap_cache.values()) and not any(overlap_csv.values())}
    ok &= report["patient_overlap"]["passed"]

    # 3. Нормализация: пересчёт из X_train.npy тем же кодом
    X_train = np.load(cache / "X_train.npy")  # целиком, как при сборке кэша (~0.8 ГБ)
    mean, std = compute_norm_stats(X_train, data_cfg["normalization"])
    m_meta, s_meta = np.asarray(meta["norm_mean"], np.float32), np.asarray(meta["norm_std"], np.float32)
    report["normalization"] = {
        "mode": data_cfg["normalization"], "computed_on": "X_train.npy",
        "max_abs_diff_mean": float(np.abs(mean - m_meta).max()), "max_abs_diff_std": float(np.abs(std - s_meta).max()),
        "max_rel_diff_std": float((np.abs(std - s_meta) / s_meta).max()),
    }
    report["normalization"]["passed"] = bool(np.allclose(mean, m_meta, rtol=1e-6, atol=1e-7)
                                             and np.allclose(std, s_meta, rtol=1e-6, atol=1e-7))
    ok &= report["normalization"]["passed"]
    report["passed"] = bool(ok)

    out = _resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for split, s in report["splits"].items():
        print(f"{split:<5} n={s['n']:>5} folds={s['strat_folds_in_csv']} set={s['record_set_equals_csv']} "
              f"patient_id={s['patient_id_matches_csv']} labels={s['labels_match_csv']} "
              f"excluded={s['excluded_no_superclass_csv']} signals(Δ={s['signal_max_abs_diff_vs_wfdb']:.1e}) -> "
              f"{'OK' if s['passed'] else 'FAIL'}")
    print(f"пересечение пациентов: кэш {overlap_cache}, CSV {overlap_csv} -> "
          f"{'OK' if report['patient_overlap']['passed'] else 'FAIL'}")
    n = report["normalization"]
    print(f"нормализация: max|Δmean|={n['max_abs_diff_mean']:.2e}, max|Δstd|={n['max_abs_diff_std']:.2e} -> "
          f"{'OK' if n['passed'] else 'FAIL'}")
    print(f"отпечаток: meta {report['fingerprint_meta']}, конфиг {report['fingerprint_config']}")
    print(f"ИТОГ: {'ПРОЙДЕНО' if ok else 'НЕ ПРОЙДЕНО'} | отчёт: {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
