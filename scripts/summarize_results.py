"""Собирает results/experiments.csv и results/summary.md из metrics.json всех запусков.

Берутся папки results/<дата>_<имя>/ (копия results/baseline/ пропускается). Только val-метрики (фолд 9).
Запуск:  python scripts/summarize_results.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

RESULTS = Path(__file__).resolve().parents[1] / "results"
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def load_runs() -> list[dict[str, Any]]:
    """Одна строка на запуск, по порядку имён папок (дата в начале имени)."""
    rows: list[dict[str, Any]] = []
    for path in sorted(RESULTS.glob("20*/metrics.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        val = m["val"]
        setup = m.get("train_setup", {})
        git = m.get("env", {}).get("git", {})
        rows.append({
            "run": path.parent.name,
            "experiment": m["experiment"],
            "augment": "да" if setup.get("augment") else ("нет" if setup else "—"),
            # старые запуски (до параметра leads) обучались на всех 12; у бейзлайна свои признаки
            "leads": ",".join(m["leads"]) if "leads" in m else ("все 12" if setup else "—"),
            "n_leads": len(m["leads"]) if "leads" in m else (12 if setup else ""),
            "n_parameters": m.get("model", {}).get("n_parameters", ""),
            "epochs_run": m.get("epochs_run", ""),
            "best_epoch": m.get("best_epoch", ""),
            "val_macro_auroc": round(val["macro_auroc"], 4),
            **{f"auroc_{c}": round(val["auroc"][c], 4) for c in CLASSES},
            "val_macro_pr_auc": round(val["macro_pr_auc"], 4),
            "time_sec": m["time_sec"]["total"],
            "seed": m.get("seed", ""),
            "git_commit": git.get("commit", ""),
            "git_dirty": git.get("dirty", ""),
        })
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Пишет таблицу в CSV (UTF-8 с BOM — чтобы Excel не ломал кириллицу)."""
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def leads_section(rows: list[dict[str, Any]]) -> list[str]:
    """Сравнение запусков cnn1d_aug_leads_* по наборам отведений (val, фолд 9); лучший по macro AUROC жирным."""
    sel = sorted((r for r in rows if r["experiment"].startswith("cnn1d_aug_leads_")), key=lambda r: r["n_leads"])
    if not sel:
        return []
    best = max(r["val_macro_auroc"] for r in sel)
    cols = ["Отведения", "macro AUROC"] + CLASSES
    out = ["## Сравнение по отведениям (cnn1d_aug, val, фолд 9)", "",
           "| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in sel:
        is_best = r["val_macro_auroc"] == best
        name = f"{r['n_leads']}: {r['leads']}" + (" ← лучший" if is_best else "")
        auc = f"**{r['val_macro_auroc']}**" if is_best else str(r["val_macro_auroc"])
        out.append("| " + " | ".join([name, auc] + [str(r[f"auroc_{c}"]) for c in CLASSES]) + " |")
    return [*out, ""]


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    """Пишет summary.md: таблица, отсортированная по val macro AUROC, и оговорки."""
    ranked = sorted(rows, key=lambda r: r["val_macro_auroc"], reverse=True)
    for r in ranked:  # для бейзлайна эпох нет
        r["epochs"] = f"{r['epochs_run']} ({r['best_epoch']})" if r["epochs_run"] != "" else "—"

    def table(cols: list[tuple[str, str]]) -> list[str]:
        out = ["| " + " | ".join(h for h, _ in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
        for r in ranked:
            out.append("| " + " | ".join(str(r[k]) for _, k in cols) + " |")
        return out

    lines = [
        "# Сводка экспериментов (val, strat_fold 9)",
        "",
        "Сгенерировано `scripts/summarize_results.py` из `results/*/metrics.json`; таблица в `results/experiments.csv`.",
        "Сортировка по val macro AUROC.",
        "",
        "## Основные метрики",
        "",
        *table([("Запуск", "run"), ("Отведений", "n_leads"), ("Аугм.", "augment"),("Параметров", "n_parameters"), ("Эпох (лучшая)", "epochs"),
                ("Время, с", "time_sec"), ("macro AUROC", "val_macro_auroc"), ("macro PR-AUC", "val_macro_pr_auc")]),
        "",
        "## AUROC по классам",
        "",
        *table([("Запуск", "run")] + [(c, f"auroc_{c}") for c in CLASSES]),
        "",
        *leads_section(ranked),
        "## Оговорки",
        "",
        "- Все числа — на **val (фолд 9)**; test (фолд 10) не использовался. Ранняя остановка и выбор эпохи идут по val,"
        " поэтому метрики слегка оптимистичны; несмещённая оценка — только на test в финале.",
        "- Один seed (42) на каждый запуск, доверительных интервалов нет: разницу в несколько тысячных AUROC"
        " нельзя считать значимой без повторов с разными seed.",
        "- «Эпох (лучшая)»: число выполненных эпох и в скобках номер лучшей; у бейзлайна эпох нет.",
        "- `2026-11-05_resnet1d` и `2026-11-05_resnet1d_<время>` — повтор одной и той же конфигурации"
        " (одинаковый AUROC подтверждает воспроизводимость), это не два разных эксперимента.",
        "- `git_dirty = True` в CSV: код нейросетей на момент запусков не был закоммичен, хэш коммита не идентифицирует"
        " код однозначно.",
        "- Метрики macro F1 с порогами по val и калибровка пока не считались.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """CLI: собрать CSV и Markdown."""
    rows = load_runs()
    write_csv(rows, RESULTS / "experiments.csv")
    write_markdown(rows, RESULTS / "summary.md")
    print(f"Запусков: {len(rows)} -> results/experiments.csv, results/summary.md")


if __name__ == "__main__":
    main()
