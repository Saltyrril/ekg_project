"""Статическая демо-страница для GitHub Pages: 5 заранее посчитанных примеров валидационного фолда 9.

Что пишет в output_dir (по умолчанию docs/demo):
  index.html                          страница без JavaScript и внешних ресурсов
  examples/<класс>_<ecg_id>/ecg.png   12 отведений (та же отрисовка, что в app/app.py)
  examples/<класс>_<ecg_id>/probs.json  вероятности, T и пороги с val, решение «выше/ниже порога»
  examples/<класс>_<ecg_id>/ig_<класс>.png  карта Integrated Gradients по предсказанному классу
  manifest.json                       конфиг, выбранные записи, проверки, git, время

Отбор: для каждого класса одна случайная (seed) запись фолда 9 с единственной меткой этого класса,
которую модель нашла (калиброванная вероятность класса по val_probs.npy выше порога с val). Вручную не отбирается.
Проверки: strat_fold выбранных записей == 9; пересчёт в fp32 воспроизводит val_probs.npy (max |Δp| <= порога);
в выходной папке нет весов и сырых сигналов, её размер не больше max_output_mb.
Test (фолд 10) не читается. Весь код модели, инференса и графиков импортируется из app/ (он требует app/model,
который собирает scripts/export_app_bundle.py).

Запуск:  .venv\\Scripts\\python.exe scripts\\export_static_demo.py --config configs\\static_demo.yaml
"""
from __future__ import annotations

import argparse
import dataclasses
import html
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "app"
sys.path.insert(0, str(PROJECT_ROOT))
from src.data import CANONICAL_FOLDS, _cache_dir, _resolve, load_metadata  # noqa: E402
from src.error_analysis import format_codes  # noqa: E402
from src.evaluate import apply_postprocess, load_val_inputs  # noqa: E402

VAL_FOLD = CANONICAL_FOLDS["val"][0]
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".ckpt", ".npy", ".npz", ".dat", ".hea", ".mat", ".h5"}
PHYSIONET_URL = "https://physionet.org/content/ptb-xl/"
REPO_URL = "https://github.com/Saltyrril/ekg_project"


# ----------------------------------------------------------------------------- вспомогательное

def load_app_module() -> ModuleType:
    """Импортирует app/app.py (графики, имена классов) вместе с app/inference.py. Сервер Gradio не запускается."""
    sys.path.insert(0, str(APP_DIR))  # app.py импортирует inference и ecg_models как модули верхнего уровня
    spec = importlib.util.spec_from_file_location("ecg_demo_app", APP_DIR / "app.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Не удалось загрузить {APP_DIR / 'app.py'}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def git_state() -> dict[str, Any]:
    """Короткий хэш HEAD и признак грязного дерева."""
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    try:
        return {"commit": run("rev-parse", "--short", "HEAD"), "dirty": bool(run("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Логистическая функция."""
    return 1.0 / (1.0 + np.exp(-z))


def pick_examples(y: np.ndarray, pred: np.ndarray, n_per_class: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    """(класс, индекс записи val): случайные записи с единственной меткой класса, найденные моделью."""
    single = y.sum(axis=1) == 1
    out: list[tuple[int, int]] = []
    for c in range(y.shape[1]):
        cand = np.flatnonzero(single & (y[:, c] == 1) & (pred[:, c] == 1))
        if len(cand) < n_per_class:
            raise RuntimeError(f"Класс {c}: подходящих записей {len(cand)} < {n_per_class}")
        out += [(c, int(i)) for i in sorted(rng.choice(cand, size=n_per_class, replace=False).tolist())]
    return out


def dir_size_mb(path: Path) -> float:
    """Суммарный размер файлов папки в МБ."""
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 2**20


# ----------------------------------------------------------------------------- HTML

def _ci(point: float, ci: list[float], digits: int = 3) -> str:
    return f"{point:.{digits}f} <span class=\"ci\">[{ci[0]:.{digits}f}; {ci[1]:.{digits}f}]</span>"


def metrics_tables(fm: dict[str, Any]) -> str:
    """Таблицы из results/final/metrics.json: macro-метрики и AUROC по классам (модель, LightGBM, разница)."""
    model, base, diff = fm["models"]["model"], fm["models"]["baseline"], fm["difference_model_minus_baseline"]
    rows = []
    for key, label in (("macro_auroc", "macro AUROC"), ("macro_f1", "macro F1 (пороги с val)"),
                       ("macro_pr_auc", "macro PR-AUC")):
        rows.append(f"<tr><td>{label}</td><td>{_ci(model['point'][key], model['ci'][key])}</td>"
                    f"<td>{_ci(base['point'][key], base['ci'][key])}</td>"
                    f"<td>{_ci(diff[key]['point'], diff[key]['ci'])}</td></tr>")
    per_class = []
    for c in fm["classes"]:
        k = f"auroc_{c}"
        per_class.append(f"<tr><td>{c}</td><td>{_ci(model['point'][k], model['ci'][k])}</td>"
                         f"<td>{_ci(base['point'][k], base['ci'][k])}</td><td>{_ci(diff[k]['point'], diff[k]['ci'])}</td></tr>")
    head = "<tr><th>Метрика</th><th>CNN 1D (модель)</th><th>LightGBM (бейзлайн)</th><th>Разница</th></tr>"
    head_c = "<tr><th>Класс</th><th>AUROC CNN 1D</th><th>AUROC LightGBM</th><th>Разница</th></tr>"
    return (f"<div class=\"table-wrap\"><table>{head}{''.join(rows)}</table></div>"
            f"<h3>AUROC по классам</h3><div class=\"table-wrap\"><table>{head_c}{''.join(per_class)}</table></div>")


def example_section(ex: dict[str, Any], class_title: Any) -> str:
    """Карточка одного примера: таблица вероятностей и две картинки."""
    rows = []
    for c, p, thr, above in zip(ex["classes"], ex["probs"], ex["thresholds"], ex["above_threshold"]):
        verdict = "<b>выше порога</b>" if above else "ниже порога"
        cls = " class=\"hit\"" if above else ""
        rows.append(f"<tr{cls}><td>{html.escape(class_title(c))}</td><td>{p:.3f}</td><td>{thr:.3f}</td><td>{verdict}</td></tr>")
    d = ex["dir"]
    target = html.escape(class_title(ex["target_class"]))
    return f"""
<section class="example" id="{html.escape(d)}">
  <h3>{target} · ecg_id {ex['ecg_id']}</h3>
  <p class="meta">Разметка PTB-XL: <b>{html.escape(', '.join(class_title(c) for c in ex['labels']))}</b>;
  коды SCP: <code>{html.escape(ex['scp_codes'])}</code>. Запись взята из <b>валидационного фолда {ex['strat_fold']}</b>:
  на нём подбирались температура и пороги, поэтому это демонстрация работы модели, а не оценка качества.</p>
  <div class="table-wrap"><table>
    <tr><th>Класс</th><th>Вероятность</th><th>Порог (с val)</th><th>Ответ модели</th></tr>
    {''.join(rows)}
  </table></div>
  <p class="note">Вероятности после температурного шкалирования (T = {ex['temperature']:.3f}, подобрана на val).
  Классы не исключают друг друга, сумма не обязана быть 1.</p>
  <figure>
    <img src="examples/{html.escape(d)}/ecg.png" alt="12 отведений ЭКГ, ecg_id {ex['ecg_id']}" loading="lazy">
    <figcaption>12 отведений, 100 Гц, 10 с. Сетка 0.2 с × 0.5 мВ.</figcaption>
  </figure>
  <figure>
    <img src="examples/{html.escape(d)}/{html.escape(ex['ig_file'])}" alt="Карта Integrated Gradients для класса {target}" loading="lazy">
    <figcaption>Integrated Gradients по логиту класса {target}: красный фон повышает логит, синий понижает.
    Карта показывает, к чему чувствительна модель, <b>а не причину диагноза</b>.</figcaption>
  </figure>
</section>"""


DISCLAIMER = """<div class="disclaimer" role="note">
  <div class="disclaimer-title">⚠ Исследовательский прототип, не медицинский инструмент</div>
  Модель обучена на одном открытом датасете (PTB-XL) и не проходила клинической проверки. Её ответы нельзя использовать
  для диагностики, лечения или любых решений о здоровье. Если вас беспокоит ЭКГ, обратитесь к врачу.
</div>"""

CSS = """
:root { --bg: #ffffff; --fg: #1d1d1f; --muted: #5f6368; --line: #dadce0; --card: #f8f9fa; --hit: #fff4d6;
        --warn-bg: #fff4e5; --warn-fg: #4a2400; --warn-line: #b54708; --warn-title: #8a3200; --link: #0b57d0; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #121314; --fg: #e8eaed; --muted: #9aa0a6; --line: #3c4043; --card: #1e1f21; --hit: #3a3218;
          --warn-bg: #3a2410; --warn-fg: #ffe1c2; --warn-line: #f0883e; --warn-title: #ffb37a; --link: #8ab4f8; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg);
       font: 16px/1.55 system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif; }
main { max-width: 1040px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 1.7rem; line-height: 1.25; margin: 28px 0 8px; }
h2 { font-size: 1.3rem; margin: 40px 0 10px; padding-top: 8px; border-top: 1px solid var(--line); }
h3 { font-size: 1.1rem; margin: 22px 0 8px; }
a { color: var(--link); }
code { font-size: 0.88em; overflow-wrap: anywhere; }
.lead { color: var(--muted); margin-top: 0; }
.disclaimer { border: 2px solid var(--warn-line); background: var(--warn-bg); color: var(--warn-fg);
              border-radius: 10px; padding: 14px 18px; line-height: 1.45; }
.disclaimer-title { font-size: 1.1rem; font-weight: 700; margin-bottom: 4px; color: var(--warn-title); }
.table-wrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 0.95rem; font-variant-numeric: tabular-nums; }
th, td { border-bottom: 1px solid var(--line); padding: 6px 10px; text-align: left; white-space: nowrap; }
th { font-weight: 600; }
tr.hit td { background: var(--hit); }
.ci { color: var(--muted); font-size: 0.9em; }
.example { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 4px 18px 14px; margin: 18px 0; }
.meta, .note { font-size: 0.93rem; }
.note { color: var(--muted); }
figure { margin: 14px 0 0; }
figure img { width: 100%; height: auto; display: block; background: #fff; border-radius: 6px; }
figcaption { font-size: 0.88rem; color: var(--muted); margin-top: 4px; }
ul { padding-left: 1.3em; }
footer { margin-top: 40px; font-size: 0.9rem; color: var(--muted); }
"""


def build_html(examples: list[dict[str, Any]], fm: dict[str, Any], app: ModuleType, manifest: dict[str, Any]) -> str:
    """Собирает index.html (на русском, без JS и внешних ресурсов)."""
    classes_li = "".join(f"<li><b>{html.escape(app.class_title(c))}</b></li>" for c in fm["classes"])
    nav = " · ".join(f"<a href=\"#{html.escape(e['dir'])}\">{html.escape(e['target_class'])}</a>" for e in examples)
    hyp = fm["difference_model_minus_baseline"]["auroc_HYP"]
    excl = fm["excluded_no_superclass"]
    return f"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ЭКГ PTB-XL: демо</title>
<meta name="description" content="Исследовательский прототип: multi-label классификация 12-канальной ЭКГ PTB-XL по 5 суперклассам. Не медицинский инструмент.">
<style>{CSS}</style>
</head>
<body>
<main>
{DISCLAIMER}

<h1>Классификация 12-канальной ЭКГ (PTB-XL, 5 суперклассов)</h1>
<p class="lead">Статическая страница с пятью заранее посчитанными примерами. Ничего не загружается и не вычисляется
в браузере. Код и подробности: <a href="{REPO_URL}">репозиторий проекта</a>.</p>

<h2>Как это устроено</h2>
<p><b>Задача.</b> По 10-секундной записи 12-канальной ЭКГ (100 Гц) модель оценивает вероятность каждого из пяти
диагностических суперклассов. Классы не исключают друг друга (multi-label): у записи может быть несколько меток.</p>
<ul>{classes_li}</ul>
<p><b>Данные.</b> <a href="{PHYSIONET_URL}">PTB-XL</a> (Wagner et al., 2020, <i>Scientific Data</i>; PhysioNet,
лицензия CC BY 4.0). Стандартное разбиение по <code>strat_fold</code>: обучение на фолдах 1–8, выбор модели,
температура и пороги на фолде 9 (val), финальная оценка один раз на фолде 10 (test). Записи без диагностического
суперкласса исключены (train {excl['train']}, val {excl['val']}, test {excl['test']}).</p>
<p><b>Модель.</b> Одномерная свёрточная сеть (<code>{html.escape(manifest['model_run'])}</code>) с аугментациями,
вход — все 12 отведений, z-нормализация статистиками train. Вероятности откалиброваны температурным шкалированием,
пороги классов подобраны на val.</p>

<h2>Качество на тесте</h2>
<p>Фолд 10, {fm['n_records']} записей, {fm['n_patients']} пациентов. В скобках 95% доверительный интервал
(перцентильный бутстреп по пациентам, {fm['bootstrap']['n_boot']} реплик, парный для разницы).
Пороги и температура взяты с val без пересчёта.</p>
{metrics_tables(fm)}

<h2>Примеры</h2>
<p>По одной случайной записи на класс (seed {manifest['config']['seed']}) из валидационного фолда {manifest['config']['fold']}:
у записи ровно одна метка, и модель нашла этот класс (вероятность выше порога). Записи не отбирались вручную, но
<b>отбор среди верно найденных означает, что ошибок модели здесь нет</b>; они есть в метриках выше.
Переход к примеру: {nav}.</p>
{''.join(example_section(e, app.class_title) for e in examples)}

<h2>Ограничения</h2>
<ul>
  <li><b>Один seed.</b> Модель обучена один раз; разброс качества между повторными обучениями не оценивался,
  доверительные интервалы отражают только конечность тестовой выборки.</li>
  <li><b>Оценка на val оптимистична.</b> На фолде 9 выбирались модель, эпоха, температура и пороги, поэтому
  вероятности и решения на примерах выше выглядят лучше, чем на новых данных. Честная оценка — только тестовые метрики.</li>
  <li><b>Карта значимости — не объяснение.</b> Integrated Gradients показывает чувствительность логита модели к отсчётам
  сигнала относительно «нулевой» ЭКГ (среднего по train). Это не причина диагноза и не разметка патологии: модель может
  опираться на признаки, которые врач счёл бы неважными.</li>
  <li><b>HYP без преимущества над LightGBM.</b> Для гипертрофии разница AUROC с бейзлайном
  {hyp['point']:+.3f} [{hyp['ci'][0]:+.3f}; {hyp['ci'][1]:+.3f}], интервал включает ноль.</li>
  <li><b>Один датасет.</b> Внешней проверки на других больницах, аппаратах и популяциях не было; 100 Гц теряют
  часть деталей формы комплексов.</li>
</ul>

<footer>
{DISCLAIMER}
<p>Данные: PTB-XL, PhysioNet, CC BY 4.0. Страница сгенерирована <code>scripts/export_static_demo.py</code>
({html.escape(manifest['created'])}, git {html.escape(str(manifest['git']['commit']))}). Веса модели и сырые
сигналы здесь не публикуются.</p>
</footer>
</main>
</body>
</html>
"""


# ----------------------------------------------------------------------------- main

def main() -> int:
    """Отбор примеров, пересчёт и проверки, запись PNG/JSON/HTML."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/static_demo.yaml")
    args = parser.parse_args()
    t0 = time.perf_counter()
    cfg = yaml.safe_load(_resolve(args.config).read_text(encoding="utf-8"))
    fold = int(cfg["fold"])
    if fold != VAL_FOLD:
        raise ValueError(f"Демо строится только на валидационном фолде {VAL_FOLD}, в конфиге fold={fold}")
    run_dir, out_dir = _resolve(cfg["run"]), _resolve(cfg["output_dir"])
    rng = np.random.default_rng(int(cfg["seed"]))

    # --- val: сохранённые вероятности, метки, решения с порогами val
    val_probs, y_val, _, classes, _ = load_val_inputs(run_dir)
    post = json.loads((run_dir / "postprocess.json").read_text(encoding="utf-8"))
    _, pred = apply_postprocess(val_probs, post, post["probs_eps"])
    data_cfg = yaml.safe_load((run_dir / "data_config.yaml").read_text(encoding="utf-8"))["data"]
    ecg_val = np.load(_cache_dir(data_cfg) / "ecg_id_val.npy")
    root = _resolve(data_cfg["root"])
    db, _ = load_metadata(root)
    fold_ids = set(db.index[db["strat_fold"] == fold].tolist())
    if not set(ecg_val.tolist()) <= fold_ids:
        raise RuntimeError(f"В ecg_id_val.npy есть записи не из фолда {fold}")

    # --- модель и графики из app/ (те же, что в Gradio-демо)
    app = load_app_module()
    bundle = app.load_bundle(_resolve(cfg["model_dir"]))
    bundle.ig_steps = int(cfg["ig_steps"])
    if bundle.meta["run"] != run_dir.name or bundle.classes != classes:
        raise ValueError(f"app/model собран из {bundle.meta['run']}, а в конфиге {run_dir.name}")
    if not (np.allclose(bundle.temperature, post["temperature"]) and np.allclose(bundle.thresholds, post["thresholds"])):
        raise ValueError("Температура/пороги в app/model не совпадают с postprocess.json запуска")
    file_col = {100: "filename_lr", 500: "filename_hr"}[bundle.fs]

    # --- отбор и пересчёт (сначала все проверки, потом запись файлов)
    examples: list[dict[str, Any]] = []
    for c, i in pick_examples(y_val, pred, int(cfg["n_per_class"]), rng):
        eid = int(ecg_val[i])
        sf = int(db.loc[eid, "strat_fold"])
        if sf != fold:
            raise RuntimeError(f"ecg_id {eid}: strat_fold {sf} != {fold}")
        rel = root / db.loc[eid, file_col]
        rec = app.load_record([rel.with_suffix(".hea"), rel.with_suffix(".dat")], bundle)
        rec = dataclasses.replace(rec, name=f"ecg_id {eid} (PTB-XL, валидационный фолд {fold})")
        x = app.normalize(rec.signal, bundle)
        logits = app.predict_logits(x[None], bundle)
        p_raw = sigmoid(logits[0])
        p_cal = app.calibrated_probs(logits, bundle)[0]
        d = float(np.abs(p_raw - val_probs[i]).max())
        if d > float(cfg["max_abs_diff"]):
            raise RuntimeError(f"ecg_id {eid}: max |Δp| = {d:.4f} > {cfg['max_abs_diff']} — пересчёт не воспроизводит val_probs.npy")
        if p_cal[c] < bundle.thresholds[c]:
            raise RuntimeError(f"ecg_id {eid}: после пересчёта {classes[c]} ниже порога (val_probs и fp32 разошлись у порога)")
        examples.append({
            "c": c, "rec": rec, "x": x, "dir": f"{classes[c]}_{eid}", "target_class": classes[c], "ecg_id": eid,
            "strat_fold": sf, "labels": [k for k, v in zip(classes, y_val[i]) if v],
            "scp_codes": format_codes(db.loc[eid, "scp_codes"]), "classes": classes,
            "probs": p_cal.astype(float).tolist(), "probs_raw": p_raw.astype(float).tolist(),
            "val_probs_saved": val_probs[i].astype(float).tolist(), "max_abs_diff": d,
            "temperature": float(bundle.temperature[0]), "thresholds": bundle.thresholds.astype(float).tolist(),
            "above_threshold": [bool(p >= t) for p, t in zip(p_cal, bundle.thresholds)],
            "ig_file": f"ig_{classes[c]}.png",
        })

    # --- запись: PNG и probs.json
    ex_root = out_dir / "examples"
    if ex_root.exists():
        shutil.rmtree(ex_root)  # только сгенерированная папка примеров
    for ex in examples:
        d_out = ex_root / ex["dir"]
        d_out.mkdir(parents=True)
        app.plot_leads(ex["rec"], bundle.leads, bundle.fs).savefig(d_out / "ecg.png")
        c = ex["c"]
        attr = app.integrated_gradients(ex["x"], c, bundle)
        title = (f"Integrated Gradients для класса {app.class_title(classes[c])}: вероятность {ex['probs'][c]:.2f} "
                 f"(порог {ex['thresholds'][c]:.2f}), ecg_id {ex['ecg_id']}\n"
                 "Карта показывает чувствительность модели (красный повышает логит, синий понижает), а не причину диагноза")
        fig = app.plot_saliency(ex["rec"], attr, bundle.leads, bundle.fs, title)
        fig.subplots_adjust(top=0.915)
        fig.savefig(d_out / ex["ig_file"])
        probs_json = {
            "ecg_id": ex["ecg_id"], "strat_fold": ex["strat_fold"], "split": f"val (strat_fold {fold})",
            "labels_ptbxl": ex["labels"], "scp_codes": ex["scp_codes"], "classes": classes,
            "probs_calibrated": dict(zip(classes, ex["probs"])), "probs_raw": dict(zip(classes, ex["probs_raw"])),
            "val_probs_saved_raw": dict(zip(classes, ex["val_probs_saved"])), "max_abs_diff_vs_val_probs": ex["max_abs_diff"],
            "temperature": ex["temperature"], "thresholds": dict(zip(classes, ex["thresholds"])),
            "thresholds_fitted_on": post["fitted_on"],
            "decision": {k: "выше порога" if a else "ниже порога" for k, a in zip(classes, ex["above_threshold"])},
            "ig_target_class": classes[c], "ig_steps": bundle.ig_steps, "model_run": run_dir.name,
        }
        (d_out / "probs.json").write_text(json.dumps(probs_json, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- manifest и страница
    fm = json.loads(_resolve(cfg["final_metrics"]).read_text(encoding="utf-8"))
    manifest = {
        "config": cfg, "model_run": run_dir.name, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "git": git_state(),
        "examples": [{k: ex[k] for k in ("dir", "target_class", "ecg_id", "strat_fold", "labels", "max_abs_diff")}
                     for ex in examples],
        "selection": f"по {cfg['n_per_class']} случайной записи на класс (seed {cfg['seed']}) из фолда {fold}: одна метка, "
                     "вероятность класса по val_probs.npy после T выше порога с val",
        "env": {"python": platform.python_version(), "numpy": np.__version__,
                "torch": __import__("torch").__version__, "matplotlib": __import__("matplotlib").__version__},
    }
    (out_dir / "index.html").write_text(build_html(examples, fm, app, manifest), encoding="utf-8")

    # --- проверки выходной папки
    bad = [p for p in out_dir.rglob("*") if p.suffix.lower() in FORBIDDEN_SUFFIXES]
    if bad:
        raise RuntimeError(f"В {out_dir} попали веса/сырые сигналы: {bad}")
    size = dir_size_mb(out_dir)
    manifest["output_size_mb"] = round(size, 2)
    manifest["time_sec"] = round(time.perf_counter() - t0, 1)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if size > float(cfg["max_output_mb"]):
        raise RuntimeError(f"{out_dir}: {size:.1f} МБ > {cfg['max_output_mb']} МБ")

    print(f"{'класс':<6}{'ecg_id':>8}{'strat_fold':>12}{'max |Δp|':>12}  метки")
    for ex in examples:
        print(f"{ex['target_class']:<6}{ex['ecg_id']:>8}{ex['strat_fold']:>12}{ex['max_abs_diff']:>12.2e}  {'+'.join(ex['labels'])}")
    print(f"Готово за {manifest['time_sec']} с: {out_dir} ({size:.2f} МБ)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
