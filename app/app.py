"""Gradio-демо: multi-label классификация 12-канальной ЭКГ PTB-XL по 5 суперклассам.

Исследовательский прототип, НЕ медицинский инструмент. Запуск локально (из корня проекта):
    python scripts/export_app_bundle.py --config configs/app.yaml   # один раз: собрать app/model, app/examples
    python app/app.py                                               # http://127.0.0.1:7860
На Hugging Face Spaces этот файл указан как app_file в app/README.md.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MultipleLocator
from scipy.ndimage import uniform_filter1d

from inference import APP_DIR, Bundle, Record, calibrated_probs, integrated_gradients, load_bundle, load_record, normalize, predict_logits

CLASS_NAMES_RU: dict[str, str] = {
    "NORM": "норма",
    "MI": "инфаркт миокарда",
    "STTC": "изменения ST/T",
    "CD": "нарушения проводимости",
    "HYP": "гипертрофия",
}
REPO_URL = "https://github.com/Saltyrril/ekg_project"
EXAMPLES_DIR = APP_DIR / "examples"
SMOOTH_SAMPLES = 5  # только для показа карты: скользящая сумма 50 мс (как в src/explain.py)

DISCLAIMER_HTML = """
<div class="disclaimer" role="alert">
  <div class="disclaimer-title">⚠ Исследовательский прототип, не медицинский инструмент</div>
  <div>Модель обучена на одном открытом датасете (PTB-XL) и не проходила клинической проверки.
  Её ответы <b>нельзя</b> использовать для диагностики, лечения или любых решений о здоровье.
  Если вас беспокоит ЭКГ, обратитесь к врачу.</div>
</div>
"""

CSS = """
.disclaimer { border: 2px solid #b54708; background: #fff4e5; color: #4a2400; border-radius: 10px;
              padding: 14px 18px; font-size: 15px; line-height: 1.45; }
.disclaimer-title { font-size: 18px; font-weight: 700; margin-bottom: 4px; color: #8a3200; }
.disclaimer b { color: #8a3200; }
"""

BUNDLE: Bundle = load_bundle()
EXAMPLES_META: dict[str, dict[str, Any]] = {}
if (EXAMPLES_DIR / "examples.json").exists():
    EXAMPLES_META = {e["record"]: e for e in json.loads((EXAMPLES_DIR / "examples.json").read_text(encoding="utf-8"))}


def class_title(c: str) -> str:
    """'MI — инфаркт миокарда'."""
    return f"{c} — {CLASS_NAMES_RU.get(c, c)}"


# ----------------------------------------------------------------------------- графики

def _style_lead_axis(ax: Any, lim: float, t_end: float, label: str, bottom: bool) -> None:
    """Сетка «миллиметровки» ЭКГ: 0.2 с / 0.5 мВ крупно, 0.04 с / 0.1 мВ мелко."""
    ax.set_xlim(0, t_end)
    ax.set_ylim(-lim, lim)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.04))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.grid(which="major", color="#f2a3a3", lw=0.6)
    ax.grid(which="minor", color="#fbdcdc", lw=0.3)
    ax.set_axisbelow(True)
    ax.tick_params(which="both", length=0, labelleft=False, labelbottom=False)
    ax.text(0.005, 0.95, label, transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold")
    for s in ax.spines.values():
        s.set_visible(False)
    if bottom:  # подписи только целых секунд, хотя крупная сетка через 0.2 с
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f} с" if abs(v - round(v)) < 1e-6 else ""))
        ax.tick_params(axis="x", labelbottom=True, labelsize=8)


def _y_limit(centered: np.ndarray) -> float:
    """Одна шкала на все отведения: от ±1 до ±4 мВ с шагом 0.5 (выше 4 мВ — артефакт, обрезается)."""
    return float(np.clip(np.ceil(2 * 1.05 * np.abs(centered).max()) / 2, 1.0, 4.0))


def lead_label(lead: str) -> str:
    """Клиническое написание: AVR -> aVR."""
    return "a" + lead[1:] if lead.upper().startswith("AV") else lead


def plot_leads(rec: Record, leads: list[str], fs: int) -> Figure:
    """12 отведений в две колонки (I–aVF слева, V1–V6 справа), одна шкала мВ на все отведения."""
    sig = rec.signal
    n_leads, length = sig.shape
    n_rows = (n_leads + 1) // 2
    t = np.arange(length) / fs
    centered = sig - np.median(sig, axis=1, keepdims=True)
    lim = _y_limit(centered)
    fig = Figure(figsize=(13, 9), dpi=100)
    axes = fig.subplots(n_rows, 2, sharex=True)
    for k, lead in enumerate(leads):
        ax = axes[k % n_rows, k // n_rows]
        ax.plot(t, np.clip(centered[k], -lim, lim), color="#111111", lw=0.8)
        _style_lead_axis(ax, lim, t[-1], lead_label(lead), bottom=k % n_rows == n_rows - 1)
    fig.suptitle(f"Запись {rec.name}: 12 отведений, {fs} Гц, 10 с", fontsize=11, x=0.01, ha="left")
    fig.text(0.99, 0.005, f"сетка: 0.2 с × 0.5 мВ (мелкая 0.04 с × 0.1 мВ); по вертикали ±{lim:g} мВ; "
             "каждое отведение сдвинуто на свою медиану", fontsize=7.5, color="#6b6b6b", ha="right")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.05, hspace=0.08, wspace=0.03)
    return fig


def plot_saliency(rec: Record, attr: np.ndarray, leads: list[str], fs: int, title: str) -> Figure:
    """Верх: Σ|IG| по отведениям во времени. Ниже: отведения на фоне знакового IG (красный «за», синий «против»).

    Фон сглажен скользящей суммой в 50 мс только для показа; шкала цвета одна на всю запись
    (99.5-й перцентиль |IG|), поэтому яркость отведений сравнима.
    """
    sig = rec.signal
    n_leads, length = sig.shape
    n_rows = (n_leads + 1) // 2
    t = np.arange(length) / fs
    centered = sig - np.median(sig, axis=1, keepdims=True)
    lim = _y_limit(centered)
    shown = uniform_filter1d(attr, size=SMOOTH_SAMPLES, axis=1, mode="nearest") * SMOOTH_SAMPLES
    vmax = float(np.percentile(np.abs(shown), 99.5)) or 1.0
    abs_attr = np.abs(attr)
    lead_share = abs_attr.sum(axis=1) / max(float(abs_attr.sum()), 1e-12)

    fig = Figure(figsize=(13, 10), dpi=100)
    gs = fig.add_gridspec(1 + n_rows, 2, height_ratios=[1.6] + [1] * n_rows, hspace=0.18, wspace=0.08)
    top = fig.add_subplot(gs[0, :])
    top.fill_between(t, abs_attr.sum(axis=0), color="#7a5195", alpha=0.35, lw=0)
    top.plot(t, abs_attr.sum(axis=0), color="#3d2a4d", lw=0.9)
    top.set_xlim(0, t[-1])
    top.set_ylim(bottom=0)
    top.set_ylabel("Σ|IG| по отведениям", fontsize=9)
    top.tick_params(labelsize=8)
    top.legend(handles=[Patch(color="#d6604d", label="повышает логит класса («за»)"),
                        Patch(color="#4393c3", label="понижает («против»)")],
               loc="upper right", fontsize=8, ncol=2, frameon=False)

    for k, lead in enumerate(leads):
        ax = fig.add_subplot(gs[1 + k % n_rows, k // n_rows])
        ax.imshow(shown[k][None, :], aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest",
                  extent=(0, t[-1] + 1 / fs, -lim, lim), alpha=0.65, zorder=0)
        ax.plot(t, np.clip(centered[k], -lim, lim), color="#0b0b0b", lw=0.75, zorder=2)
        ax.set_xlim(0, t[-1])
        ax.set_ylim(-lim, lim)
        ax.xaxis.set_major_locator(MultipleLocator(1.0))
        ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
        ax.text(0.005, 0.95, f"{lead_label(lead)}  {lead_share[k]:.0%}", transform=ax.transAxes, ha="left", va="top",
                fontsize=9, fontweight="bold", zorder=3,
                bbox={"facecolor": "white", "alpha": 0.7, "lw": 0, "pad": 1})
        for s in ax.spines.values():
            s.set_visible(False)
        if k % n_rows == n_rows - 1:
            ax.tick_params(axis="x", labelbottom=True, labelsize=8)
    fig.suptitle(title, fontsize=11, x=0.01, ha="left")
    fig.text(0.99, 0.004, "процент у отведения — его доля суммарного |IG|; фон сглажен на 50 мс; "
             f"по вертикали ±{lim:g} мВ; время в секундах", fontsize=7.5, color="#6b6b6b", ha="right")
    fig.subplots_adjust(left=0.05, right=0.99, top=0.94, bottom=0.04)
    return fig


# ----------------------------------------------------------------------------- обработчики

def _saliency_figure(rec: Record, x: np.ndarray, probs: np.ndarray, cls: str) -> Figure:
    """Считает IG для класса `cls` и рисует карту."""
    target = BUNDLE.classes.index(cls)
    attr = integrated_gradients(x, target, BUNDLE)
    title = f"Карта значимости (Integrated Gradients) для класса {class_title(cls)}, вероятность {probs[target]:.2f}"
    return plot_saliency(rec, attr, BUNDLE.leads, BUNDLE.fs, title)


def _results_table(probs: np.ndarray) -> str:
    """Markdown-таблица: вероятность, порог с val и вывод модели по каждому классу."""
    rows = ["| Класс | Вероятность | Порог (подобран на val) | Ответ модели |", "|---|---:|---:|---|"]
    for c, p, thr in zip(BUNDLE.classes, probs, BUNDLE.thresholds):
        verdict = "**выше порога**" if p >= thr else "ниже порога"
        rows.append(f"| {class_title(c)} | {p:.3f} | {thr:.3f} | {verdict} |")
    rows.append("")
    rows.append("Классы не исключают друг друга (multi-label), поэтому вероятности не обязаны давать в сумме 1. "
                "Вероятности откалиброваны температурой, подобранной на валидационном фолде 9.")
    return "\n".join(rows)


def _record_info(rec: Record, probs: np.ndarray) -> str:
    """Сведения о записи, предупреждения предобработки и (для встроенных примеров) разметка PTB-XL."""
    lines = [f"**Запись:** `{rec.name}`, исходная частота {rec.fs_original} Гц."]
    above = [c for c, p, thr in zip(BUNDLE.classes, probs, BUNDLE.thresholds) if p >= thr]
    lines.append(f"**Выше порога:** {', '.join(class_title(c) for c in above) if above else 'ни один класс'}.")
    ex = EXAMPLES_META.get(rec.name)
    if ex:
        lines.append(f"**Разметка PTB-XL для этого примера:** {', '.join(class_title(c) for c in ex['labels'])} "
                     f"(ecg_id {ex['ecg_id']}, валидационный фолд {ex['strat_fold']}: на нём подбирались порог "
                     "и температура, так что это демонстрация, а не оценка качества).")
    for w in rec.warnings:
        lines.append(f"⚠ {w}")
    return "\n\n".join(lines)


def analyze(files: list[str] | None) -> tuple[Any, ...]:
    """Полный анализ: чтение, вероятности, график отведений, карта значимости для самого вероятного класса."""
    if not files:
        raise gr.Error("Загрузите два файла одной записи PTB-XL: <имя>.hea и <имя>.dat (или выберите пример).")
    try:
        rec = load_record(files, BUNDLE)
    except ValueError as e:
        raise gr.Error(str(e)) from e
    x = normalize(rec.signal, BUNDLE)
    probs = calibrated_probs(predict_logits(x[None], BUNDLE), BUNDLE)[0]
    top_cls = BUNDLE.classes[int(np.argmax(probs))]
    label = {class_title(c): float(p) for c, p in zip(BUNDLE.classes, probs)}
    state = {"rec": rec, "x": x, "probs": probs}
    return (
        label,
        _results_table(probs),
        _record_info(rec, probs),
        plot_leads(rec, BUNDLE.leads, BUNDLE.fs),
        gr.Dropdown(value=top_cls, interactive=True),
        _saliency_figure(rec, x, probs, top_cls),
        state,
    )


def update_saliency(cls: str, state: dict[str, Any] | None) -> Figure | None:
    """Пересчитывает карту значимости при выборе другого класса."""
    if not state:
        raise gr.Error("Сначала проанализируйте запись.")
    return _saliency_figure(state["rec"], state["x"], state["probs"], cls)


# ----------------------------------------------------------------------------- интерфейс

def _examples() -> tuple[list[list[list[str]]], list[str]]:
    """Встроенные примеры (записи валидационного фолда 9) и подписи к ним."""
    ex = sorted(EXAMPLES_META.values(), key=lambda e: BUNDLE.classes.index(e["labels"][0]))
    files = [[[str(EXAMPLES_DIR / f) for f in e["files"]]] for e in ex]
    labels = [f"{'+'.join(e['labels'])} · ecg_id {e['ecg_id']}" for e in ex]
    return files, labels


def build_demo() -> gr.Blocks:
    """Собирает интерфейс."""
    meta = BUNDLE.meta
    # delete_cache: раз в час удалять загрузки старше часа из временного кэша Gradio
    with gr.Blocks(title="ЭКГ PTB-XL: исследовательский прототип", delete_cache=(3600, 3600)) as demo:
        gr.HTML(DISCLAIMER_HTML)
        gr.Markdown(
            "## Классификация 12-канальной ЭКГ (PTB-XL, 5 суперклассов)\n"
            "Загрузите запись из [PTB-XL](https://physionet.org/content/ptb-xl/): **два файла** `<имя>.hea` и "
            "`<имя>.dat` из `records100/` (100 Гц) или `records500/` (500 Гц, будет пересчитана в 100 Гц). "
            "Модель вернёт вероятности пяти диагностических суперклассов и карту значимости."
        )
        state = gr.State()
        with gr.Row():
            with gr.Column(scale=2):
                files = gr.File(label="Запись WFDB: .hea + .dat", file_count="multiple",
                                file_types=[".hea", ".dat"], type="filepath")
                run_btn = gr.Button("Анализировать", variant="primary")
            with gr.Column(scale=3):
                probs_label = gr.Label(num_top_classes=len(BUNDLE.classes), label="Вероятности классов")
        table = gr.Markdown()
        info = gr.Markdown()
        leads_plot = gr.Plot(label="12 отведений", format="png")
        gr.Markdown("### Карта значимости")
        cls_dd = gr.Dropdown(choices=[(class_title(c), c) for c in BUNDLE.classes], value=BUNDLE.classes[0],
                             label="Класс, для которого строится карта", interactive=True)
        sal_plot = gr.Plot(label="Integrated Gradients", format="png")
        gr.Markdown(
            "**Как читать карту.** Integrated Gradients показывает вклад каждого отсчёта в логит выбранного "
            "класса относительно «нулевой» ЭКГ (среднего по обучающей выборке). Красный фон повышает "
            "уверенность модели в классе, синий понижает. Это атрибуция ответа модели, **а не объяснение "
            "причины** и не разметка патологии: модель может опираться на признаки, которые врач счёл бы "
            "неважными. Проверку карт (тест стирания, контроль случайными весами) см. в репозитории."
        )
        outputs = [probs_label, table, info, leads_plot, cls_dd, sal_plot, state]
        ex_files, ex_labels = _examples()
        if ex_files:
            gr.Examples(examples=ex_files, inputs=[files], outputs=outputs, fn=analyze, run_on_click=True,
                        cache_examples=False, example_labels=ex_labels,
                        label="Примеры: случайные записи валидационного фолда 9 PTB-XL (по одной на класс)")
        run_btn.click(analyze, inputs=[files], outputs=outputs)
        cls_dd.input(update_saliency, inputs=[cls_dd, state], outputs=[sal_plot])
        gr.Markdown(
            f"---\nМодель: `{meta['run']}` ({meta['model']['name']}, 100 Гц, 12 отведений). Обучение на "
            "фолдах 1–8 PTB-XL, ранняя остановка, температура и пороги на фолде 9. Тестовые метрики с 95% ДИ "
            f"и ограничения описаны в [README проекта]({REPO_URL}).\n\n"
            "Данные: PTB-XL (Wagner et al., 2020, *Scientific Data*), PhysioNet, лицензия CC BY 4.0. "
            "Загруженные файлы лежат только во временном кэше Gradio и удаляются примерно через час; "
            "приложение их никуда не отправляет и не сохраняет."
        )
        gr.HTML(DISCLAIMER_HTML)
    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.launch(css=CSS, theme=gr.themes.Soft())
