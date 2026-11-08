"""PDF-отчёт по одной записи ЭКГ (matplotlib, без дополнительных зависимостей).

Страницы A4 (альбомные): 1) сводка: запись, качество сигнала, вероятности классов; 2) 12 отведений;
3..) карты значимости для классов выше порога (или для самого вероятного); последняя) модель, методика, ограничения.
Отчёт — исследовательский артефакт, а не медицинский документ; это написано на каждой странице.
"""
from __future__ import annotations

import hashlib
import textwrap
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from matplotlib import rc_context
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

if TYPE_CHECKING:
    from inference import Bundle, Record
    from quality import QualityReport

A4 = (11.69, 8.27)
STATUS_COLOR = {"ok": "#1a7f37", "warning": "#b54708", "poor": "#b42318"}
DISCLAIMER = ("ИССЛЕДОВАТЕЛЬСКИЙ ПРОТОТИП, НЕ МЕДИЦИНСКОЕ ИЗДЕЛИЕ. Модель обучена на одном открытом датасете (PTB-XL) и не "
              "проходила клинической проверки. Отчёт нельзя использовать для диагностики, лечения или любых решений о "
              "здоровье. При вопросах по ЭКГ обратитесь к врачу.")
LIMITATIONS = [
    "Модель различает только пять диагностических суперклассов PTB-XL (NORM, MI, STTC, CD, HYP). Она не определяет "
    "аритмии (например, фибрилляцию предсердий), не измеряет пульс и интервалы и не оценивает редкие состояния.",
    "Обучение на одном датасете (PTB, Германия, записи 1989–1996, одни приборы). Качество на других приборах, "
    "популяциях и способах записи неизвестно и может быть заметно ниже.",
    "Метки PTB-XL взяты из заключений, часть из них получена автоматически. Метка ставится при любом likelihood.",
    "Классы не исключают друг друга (multi-label): вероятности не обязаны давать в сумме 1. Порог каждого класса "
    "подобран на валидационном фолде 9 по максимуму F1; «выше порога» — не диагноз.",
    "Проверка качества сигнала эвристическая: пороги подобраны на PTB-XL (фолд 9). Её «хорошее» не гарантирует "
    "верности ответа модели, а замечания не обязательно означают плохую запись.",
    "Карта значимости (Integrated Gradients) показывает чувствительность ответа модели к участкам сигнала, а не "
    "причину диагноза и не разметку патологии.",
    "Один запуск обучения (один seed): разница между вариантами модели на val лежит в пределах шума.",
]


def _wrap(text: str, width: int) -> str:
    return textwrap.fill(text, width=width)


def _frame(fig: Figure, title: str | None, page: int, total: int, generated: str, y: float = 0.012) -> None:
    """Заголовок (если есть; у графиков он уже нарисован) и колонтитул страницы."""
    if title:
        fig.text(0.04, 0.955, title, fontsize=15, fontweight="bold", va="center")
    fig.text(0.04, y, "Исследовательский прототип, не медицинский документ.", fontsize=7.5, color="#6b6b6b")
    fig.text(0.96, y, f"{generated} · страница {page} из {total}", fontsize=7.5, color="#6b6b6b", ha="right")


def _disclaimer(fig: Figure, y: float, h: float) -> None:
    fig.add_artist(Rectangle((0.04, y), 0.92, h, transform=fig.transFigure, facecolor="#fff4e5", edgecolor="#b54708", lw=1.5))
    fig.text(0.05, y + h / 2, _wrap(DISCLAIMER, 150), fontsize=8.5, color="#4a2400", va="center")


def summary_page(rec: Record, probs: np.ndarray, bundle: Bundle, quality: QualityReport, titles: dict[str, str],
                 generated: str, total: int) -> Figure:
    """Страница 1: запись, качество сигнала, вероятности и вердикты по классам."""
    fig = Figure(figsize=A4)
    _frame(fig, "Отчёт по ЭКГ: результат модели", 1, total, generated)
    _disclaimer(fig, 0.855, 0.065)
    poor = quality.status == "poor"
    above = [c for c, p, t in zip(bundle.classes, probs, bundle.thresholds) if p >= t]

    left = ["Запись", f"  Имя: {rec.name}", f"  Формат: {rec.fmt}, исходная частота {rec.fs_original} Гц",
            f"  Использовано: 12 отведений, {bundle.length / bundle.fs:.0f} с, {bundle.fs} Гц", ""]
    fig.text(0.04, 0.825, "\n".join(left[:1]), fontsize=11, fontweight="bold", va="top")
    fig.text(0.04, 0.80, "\n".join(left[1:]), fontsize=9, va="top", linespacing=1.5)

    fig.text(0.04, 0.70, "Качество сигнала: ", fontsize=11, fontweight="bold", va="top")
    fig.text(0.205, 0.70, quality.status_ru.upper(), fontsize=11, fontweight="bold", va="top", color=STATUS_COLOR[quality.status])
    lines: list[str] = []
    for issue in quality.issues:
        where = f" [{', '.join(issue.leads)}]" if issue.leads else ""
        mark = "✖" if issue.severity == "poor" else "•"
        lines.append(_wrap(f"{mark} {issue.message}{where}", 62).replace("\n", "\n  "))
    for w in rec.warnings:
        lines.append(_wrap(f"• {w}", 62).replace("\n", "\n  "))
    if not lines:
        lines = ["Замечаний нет."]
    fig.text(0.04, 0.665, "\n".join(lines), fontsize=8.2, va="top", linespacing=1.35)
    if poor:
        fig.text(0.04, 0.115, _wrap("Качество сигнала низкое: ответ модели ненадёжен, не опирайтесь на него.", 62),
                 fontsize=9.5, fontweight="bold", color=STATUS_COLOR["poor"], va="top")

    # таблица результатов
    fig.text(0.53, 0.825, "Результат модели", fontsize=11, fontweight="bold", va="top")
    ax = fig.add_axes((0.53, 0.52, 0.43, 0.27))
    ax.axis("off")
    cells = [[titles[c], f"{p:.3f}", f"{t:.3f}", ("выше порога" if p >= t else "ниже порога") + (" (ненадёжно)" if poor else "")]
             for c, p, t in zip(bundle.classes, probs, bundle.thresholds)]
    table = ax.table(cellText=cells, colLabels=["Класс", "Вероятность", "Порог (val)", "Ответ"], loc="upper left",
                     cellLoc="left", colWidths=[0.43, 0.18, 0.18, 0.21], edges="horizontal")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.55)
    for (r, col), cell in table.get_celld().items():
        cell.set_edgecolor("#c9c9c9")
        if r == 0:
            cell.set_text_props(fontweight="bold")
        elif col == 3 and cells[r - 1][3].startswith("выше"):
            cell.set_text_props(fontweight="bold", color=STATUS_COLOR["warning" if poor else "poor"])
    verdict = ("Выше порога: " + ", ".join(titles[c] for c in above)) if above else "Ни один класс не выше порога."
    fig.text(0.53, 0.495, _wrap(verdict + (" (ответ ненадёжен из-за качества сигнала)." if poor and above else ""), 70),
             fontsize=9, va="top")

    # столбики вероятностей с порогами
    bx = fig.add_axes((0.64, 0.09, 0.32, 0.29))
    y = np.arange(len(bundle.classes))[::-1]
    bx.barh(y, probs, color=["#b54708" if p >= t else "#98a2b3" for p, t in zip(probs, bundle.thresholds)], height=0.55)
    bx.scatter(bundle.thresholds, y, marker="|", s=420, color="#101828", zorder=3)
    bx.set_yticks(y, labels=bundle.classes, fontsize=9)
    bx.set_xlim(0, 1)
    bx.tick_params(axis="x", labelsize=8)
    bx.set_xlabel("откалиброванная вероятность; чёрная черта — порог класса (val)", fontsize=8)
    for s in ("top", "right"):
        bx.spines[s].set_visible(False)
    fig.text(0.53, 0.45, _wrap("Столбик оранжевый, если вероятность не меньше порога класса. Вероятности откалиброваны "
                               "температурой, подобранной на валидационном фолде 9.", 82), fontsize=7.8, color="#444444", va="top")
    return fig


def method_page(rec: Record, bundle: Bundle, generated: str, page: int, total: int, model_dir: Path) -> Figure:
    """Последняя страница: версия модели, методика и ограничения."""
    meta = bundle.meta
    try:
        sha = hashlib.sha256((model_dir / "best.pt").read_bytes()).hexdigest()[:12]
    except OSError:
        sha = "н/д"
    fig = Figure(figsize=A4)
    _frame(fig, "Модель, методика и ограничения", page, total, generated)
    thr = ", ".join(f"{c} {t:.2f}" for c, t in zip(bundle.classes, bundle.thresholds))
    model = [
        f"Модель: {meta['model']['name']} (1D-CNN), запуск {meta['run']}, коммит обучения {meta.get('git_commit', 'н/д')}",
        f"Веса best.pt: sha256 {sha}…; пакет собран {meta.get('created', 'н/д')}",
        "Вход: 12 отведений × 10 с × 100 Гц, мВ; z-нормализация по отведениям статистиками обучающей выборки.",
        "Обучение: PTB-XL, фолды 1–8; ранняя остановка, температура и пороги: фолд 9; тестовый фолд 10 использован "
        "один раз для финальной оценки.",
        f"Температура T = {float(bundle.temperature[0]):.3f}; пороги классов: {thr}.",
        "Предобработка: порядок отведений по именам, перевод в мВ по единицам файла, пересчёт в 100 Гц "
        "(resample_poly), первые 10 с записи.",
        f"Объяснения: Integrated Gradients по логиту класса, {bundle.ig_steps} шагов, нулевой baseline (среднее train).",
    ]
    fig.text(0.04, 0.90, "Версия и методика", fontsize=11, fontweight="bold", va="top")
    fig.text(0.04, 0.87, "\n".join("• " + _wrap(m, 140).replace("\n", "\n  ") for m in model), fontsize=8.5, va="top", linespacing=1.4)
    fig.text(0.04, 0.70, "Ограничения", fontsize=11, fontweight="bold", va="top")
    fig.text(0.04, 0.67, "\n".join("• " + _wrap(m, 140).replace("\n", "\n  ") for m in LIMITATIONS), fontsize=8.5, va="top", linespacing=1.4)
    fig.text(0.04, 0.075, _wrap("Данные: PTB-XL (Wagner P. et al., Scientific Data 7, 154, 2020), PhysioNet, лицензия CC BY 4.0. "
                                "Код и метрики с доверительными интервалами: https://github.com/Saltyrril/ekg_project", 150),
             fontsize=7.8, color="#444444", va="top")
    return fig


def _fit_page(fig: Figure) -> Figure:
    """Подгоняет график приложения под лист A4 (вектор, шрифты в пунктах остаются прежними)."""
    fig.set_size_inches(*A4)
    fig.subplots_adjust(bottom=0.08)  # место под колонтитул над собственной подписью графика
    return fig


def build_pdf(path: str | Path, *, rec: Record, probs: np.ndarray, bundle: Bundle, quality: QualityReport,
              titles: dict[str, str], leads_fig: Figure, saliency: list[tuple[str, Figure]], model_dir: Path) -> Path:
    """Собирает PDF. `saliency` — список (класс, рисунок карты); рисунки меняют размер под A4 (передавайте свежие)."""
    path = Path(path)
    generated = time.strftime("%Y-%m-%d %H:%M")
    total = 3 + len(saliency)
    with rc_context({"pdf.fonttype": 42}), PdfPages(path, metadata={"Title": f"ЭКГ: отчёт модели — {rec.name}",
                                                                      "Creator": "ekg_project (research prototype)"}) as pdf:
        pdf.savefig(summary_page(rec, probs, bundle, quality, titles, generated, total))
        leads_fig = _fit_page(leads_fig)
        _frame(leads_fig, None, 2, total, generated, y=0.032)
        pdf.savefig(leads_fig)
        for k, (cls, fig) in enumerate(saliency):
            fig = _fit_page(fig)
            _frame(fig, None, 3 + k, total, generated, y=0.032)
            pdf.savefig(fig)
        pdf.savefig(method_page(rec, bundle, generated, total, total, model_dir))
    return path


def classes_for_saliency(probs: np.ndarray, thresholds: np.ndarray, classes: list[str]) -> list[str]:
    """Классы выше порога (по убыванию вероятности); если таких нет, самый вероятный."""
    order = np.argsort(-probs)
    above = [classes[i] for i in order if probs[i] >= thresholds[i]]
    return above or [classes[int(order[0])]]

