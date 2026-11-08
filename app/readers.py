"""Чтение записей ЭКГ из файлов: WFDB (.hea + .dat), EDF/EDF+ (.edf) и CSV (.csv, .txt).

Читатели возвращают «сырую» запись (RawRecord): сигнал в единицах файла, имена каналов, частоту.
Приведение к входу модели (порядок отведений, мВ, 100 Гц, 10 с) делает inference.prepare_record.
Модуль не зависит от torch и Gradio. EDF читается собственным парсером на numpy: в wfdb 4.3 читателя
EDF нет, а pyedflib/mne ради одного формата подключать не нужно.
"""
from __future__ import annotations

import io
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

STD_LEADS: list[str] = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
SUPPORTED_EXTENSIONS: list[str] = [".hea", ".dat", ".edf", ".csv", ".txt"]
_TIME_COLUMNS = {"time", "t", "timestamp", "sec", "seconds", "s", "ms", "время", "time_s", "time_ms", "times"}


@dataclass
class RawRecord:
    """Запись как в файле: сигнал (C, T) в единицах файла, имена и единицы каналов, частота, формат."""

    signal: np.ndarray
    sig_names: list[str]
    fs: int
    name: str
    units: list[str]
    fmt: str
    notes: list[str] = field(default_factory=list)


def canon_lead(name: str) -> str:
    """Каноническое имя отведения: 'ECG Lead aVR' -> 'AVR', 'aV_L' -> 'AVL', 'EKG V1' -> 'V1'."""
    s = re.sub(r"[^A-Z0-9]", "", name.upper())
    while True:
        for prefix in ("ECG", "EKG", "LEAD"):
            if s.startswith(prefix) and len(s) > len(prefix):
                s = s[len(prefix):]
                break
        else:
            return s


# ----------------------------------------------------------------------------- WFDB

def read_wfdb(paths: list[str | Path]) -> RawRecord:
    """Читает пару .hea + .dat (одну запись).

    Gradio кладёт каждый загруженный файл в свою временную папку, а .hea ссылается на .dat по имени,
    поэтому файлы сначала копируются в одну папку с исходными именами.
    """
    files = [Path(p) for p in paths]
    heads = [f for f in files if f.suffix.lower() == ".hea"]
    if len(heads) != 1:
        raise ValueError(f"Нужен ровно один файл .hea (заголовок записи WFDB), получено: {len(heads)}.")
    dats = [f for f in files if f.suffix.lower() == ".dat"]
    if not dats:
        raise ValueError("Нет файла .dat с сигналом. Загрузите оба файла записи: <имя>.hea и <имя>.dat.")
    if heads[0].stem not in {f.stem for f in dats}:
        raise ValueError(f"Файлы от разных записей: {heads[0].name} и {', '.join(f.name for f in dats)}. "
                         f"Нужна пара {heads[0].stem}.hea + {heads[0].stem}.dat.")
    with tempfile.TemporaryDirectory() as tmp:
        for f in files:
            shutil.copy(f, Path(tmp) / f.name)
        try:
            sig, fields = wfdb.rdsamp(str(Path(tmp) / heads[0].stem))
        except Exception as e:  # wfdb бросает разные исключения при битом/несовпадающем файле
            raise ValueError(f"Не удалось прочитать запись WFDB ({type(e).__name__}: {e}). "
                             f"Проверьте, что .hea и .dat от одной записи.") from e
    n_ch = sig.shape[1]
    units = list(fields.get("units") or [])
    return RawRecord(signal=np.ascontiguousarray(sig.T, dtype=np.float32), sig_names=list(fields["sig_name"]),
                     fs=int(round(fields["fs"])), name=heads[0].stem,
                     units=units if len(units) == n_ch else [""] * n_ch, fmt="WFDB")


# ----------------------------------------------------------------------------- EDF

def read_edf(path: str | Path) -> RawRecord:
    """Читает EDF / EDF+C (непрерывный). Аннотационные каналы пропускаются.

    Цифровые значения переводятся в физические по заголовку (physical/digital min/max). Если у каналов разные
    частоты, берётся самая распространённая группа каналов, остальные отбрасываются с пометкой.
    """
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) < 256:
        raise ValueError("Файл слишком короткий для EDF (нет заголовка 256 байт).")

    def text(a: int, b: int) -> str:
        return raw[a:b].decode("ascii", "replace").strip()

    try:
        n_header, n_rec, dur, ns = int(text(184, 192)), int(text(236, 244)), float(text(244, 252)), int(text(252, 256))
    except ValueError as e:
        raise ValueError("Файл не похож на EDF: не читаются поля заголовка.") from e
    if text(192, 236).startswith("EDF+D"):
        raise ValueError("EDF+D (запись с разрывами) не поддерживается. Нужна непрерывная запись (EDF или EDF+C).")
    if ns <= 0 or dur <= 0 or n_header < 256 * (ns + 1) or len(raw) < n_header:
        raise ValueError("Заголовок EDF повреждён (число каналов, длительность записи или размер заголовка).")

    pos = 256

    def fields(width: int) -> list[str]:
        nonlocal pos
        out = [raw[pos + i * width: pos + (i + 1) * width].decode("ascii", "replace").strip() for i in range(ns)]
        pos += ns * width
        return out

    labels, _transducer, dims = fields(16), fields(80), fields(8)
    try:
        pmin, pmax = [float(v) for v in fields(8)], [float(v) for v in fields(8)]
        dmin, dmax = [float(v) for v in fields(8)], [float(v) for v in fields(8)]
        _prefilter = fields(80)
        nsamp = [int(v) for v in fields(8)]
    except ValueError as e:
        raise ValueError("Заголовки каналов EDF повреждены (числовые поля не читаются).") from e

    rec_len = sum(nsamp)
    if rec_len <= 0:
        raise ValueError("В EDF нет отсчётов.")
    data = np.frombuffer(raw, dtype="<i2", count=(len(raw) - n_header) // 2 // rec_len * rec_len, offset=n_header)
    notes: list[str] = []
    n_full = len(data) // rec_len
    if n_rec < 0:
        n_rec = n_full
    elif n_full < n_rec:
        notes.append(f"Файл EDF обрезан: в заголовке {n_rec} блоков, прочитано {n_full}.")
        n_rec = n_full
    if n_rec == 0:
        raise ValueError("В EDF нет ни одного полного блока данных.")
    data = data[: n_rec * rec_len].reshape(n_rec, rec_len)

    chans: list[tuple[str, str, int, np.ndarray]] = []  # (метка, единицы, round(fs), физический сигнал)
    start = 0
    for i in range(ns):
        sl = data[:, start: start + nsamp[i]]
        start += nsamp[i]
        if labels[i].lower().startswith("edf annotations") or nsamp[i] <= 0:
            continue
        gain = (pmax[i] - pmin[i]) / (dmax[i] - dmin[i]) if dmax[i] != dmin[i] else 1.0
        phys = (sl.reshape(-1).astype(np.float64) - dmin[i]) * gain + pmin[i]
        chans.append((labels[i], dims[i], int(round(nsamp[i] / dur)), phys.astype(np.float32)))
    if not chans:
        raise ValueError("В EDF нет сигнальных каналов (только аннотации).")

    fs_counts: dict[int, int] = {}
    for _, _, f, _ in chans:
        fs_counts[f] = fs_counts.get(f, 0) + 1
    fs = max(fs_counts, key=lambda f: (fs_counts[f], f))
    dropped = [c[0] for c in chans if c[2] != fs]
    if dropped:
        notes.append(f"Каналы с другой частотой, чем основная ({fs} Гц), пропущены: {', '.join(dropped)}.")
    chans = [c for c in chans if c[2] == fs]
    return RawRecord(signal=np.stack([c[3] for c in chans]), sig_names=[c[0] for c in chans], fs=fs, name=path.stem,
                     units=[c[1] for c in chans], fmt="EDF", notes=notes)


# ----------------------------------------------------------------------------- CSV

def _split_lines(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]


def _detect_separator(lines: list[str]) -> str:
    """Разделитель по первым строкам: ';', табуляция, ',' или пробелы. Берётся тот, что встречается одинаково часто."""
    head = lines[:5]
    for sep in (";", "\t", ","):
        counts = {ln.count(sep) for ln in head}
        if len(counts) == 1 and counts != {0}:
            return sep
    return r"\s+"


def read_csv(path: str | Path, fs: float | None = None, unit: str | None = None) -> RawRecord:
    """Читает CSV/TXT: столбцы = отведения, строки = отсчёты.

    Заголовок опционален. Имена отведений распознаются в любом регистре (I, II, III, aVR, aVL, aVF, V1–V6, а также
    'ECG I' и т. п.). Без имён нужно ровно 12 столбцов в порядке I, II, III, aVR, aVL, aVF, V1–V6. Столбец времени
    (time, t, ms, ...) даёт частоту; иначе её нужно указать явно (fs). Десятичная запятая допускается.
    """
    path = Path(path)
    try:
        content = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        content = path.read_text(encoding="latin-1")
    lines = _split_lines(content)
    if len(lines) < 2:
        raise ValueError("В CSV меньше двух строк с данными.")
    sep = _detect_separator(lines)
    try:
        df = pd.read_csv(io.StringIO(content), sep=sep, header=None, dtype=str, comment="#", skip_blank_lines=True,
                         engine="python")
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        raise ValueError(f"CSV не читается как таблица (разное число столбцов в строках?): {e}") from e
    df = df.apply(lambda col: col.str.strip().str.replace(",", ".", regex=False) if sep != "," else col.str.strip())
    num = df.apply(pd.to_numeric, errors="coerce")

    notes: list[str] = []
    has_header = num.iloc[0].notna().mean() < 0.5
    names: list[str]
    if has_header:
        names = [str(v) for v in df.iloc[0]]
        num = num.iloc[1:].reset_index(drop=True)
    else:
        names = [f"col{i}" for i in range(num.shape[1])]
    num = num.dropna(how="all").reset_index(drop=True)

    time_idx = next((i for i, n in enumerate(names) if has_header and n.strip().lower() in _TIME_COLUMNS), None)
    if time_idx is None and not has_header and num.shape[1] == len(STD_LEADS) + 1 and (num.iloc[:, 0].diff().dropna() > 0).all():
        time_idx = 0
        notes.append("Первый столбец без заголовка принят за время (строго возрастает, остальных столбцов 12).")

    lead_cols = [i for i in range(num.shape[1]) if i != time_idx]
    by_canon = {canon_lead(names[i]): i for i in reversed(lead_cols)} if has_header else {}
    if all(ld in by_canon for ld in STD_LEADS):
        order = [by_canon[ld] for ld in STD_LEADS]
    elif len(lead_cols) == len(STD_LEADS):
        order = lead_cols
        notes.append("Имена отведений не распознаны (или нет заголовка): столбцы приняты в порядке "
                     "I, II, III, aVR, aVL, aVF, V1–V6.")
    else:
        found = [names[i] for i in lead_cols] if has_header else f"{len(lead_cols)} столбцов без заголовка"
        raise ValueError(f"Нужны 12 стандартных отведений (I, II, III, aVR, aVL, aVF, V1–V6). В CSV: {found}.")
    signal = num.iloc[:, order].to_numpy(dtype=np.float32).T

    fs_time: float | None = None
    if time_idx is not None:
        t = num.iloc[:, time_idx].to_numpy(dtype=np.float64)
        dt = np.diff(t)
        if len(dt) and np.isfinite(dt).all() and np.median(dt) > 0:
            step = float(np.median(dt))
            ms = (has_header and names[time_idx].strip().lower().endswith("ms")) or step >= 1.0
            fs_time = 1000.0 / step if ms else 1.0 / step
            if ms:
                notes.append("Время в столбце принято за миллисекунды.")
            if dt.max() > 1.5 * step:
                notes.append("Шаг по времени неравномерный (пропуски отсчётов): сигнал считается равномерным.")
    if fs is None or fs <= 0:
        if fs_time is None:
            raise ValueError("В CSV нет столбца времени: укажите частоту дискретизации (Гц) в поле интерфейса.")
        fs_use = fs_time
        notes.append(f"Частота {fs_time:.1f} Гц определена по столбцу времени.")
    else:
        fs_use = float(fs)
        if fs_time is not None and abs(fs_time - fs_use) / fs_use > 0.02:
            notes.append(f"Указана частота {fs_use:g} Гц, но по столбцу времени получается {fs_time:.1f} Гц: взята указанная.")
    return RawRecord(signal=np.ascontiguousarray(signal), sig_names=[STD_LEADS[k] for k in range(len(STD_LEADS))],
                     fs=int(round(fs_use)), name=path.stem, units=[unit or ""] * len(STD_LEADS), fmt="CSV", notes=notes)


# ----------------------------------------------------------------------------- выбор читателя

def read_files(paths: list[str | Path], fs: float | None = None, unit: str | None = None) -> RawRecord:
    """Выбирает читателя по расширениям. `fs` и `unit` используются только для CSV."""
    files = [Path(p) for p in paths]
    exts = {f.suffix.lower() for f in files}
    if exts & {".hea", ".dat"}:
        if exts - {".hea", ".dat"}:
            raise ValueError("Нельзя смешивать форматы: загрузите либо пару .hea + .dat, либо один .edf, либо один .csv.")
        return read_wfdb(files)
    if len(files) != 1:
        raise ValueError(f"Для формата {next(iter(exts))} нужен один файл, загружено: {len(files)}.")
    ext = files[0].suffix.lower()
    if ext == ".edf":
        return read_edf(files[0])
    if ext in (".csv", ".txt"):
        return read_csv(files[0], fs=fs, unit=unit)
    raise ValueError(f"Формат '{ext}' не поддерживается. Подходят: WFDB (.hea + .dat), .edf, .csv, .txt.")
