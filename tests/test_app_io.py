"""Тесты чтения записей (WFDB / EDF / CSV) и приведения к входу модели. Запуск: python -m unittest discover -s tests -v

Нужны локальные app/model (scripts/export_app_bundle.py) и app/examples. Данные PTB-XL и фолд 10 не читаются.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import wfdb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from inference import load_bundle, load_record  # noqa: E402
from readers import STD_LEADS, canon_lead  # noqa: E402

EXAMPLE = ROOT / "app" / "examples" / "01421_lr"
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def write_edf(path: Path, sig: np.ndarray, fs: int, labels: list[str], dim: str = "mV", scale: float = 1.0,
              with_annotations: bool = False) -> None:
    """Минимальный писатель EDF (блоки по 1 с), int16, физический диапазон по данным."""
    sig = sig * scale
    ns, n = sig.shape
    n_rec = n // fs
    lim = float(np.ceil(np.abs(sig).max() * 1.01))
    chans = list(zip(labels, sig))
    if with_annotations:
        chans.append(("EDF Annotations", np.zeros(n)))
    total = len(chans)
    head = (["0".ljust(8), "patient".ljust(80), "rec".ljust(80), "01.01.20", "00.00.00", str(256 * (total + 1)).ljust(8),
             "EDF+C".ljust(44), str(n_rec).ljust(8), "1".ljust(8), str(total).ljust(4)])
    per = [[lab.ljust(16) for lab, _ in chans], [" " * 80] * total,
           [(dim if lab != "EDF Annotations" else "").ljust(8) for lab, _ in chans],
           [str(-lim).ljust(8)] * total, [str(lim).ljust(8)] * total, ["-32768".ljust(8)] * total,
           ["32767".ljust(8)] * total, [" " * 80] * total,
           [str(fs if lab != "EDF Annotations" else 30).ljust(8) for lab, _ in chans], [" " * 32] * total]
    header = "".join(head) + "".join("".join(col) for col in per)
    digital = [np.round((s[: n_rec * fs] + lim) / (2 * lim) * 65535 - 32768).clip(-32768, 32767).astype("<i2") for _, s in chans]
    blocks = []
    for r in range(n_rec):
        for (lab, _), d in zip(chans, digital):
            k = fs if lab != "EDF Annotations" else 30
            blocks.append(d[r * fs: r * fs + k] if lab != "EDF Annotations" else np.zeros(k, dtype="<i2"))
    path.write_bytes(header.encode("ascii") + np.concatenate(blocks).tobytes())


class ReadersTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = load_bundle()
        sig, fields = wfdb.rdsamp(str(EXAMPLE))
        cls.sig = sig.T.astype(np.float32)  # (12, 1000), мВ, 100 Гц
        cls.names = fields["sig_name"]

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def csv(self, name: str, text: str) -> Path:
        p = self.tmp / name
        p.write_text(text, encoding="utf-8")
        return p

    def table(self, sig: np.ndarray, sep: str = ",", decimal: str = ".", header: bool = True, time: str | None = None,
              fs: int = 100, names: list[str] | None = None) -> str:
        rows = []
        if header:
            rows.append(sep.join(([time] if time else []) + (names or LEAD_NAMES)))
        for i in range(sig.shape[1]):
            cells = [f"{v:.6f}".replace(".", decimal) for v in sig[:, i]]
            if time:
                step = 1000.0 / fs if time.lower().endswith("ms") else 1.0 / fs
                cells = [f"{i * step:.6f}".replace(".", decimal)] + cells
            rows.append(sep.join(cells))
        return "\n".join(rows)

    # --- вспомогательное
    def test_canon_lead(self) -> None:
        for raw, exp in [("aVR", "AVR"), ("ECG Lead II", "II"), ("EKG V1", "V1"), ("aV_L", "AVL"), ("AVF", "AVF"), ("lead i", "I")]:
            self.assertEqual(canon_lead(raw), exp)

    # --- WFDB
    def test_wfdb_unchanged(self) -> None:
        rec = load_record([EXAMPLE.with_suffix(".hea"), EXAMPLE.with_suffix(".dat")], self.bundle)
        np.testing.assert_allclose(rec.signal, self.sig, atol=1e-6)
        self.assertEqual((rec.fmt, rec.fs_original, rec.warnings), ("WFDB", 100, []))

    # --- CSV
    def test_csv_header_names_and_order(self) -> None:
        shuffled = [3, 0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11]
        p = self.csv("a.csv", self.table(self.sig[shuffled], names=[LEAD_NAMES[i] for i in shuffled], time="time"))
        rec = load_record([p], self.bundle)
        np.testing.assert_allclose(rec.signal, self.sig, atol=1e-5)
        self.assertEqual(rec.fs_original, 100)
        self.assertEqual(rec.fmt, "CSV")

    def test_csv_semicolon_decimal_comma_ms_time(self) -> None:
        p = self.csv("b.csv", self.table(self.sig, sep=";", decimal=",", time="time_ms"))
        rec = load_record([p], self.bundle)
        np.testing.assert_allclose(rec.signal, self.sig, atol=1e-5)
        self.assertEqual(rec.fs_original, 100)

    def test_csv_no_header_needs_fs(self) -> None:
        p = self.csv("c.csv", self.table(self.sig, header=False))
        with self.assertRaisesRegex(ValueError, "частоту дискретизации"):
            load_record([p], self.bundle)
        rec = load_record([p], self.bundle, fs=100)
        np.testing.assert_allclose(rec.signal, self.sig, atol=1e-5)
        self.assertTrue(any("порядке" in w for w in rec.warnings))

    def test_csv_units_microvolts(self) -> None:
        p = self.csv("d.csv", self.table(self.sig * 1000, header=True))
        rec = load_record([p], self.bundle, fs=100, unit="uV")
        np.testing.assert_allclose(rec.signal, self.sig, atol=1e-3)

    def test_csv_resample_250hz(self) -> None:
        t = np.arange(2500) / 250
        sig = np.stack([np.sin(2 * np.pi * 1.2 * t + k) for k in range(12)]).astype(np.float32)
        rec = load_record([self.csv("e.csv", self.table(sig, fs=250, time="t"))], self.bundle)
        self.assertEqual(rec.signal.shape, (12, 1000))
        self.assertEqual(rec.fs_original, 250)
        ref = np.sin(2 * np.pi * 1.2 * (np.arange(1000) / 100))
        np.testing.assert_allclose(rec.signal[0, 50:-50], ref[50:-50], atol=0.03)  # края — переходный процесс фильтра

    def test_csv_errors(self) -> None:
        with self.assertRaisesRegex(ValueError, "12 стандартных"):
            load_record([self.csv("f.csv", self.table(self.sig[:8], names=LEAD_NAMES[:8]))], self.bundle, fs=100)
        with self.assertRaisesRegex(ValueError, "короче 10 с"):
            load_record([self.csv("g.csv", self.table(self.sig[:, :500]))], self.bundle, fs=100)
        bad = self.sig.copy()
        bad[2, 10] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN"):
            load_record([self.csv("h.csv", self.table(bad))], self.bundle, fs=100)

    # --- EDF
    def test_edf_mv(self) -> None:
        p = self.tmp / "a.edf"
        write_edf(p, self.sig, 100, [f"ECG {n}" for n in LEAD_NAMES], with_annotations=True)
        rec = load_record([p], self.bundle)
        np.testing.assert_allclose(rec.signal, self.sig, atol=2e-4)  # шаг квантования int16 ~ 1e-4 мВ
        self.assertEqual((rec.fmt, rec.fs_original), ("EDF", 100))

    def test_edf_microvolts_and_500hz(self) -> None:
        t = np.arange(5000) / 500
        sig = np.stack([0.5 * np.sin(2 * np.pi * 1.2 * t + k) for k in range(12)]).astype(np.float32)
        p = self.tmp / "b.edf"
        write_edf(p, sig, 500, LEAD_NAMES, dim="uV", scale=1000.0)
        rec = load_record([p], self.bundle)
        self.assertEqual(rec.signal.shape, (12, 1000))
        ref = 0.5 * np.sin(2 * np.pi * 1.2 * (np.arange(1000) / 100))
        np.testing.assert_allclose(rec.signal[0, 50:-50], ref[50:-50], atol=0.02)

    def test_edf_errors(self) -> None:
        p = self.tmp / "bad.edf"
        p.write_bytes(b"not an edf" * 40)
        with self.assertRaises(ValueError):
            load_record([p], self.bundle)
        with self.assertRaisesRegex(ValueError, "нет отведений"):
            q = self.tmp / "few.edf"
            write_edf(q, self.sig[:3], 100, LEAD_NAMES[:3])
            load_record([q], self.bundle)

    # --- выбор читателя
    def test_mixed_and_unknown_formats(self) -> None:
        with self.assertRaisesRegex(ValueError, "смешивать"):
            load_record([EXAMPLE.with_suffix(".hea"), self.tmp / "x.csv"], self.bundle)
        with self.assertRaisesRegex(ValueError, "не поддерживается"):
            load_record([self.tmp / "x.pdf"], self.bundle)
        self.assertEqual(STD_LEADS, self.bundle.leads)


if __name__ == "__main__":
    unittest.main()
