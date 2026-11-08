"""Тесты app/quality.py на реальной записи PTB-XL (пример приложения) и искусственных неисправностях.

Запуск: python -m unittest discover -s tests -v. Пороги подобраны на фолде 9 (scripts/check_quality.py); эти тесты
проверяют только, что очевидные неисправности ловятся, а чистая запись не помечается.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import wfdb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from quality import assess_quality  # noqa: E402

LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def codes(rep) -> set[str]:
    return {i.code for i in rep.issues}


class QualityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sig, _ = wfdb.rdsamp(str(ROOT / "app" / "examples" / "01421_lr"))
        cls.x = sig.T.astype(np.float64)

    def test_clean_record_ok(self) -> None:
        rep = assess_quality(self.x, LEADS, 100)
        self.assertEqual(rep.status, "ok", [i.message for i in rep.issues])
        self.assertTrue(40 < rep.metrics["heart_rate_bpm"] < 140)

    def test_flat_lead_is_poor(self) -> None:
        x = self.x.copy()
        x[8] = np.median(x[8])
        rep = assess_quality(x, LEADS, 100)
        self.assertEqual(rep.status, "poor")
        self.assertIn("flat_lead", codes(rep))
        self.assertEqual([i.leads for i in rep.issues if i.code == "flat_lead"], [["V3"]])

    def test_dropout_is_poor(self) -> None:
        x = self.x.copy()
        x[:, 500:] = x[:, 499:500]
        self.assertEqual(assess_quality(x, LEADS, 100).status, "poor")

    def test_wrong_units_is_poor(self) -> None:
        self.assertEqual(assess_quality(self.x * 1000, LEADS, 100).status, "poor")  # мкВ, прочитанные как мВ
        self.assertEqual(assess_quality(self.x * 0.001, LEADS, 100).status, "poor")  # В, прочитанные как мВ

    def test_pure_noise_is_poor(self) -> None:
        noise = np.random.default_rng(0).normal(0.0, 0.3, self.x.shape)
        rep = assess_quality(noise, LEADS, 100)
        self.assertEqual(rep.status, "poor")
        self.assertIn("no_qrs", codes(rep))

    def test_swapped_leads_flagged(self) -> None:
        x = self.x.copy()
        x[[1, 2]] = x[[2, 1]]
        rep = assess_quality(x, LEADS, 100)
        self.assertIn("einthoven", codes(rep))
        self.assertNotEqual(rep.status, "ok")

    def test_heavy_drift_flagged(self) -> None:
        t = np.arange(self.x.shape[1]) / 100
        x = self.x + 4.0 * np.sin(2 * np.pi * 0.3 * t)[None, :]
        self.assertIn("baseline_wander", codes(assess_quality(x, LEADS, 100)))

    def test_duplicate_leads_flagged(self) -> None:
        x = self.x.copy()
        x[6] = x[1]
        self.assertIn("duplicate_leads", codes(assess_quality(x, LEADS, 100)))


if __name__ == "__main__":
    unittest.main()
