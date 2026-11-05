"""Проверка окружения: работаем в проектном .venv, всё из requirements.txt установлено нужных версий, torch с CUDA.

Запуск (из активированного .venv):  python scripts/check_env.py
Код выхода 0 — всё в порядке, 1 — есть проблемы (они перечислены). Глубокая проверка GPU — scripts/check_cuda.py.
"""
from __future__ import annotations

import sys
from importlib import metadata
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV = PROJECT_ROOT / ".venv"


def read_requirements(path: Path) -> list[Requirement]:
    """Читает requirements.txt, пропуская пустые строки и комментарии (в т.ч. закомментированный torch)."""
    reqs: list[Requirement] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        try:
            reqs.append(Requirement(line))
        except InvalidRequirement:
            print(f"  ? не разобрал строку requirements.txt: {raw!r}")
    return reqs


def check_interpreter() -> list[str]:
    """Интерпретатор должен лежать в <проект>/.venv (а не в системном Python)."""
    exe = Path(sys.executable).resolve()
    if VENV.resolve() not in exe.parents:
        return [f"Python вне проектного .venv: {exe}. Активируйте .venv\\Scripts\\Activate.ps1 (см. README.md)."]
    return []


def check_requirements() -> list[str]:
    """Каждый пакет из requirements.txt установлен и подходит под ограничение версии."""
    problems: list[str] = []
    for req in read_requirements(PROJECT_ROOT / "requirements.txt"):
        try:
            ver = metadata.version(req.name)
        except metadata.PackageNotFoundError:
            problems.append(f"{req.name}: не установлен (pip install -r requirements.txt)")
            continue
        if req.specifier and not req.specifier.contains(ver, prereleases=True):
            problems.append(f"{req.name}: установлена {ver}, требуется {req.specifier}")
    return problems


def check_torch() -> list[str]:
    """torch установлен и собран с CUDA, GPU виден (torch ставится отдельно — см. README.md)."""
    try:
        import torch
    except ImportError:
        return ["torch не установлен: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126"]
    print(f"  torch {torch.__version__} (CUDA сборки: {torch.version.cuda})")
    if not torch.cuda.is_available():
        return ["CUDA недоступна: вероятно, стоит CPU-сборка torch (см. README.md)"]
    return []


def main() -> int:
    """Печатает результат каждой проверки; возвращает 0 при отсутствии проблем."""
    print(f"Python {sys.version.split()[0]}: {sys.executable}")
    problems: list[str] = []
    for name, fn in (("интерпретатор в .venv", check_interpreter), ("requirements.txt", check_requirements),
                     ("torch + CUDA", check_torch)):
        found = fn()
        print(f"[{'OK' if not found else 'ОШИБКА'}] {name}")
        problems += found
    for p in problems:
        print(f"  - {p}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
