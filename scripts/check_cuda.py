"""Проверка CUDA/PyTorch: версии, GPU, VRAM, mixed precision и маленький тест."""
from __future__ import annotations

import sys

import torch


def main() -> int:
    """Печатает диагностику и возвращает код выхода (0 — GPU работает)."""
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__} (собран с CUDA: {torch.version.cuda})")
    if not torch.cuda.is_available():
        print("CUDA НЕДОСТУПНА. Вероятно, установлена CPU-версия torch — см. README.md.")
        return 1
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}, VRAM: {props.total_memory / 2**30:.1f} ГБ, "
          f"compute capability: {props.major}.{props.minor}")
    x = torch.randn(1024, 1024, device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        y = x @ x
    torch.cuda.synchronize()
    print(f"Тест matmul (autocast fp16): OK, dtype={y.dtype}")
    print(f"bf16 поддерживается: {torch.cuda.is_bf16_supported()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
