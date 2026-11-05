"""Обучение 1D-CNN / ResNet1d на PTB-XL (multi-label, 5 суперклассов).

Train — strat_fold 1-8; ранняя остановка и выбор лучшей эпохи — по macro AUROC на val (фолд 9).
Test не загружается вообще. Результаты: results/<дата>_<имя>/ (config, metrics.json, best.pt, val_probs.npy).

Запуск:   python -m src.train --config configs/resnet1d.yaml
Проверка: python -m src.train --config configs/resnet1d.yaml --limit 500 --epochs 1 --output-root <папка>
Батч:     python -m src.train --config configs/resnet1d.yaml --probe-batch
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import random
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn

from src.augment import Augmenter
from src.data import PROJECT_ROOT, _resolve, lead_indices, load_config, load_splits
from src.evaluate import multilabel_metrics
from src.models import build_model, count_parameters, set_output_bias

AMP_DTYPES: dict[str, torch.dtype] = {"float16": torch.float16, "bfloat16": torch.bfloat16}


# ----------------------------------------------------------------------------- утилиты

def set_seed(seed: int) -> None:
    """Фиксирует seed python/numpy/torch и включает детерминированные алгоритмы cuDNN."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _git_state() -> dict[str, Any]:
    """Короткий хэш коммита и признак незакоммиченных изменений (для воспроизводимости)."""
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    try:
        return {"commit": run("rev-parse", "--short", "HEAD"), "dirty": bool(run("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def make_run_dir(root: Path, name: str) -> Path:
    """Создаёт <root>/<YYYY-MM-DD>_<name>; если папка есть, добавляет время, чтобы не перезаписать."""
    run_dir = root / f"{time.strftime('%Y-%m-%d')}_{name}"
    if run_dir.exists():
        run_dir = root / f"{time.strftime('%Y-%m-%d')}_{name}_{time.strftime('%H%M%S')}"
    run_dir.mkdir(parents=True)
    return run_dir


def make_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """AdamW; weight decay не применяется к bias и параметрам BatchNorm (1-мерные тензоры)."""
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim > 1]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim <= 1]
    groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr)


def make_scheduler(
    optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int, min_lr_ratio: float
) -> torch.optim.lr_scheduler.LRScheduler:
    """Линейный разогрев, затем косинусный спад lr до min_lr_ratio * lr. Шаг — на каждый шаг оптимизатора."""
    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


# ----------------------------------------------------------------------------- обучение / оценка

def train_epoch(
    model: nn.Module, X: torch.Tensor, y: torch.Tensor, *, batch_size: int, accum: int,
    optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler, criterion: nn.Module, device: torch.device,
    amp_dtype: torch.dtype | None, grad_clip: float, gen: torch.Generator,
    augment: Augmenter | None = None,
) -> float:
    """Одна эпоха по перемешанному train (с аугментацией, если задана). Возвращает средний train loss."""
    model.train()
    n = len(X)
    perm = torch.randperm(n, generator=gen)
    n_batches = math.ceil(n / batch_size)
    total = torch.zeros((), device=device)  # копим на GPU, чтобы не синхронизироваться на каждом шаге
    optimizer.zero_grad(set_to_none=True)
    for b in range(n_batches):
        idx = perm[b * batch_size:(b + 1) * batch_size]
        xb, yb = X[idx].to(device, non_blocking=True), y[idx].to(device, non_blocking=True)
        if augment is not None:
            xb = augment(xb)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(xb)
        loss = criterion(logits.float(), yb)  # лосс считаем в fp32
        scaler.scale(loss / accum).backward()
        total += loss.detach() * len(idx)
        if (b + 1) % accum == 0 or b == n_batches - 1:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
    return float(total.item() / n)


@torch.no_grad()
def predict(
    model: nn.Module, X: torch.Tensor, y: torch.Tensor, *, batch_size: int, criterion: nn.Module,
    device: torch.device, amp_dtype: torch.dtype | None,
) -> tuple[np.ndarray, float]:
    """Вероятности (sigmoid) и средний лосс на выборке; порядок записей сохраняется."""
    model.eval()
    probs: list[torch.Tensor] = []
    total = 0.0
    for i in range(0, len(X), batch_size):
        xb, yb = X[i:i + batch_size].to(device), y[i:i + batch_size].to(device)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(xb)
        logits = logits.float()
        total += criterion(logits, yb).item() * len(xb)
        probs.append(torch.sigmoid(logits).cpu())
    return torch.cat(probs).numpy(), total / len(X)


# ----------------------------------------------------------------------------- подбор батча

def probe_batch_sizes(cfg: dict[str, Any], n_classes: int, length: int, sizes: list[int]) -> None:
    """Печатает пиковую память GPU на шаге обучения (fwd+bwd+AdamW) для разных размеров батча."""
    device = torch.device("cuda")
    tc = cfg["train"]
    amp_dtype = AMP_DTYPES[tc["amp_dtype"]]
    total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"Модель: {cfg['model']['name']}, вход (12, {length}), amp={tc['amp_dtype']}, VRAM {total_gb:.1f} ГБ")
    for bs in sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            model = build_model(cfg["model"]["name"], 12, n_classes, **cfg["model"]["params"]).to(device)
            opt = make_optimizer(model, tc["lr"], tc["weight_decay"])
            scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype is torch.float16)
            xb = torch.randn(bs, 12, length, device=device)
            yb = (torch.rand(bs, n_classes, device=device) > 0.5).float()
            t0 = time.perf_counter()
            for _ in range(5):
                with torch.autocast("cuda", dtype=amp_dtype):
                    logits = model(xb)
                loss = nn.functional.binary_cross_entropy_with_logits(logits.float(), yb)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / 5
            peak = torch.cuda.max_memory_allocated() / 2**30
            print(f"  batch {bs:>4}: пик {peak:5.2f} ГБ ({100 * peak / total_gb:4.1f}% VRAM), {dt * 1000:6.0f} мс/шаг")
        except torch.cuda.OutOfMemoryError:
            print(f"  batch {bs:>4}: OOM")
        finally:
            model = opt = xb = yb = logits = loss = None  # type: ignore[assignment]


# ----------------------------------------------------------------------------- основной цикл

def fit(cfg: dict[str, Any], data_cfg: dict[str, Any], run_dir: Path, limit: int | None) -> dict[str, Any]:
    """Обучает модель, пишет best.pt / val_probs.npy / metrics.json в run_dir, возвращает metrics."""
    t_start = time.perf_counter()
    tc = cfg["train"]
    seed: int = cfg["seed"]
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype: torch.dtype | None = AMP_DTYPES[tc["amp_dtype"]] if device.type == "cuda" else None

    # Только train и val; нормализация — статистики train (внутри load_splits). Test не грузится.
    splits, meta = load_splits(data_cfg, splits=("train", "val"), normalize=True)
    tr, va = splits["train"], splits["val"]
    if limit:
        tr_sl, va_sl = slice(0, limit), slice(0, limit)
    else:
        tr_sl = va_sl = slice(None)
    # Выбор отведений (после нормализации по train: она по-отведенческая, так что результат тот же)
    lead_idx = lead_indices(meta["leads"], cfg.get("leads"))
    lead_names = [meta["leads"][i] for i in lead_idx]
    print(f"Отведения ({len(lead_idx)}): {lead_names}", flush=True)
    X_tr = torch.from_numpy(np.ascontiguousarray(tr.X[tr_sl][:, lead_idx]))
    X_va = torch.from_numpy(np.ascontiguousarray(va.X[va_sl][:, lead_idx]))
    y_tr, y_va = torch.from_numpy(tr.y[tr_sl]).float(), torch.from_numpy(va.y[va_sl]).float()
    classes: list[str] = meta["classes"]

    model = build_model(cfg["model"]["name"], X_tr.shape[1], len(classes), **cfg["model"]["params"])
    prevalence = y_tr.mean(dim=0)
    set_output_bias(model, prevalence.tolist())
    model.to(device)
    n_params = count_parameters(model)

    pos_weight = None
    if tc["pos_weight"]:  # (N - P) / P по train; смещает вероятности вверх -> ухудшает калибровку
        pos_weight = ((len(y_tr) - y_tr.sum(0)) / y_tr.sum(0).clamp(min=1)).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    bs, accum, epochs = tc["batch_size"], tc["grad_accum_steps"], tc["epochs"]
    steps_per_epoch = math.ceil(math.ceil(len(X_tr) / bs) / accum)
    optimizer = make_optimizer(model, tc["lr"], tc["weight_decay"])
    scheduler = make_scheduler(
        optimizer, epochs * steps_per_epoch, round(tc["warmup_epochs"] * steps_per_epoch), tc["min_lr"] / tc["lr"]
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype is torch.float16)
    gen = torch.Generator().manual_seed(seed)  # порядок батчей воспроизводим и не зависит от других rng
    aug_cfg: dict[str, Any] = cfg.get("augment") or {}  # секции может не быть (старые конфиги)
    augment = (
        Augmenter(aug_cfg, data_cfg["sampling_rate"], device, seed + 1) if aug_cfg.get("enabled") else None
    )

    es = tc["early_stopping"]
    best_auc, best_epoch, bad_epochs = -math.inf, 0, 0
    best_metrics: dict[str, Any] = {}
    history: list[dict[str, float]] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, epochs + 1):
        t_ep = time.perf_counter()
        train_loss = train_epoch(
            model, X_tr, y_tr, batch_size=bs, accum=accum, optimizer=optimizer, scheduler=scheduler,
            scaler=scaler, criterion=criterion, device=device, amp_dtype=amp_dtype,
            grad_clip=tc["grad_clip"], gen=gen, augment=augment,
        )
        probs, val_loss = predict(model, X_va, y_va, batch_size=tc["eval_batch_size"],
                                  criterion=criterion, device=device, amp_dtype=amp_dtype)
        m = multilabel_metrics(va.y[va_sl], probs, classes)
        val_auc = m["macro_auroc"]
        lr_now = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "val_macro_auroc": val_auc, "lr": lr_now, "time_sec": time.perf_counter() - t_ep})

        improved = val_auc > best_auc + es["min_delta"]
        if improved:
            best_auc, best_epoch, bad_epochs, best_metrics = val_auc, epoch, 0, m
            torch.save(model.state_dict(), run_dir / "best.pt")
            np.save(run_dir / "val_probs.npy", probs)  # для подбора порогов/калибровки без переобучения
        else:
            bad_epochs += 1
        stop = bad_epochs >= es["patience"]
        print(f"эпоха {epoch:>3}/{epochs} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | "
              f"val macro AUROC {val_auc:.4f}{' *' if improved else '  '} | lr {lr_now:.2e} | "
              f"{history[-1]['time_sec']:.0f} с{' | ранняя остановка' if stop else ''}", flush=True)
        if stop:
            break

    peak_gb = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None
    metrics = {
        "experiment": cfg["experiment_name"],
        "evaluated_on": "val (strat_fold 9); test не использовался",
        "val": best_metrics,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "history": history,
        "leads": lead_names,
        "model": {"name": cfg["model"]["name"], "params": cfg["model"]["params"], "n_parameters": n_params},
        "train_setup": {
            "batch_size": bs, "grad_accum_steps": accum, "effective_batch": bs * accum,
            "amp_dtype": tc["amp_dtype"] if amp_dtype else None, "pos_weight": tc["pos_weight"],
            "augment": aug_cfg if augment is not None else None,
            "limit_records": limit, "peak_gpu_memory_gb": None if peak_gb is None else round(peak_gb, 3),
        },
        "n_train": int(len(X_tr)),
        "n_val": int(len(X_va)),
        "excluded_no_superclass": {k: v["excluded_no_superclass"] for k, v in meta["stats"]["splits"].items() if k != "test"},
        "classes": classes,
        "train_prevalence": {c: round(float(p), 4) for c, p in zip(classes, prevalence)},
        "note": (
            "Ранняя остановка и выбор эпохи идут по val, поэтому val-метрики слегка оптимистичны; "
            "несмещённая оценка — только на test (фолд 10) в финале."
        ),
        "seed": seed,
        "time_sec": {"total": round(time.perf_counter() - t_start, 1)},
        "env": {
            "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "numpy": np.__version__, "git": _git_state(), "cache_fingerprint": meta["fingerprint"],
        },
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    """CLI: прочитать конфиг, обучить, записать результаты."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, help="переопределить число эпох (для проверок)")
    parser.add_argument("--limit", type=int, help="взять первые N записей train и val (для проверок)")
    parser.add_argument("--output-root", help="куда писать папку запуска (по умолчанию output_root из конфига)")
    parser.add_argument("--probe-batch", action="store_true", help="только замерить память GPU по размерам батча")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    data_cfg = load_config(cfg["data_config"])["data"]
    n_classes, length = len(data_cfg["classes"]), data_cfg["sampling_rate"] * 10
    if args.probe_batch and cfg.get("leads") not in (None, "all"):
        print("Замечание: --probe-batch использует 12 входных каналов независимо от leads (оценка сверху).")

    if args.probe_batch:
        probe_batch_sizes(cfg, n_classes, length, cfg["train"]["probe_sizes"])
        return

    run_dir = make_run_dir(_resolve(args.output_root or cfg["output_root"]), cfg["experiment_name"])
    shutil.copy(_resolve(args.config), run_dir / "config.yaml")
    (run_dir / "data_config.yaml").write_text(yaml.safe_dump({"data": data_cfg}, allow_unicode=True), encoding="utf-8")
    metrics = fit(cfg, data_cfg, run_dir, args.limit)
    print(f"Готово: лучшая эпоха {metrics['best_epoch']}, val macro AUROC {metrics['val']['macro_auroc']:.4f}. "
          f"Результаты: {run_dir}")


if __name__ == "__main__":
    main()
