from __future__ import annotations

import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import TrainConfig, save_config
from .data import MusicTokenDataset
from .model import ModernMusicTransformer, MusicTransformer, count_parameters
from .optim import Muon, split_muon_params


def build_model(config: TrainConfig, device: torch.device) -> nn.Module:
    model_cfg = config.model
    if model_cfg.architecture == "modern":
        model = ModernMusicTransformer(
            vocab_size=config.vocab_size,
            d_model=model_cfg.d_model,
            num_heads=model_cfg.num_heads,
            num_layers=model_cfg.num_layers,
            ffn_dim=model_cfg.ffn_dim,
            dropout=model_cfg.dropout,
            max_seq_len=config.seq_len,
            padding_idx=config.token_pad,
            num_kv_heads=model_cfg.num_kv_heads,
            rope_theta=model_cfg.rope_theta,
        )
    elif model_cfg.architecture == "legacy":
        model = MusicTransformer(
            vocab_size=config.vocab_size,
            d_model=model_cfg.d_model,
            num_heads=model_cfg.num_heads,
            num_layers=model_cfg.num_layers,
            ffn_dim=model_cfg.ffn_dim,
            dropout=model_cfg.dropout,
            max_seq_len=config.seq_len,
            padding_idx=config.token_pad,
        )
    else:
        raise ValueError(f"unknown model architecture: {model_cfg.architecture}")
    return model.to(device)


def _make_optimizer(config: TrainConfig, model: nn.Module, total_steps: int | None = None) -> tuple[torch.optim.Optimizer | list[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler | list[torch.optim.lr_scheduler.LRScheduler] | None]:
    def cosine_scheduler(optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LambdaLR:
        steps = max(1, total_steps or config.epochs)
        warmup = min(config.warmup_steps, max(0, steps - 1))

        def schedule(step: int) -> float:
            if warmup > 0 and step < warmup:
                return max(config.min_lr_ratio, float(step + 1) / float(warmup))
            progress = (step - warmup) / float(max(1, steps - warmup))
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
            return config.min_lr_ratio + (1.0 - config.min_lr_ratio) * cosine

        return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)

    if config.optimizer.lower() == "muon":
        muon_params, adamw_params = split_muon_params(model)
        optimizers: list[torch.optim.Optimizer] = []
        if muon_params:
            optimizers.append(Muon(muon_params, lr=config.muon_learning_rate, momentum=config.muon_momentum, weight_decay=config.weight_decay))
        if adamw_params:
            optimizers.append(torch.optim.AdamW(adamw_params, lr=config.learning_rate or 3e-4, weight_decay=config.weight_decay))
        if config.lr_schedule == "cosine":
            return optimizers, [cosine_scheduler(opt) for opt in optimizers]
        return optimizers, None

    if config.learning_rate is not None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        if config.lr_schedule == "cosine":
            return optimizer, cosine_scheduler(optimizer)
        return optimizer, None

    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9, weight_decay=config.weight_decay)

    def schedule(step: int) -> float:
        step = max(1, step + 1)
        return (config.model.d_model ** -0.5) * min(step ** -0.5, step * (config.warmup_steps ** -1.5))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: TrainConfig,
    optimizer: torch.optim.Optimizer | list[torch.optim.Optimizer] | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | list[torch.optim.lr_scheduler.LRScheduler] | None = None,
    scaler: GradScaler | None = None,
) -> float:
    training = optimizer is not None
    optimizers = optimizer if isinstance(optimizer, list) else ([optimizer] if optimizer is not None else [])
    model.train(training)
    total = 0.0
    amp_enabled = config.mixed_precision and device.type == "cuda"
    iterator = tqdm(loader, desc="train" if training else "valid", leave=False)
    if training:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    for step, (src, tgt) in enumerate(iterator, 1):
        src = src.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                logits = model(src)
                loss = criterion(logits.reshape(-1, config.vocab_size), tgt.reshape(-1))
                if training:
                    loss = loss / config.grad_accum_steps

            if training:
                if scaler is not None and amp_enabled:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if step % config.grad_accum_steps == 0:
                    if scaler is not None and amp_enabled:
                        for opt in optimizers:
                            scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if scaler is not None and amp_enabled:
                        for opt in optimizers:
                            scaler.step(opt)
                        scaler.update()
                    else:
                        for opt in optimizers:
                            opt.step()
                    for opt in optimizers:
                        opt.zero_grad(set_to_none=True)
                    if scheduler is not None:
                        schedulers = scheduler if isinstance(scheduler, list) else [scheduler]
                        for sched in schedulers:
                            sched.step()

        display_loss = loss.item() * (config.grad_accum_steps if training else 1)
        total += display_loss
        iterator.set_postfix(loss=f"{display_loss:.4f}")

    return total / max(1, len(loader))


def train(config: TrainConfig) -> Path:
    config.ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = MusicTokenDataset(config.processed_dir / "train", config.seq_len, config.token_pad)
    val_ds = MusicTokenDataset(config.processed_dir / "validation", config.seq_len, config.token_pad)
    if not train_ds.data:
        raise RuntimeError(f"No preprocessed training files found in {config.processed_dir / 'train'}")

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    ) if val_ds.data else None

    model = build_model(config, device)
    if config.resume_from is not None and Path(config.resume_from).exists():
        payload = torch.load(config.resume_from, map_location=device)
        state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        model.load_state_dict({k.removeprefix("module."): v for k, v in state_dict.items()})
        print(f"resumed model weights from {config.resume_from}")
    steps_per_epoch = max(1, len(train_loader) // max(1, config.grad_accum_steps))
    optimizer, scheduler = _make_optimizer(config, model, total_steps=steps_per_epoch * config.epochs)
    scaler = GradScaler(enabled=config.mixed_precision and device.type == "cuda")
    criterion = nn.CrossEntropyLoss(ignore_index=config.token_pad, label_smoothing=config.label_smoothing)

    print(f"device={device} params={count_parameters(model):,}")
    best_loss = float("inf")
    best_path = config.models_dir / "best_model.pt"
    save_config(config, config.models_dir / "config.json")

    epochs_without_improvement = 0
    for epoch in range(config.epochs):
        start = time.time()
        train_loss = _run_epoch(model, train_loader, criterion, device, config, optimizer, scheduler, scaler)
        val_loss = _run_epoch(model, val_loader, criterion, device, config) if val_loader is not None else float("nan")
        ppl = math.exp(min(val_loss, 20)) if not math.isnan(val_loss) else float("nan")
        first_optimizer = optimizer[0] if isinstance(optimizer, list) else optimizer
        lr = first_optimizer.param_groups[0]["lr"]
        print(
            f"epoch={epoch + 1}/{config.epochs} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_ppl={ppl:.2f} lr={lr:.2e} time={time.time() - start:.1f}s"
        )
        improved = not math.isnan(val_loss) and val_loss < best_loss
        if improved:
            best_loss = val_loss
            epochs_without_improvement = 0
            torch.save({"model": model.state_dict(), "config": str(config.models_dir / "config.json")}, best_path)
            print(f"saved {best_path}")
        elif not math.isnan(val_loss):
            epochs_without_improvement += 1
            if config.early_stopping_patience is not None and epochs_without_improvement >= config.early_stopping_patience:
                print(f"early stopping after {epochs_without_improvement} epochs without validation improvement")
                break

    if not best_path.exists():
        torch.save({"model": model.state_dict(), "config": str(config.models_dir / "config.json")}, best_path)
    return best_path
