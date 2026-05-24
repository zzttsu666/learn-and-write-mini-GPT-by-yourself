import argparse
import math
import os
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast

from dataset import build_data_bundle, get_batch
from model import GPT, GPTConfig
from utils import load_config, resolve_device, setup_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    return parser.parse_args()


@torch.no_grad()
def estimate_loss(model, train_ids, val_ids, cfg, device):
    model.eval()
    out = {}
    for split, data_ids in [("train", train_ids), ("val", val_ids)]:
        losses = torch.zeros(cfg["train"]["eval_steps"], device=device)
        for k in range(cfg["train"]["eval_steps"]):
            batch = get_batch(
                data_ids,
                cfg["train"]["batch_size"],
                cfg["model"]["block_size"],
                device,
            )
            _, loss = model(batch["x"], batch["y"])
            losses[k] = loss
        out[split] = losses.mean().item()
    model.train()
    return out


def cosine_lr(step: int, base_lr: float, warmup_steps: int, max_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, (max_steps - warmup_steps))
    progress = min(1.0, max(0.0, progress))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def save_checkpoint(path: str, model, model_cfg, cfg, step: int, best_val: float) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": cfg,
            "model_cfg": model_cfg.__dict__,
            "step": step,
            "best_val": best_val,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    setup_seed(cfg["seed"])
    device = resolve_device(cfg["device"])

    bundle = build_data_bundle(
        cfg["data"]["train_path"],
        cfg["data"]["val_path"],
        cfg["data"]["vocab_path"],
    )

    model_cfg = GPTConfig(
        vocab_size=bundle.tokenizer.vocab_size,
        block_size=cfg["model"]["block_size"],
        n_layer=cfg["model"]["n_layer"],
        n_head=cfg["model"]["n_head"],
        n_embd=cfg["model"]["n_embd"],
        dropout=cfg["model"]["dropout"],
    )
    model = GPT(model_cfg).to(device)

    start_step = 0
    best_val = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        start_step = int(ckpt.get("step", 0))
        best_val = float(ckpt.get("best_val", float("inf")))
        print(f"resumed from {args.resume} at step={start_step}, best_val={best_val:.4f}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"]["weight_decay"],
        betas=(0.9, 0.95),
    )
    grad_accum_steps = int(cfg["train"].get("grad_accum_steps", 1))
    use_amp = bool(cfg["train"].get("use_amp", True)) and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)

    ckpt_dir = cfg["train"]["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    patience = int(cfg["train"].get("early_stop_patience", 0))
    stale_evals = 0

    for step in range(start_step, cfg["train"]["max_steps"]):
        lr = cosine_lr(
            step,
            cfg["train"]["learning_rate"],
            cfg["train"]["warmup_steps"],
            cfg["train"]["max_steps"],
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for _ in range(grad_accum_steps):
            batch = get_batch(
                bundle.train_ids,
                cfg["train"]["batch_size"],
                cfg["model"]["block_size"],
                device,
            )
            with autocast("cuda", enabled=use_amp):
                _, loss = model(batch["x"], batch["y"])
                loss = loss / grad_accum_steps
            running_loss += loss.item()
            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

        if step % cfg["train"]["log_interval"] == 0:
            print(
                f"step={step:5d} train_loss={running_loss:.4f} "
                f"lr={lr:.6f} accum={grad_accum_steps} amp={use_amp}"
            )

        if step % cfg["train"]["eval_interval"] == 0:
            metrics = estimate_loss(model, bundle.train_ids, bundle.val_ids, cfg, device)
            print(
                f"[eval] step={step:5d} train={metrics['train']:.4f} val={metrics['val']:.4f}"
            )
            if metrics["val"] < best_val:
                best_val = metrics["val"]
                stale_evals = 0
                best_path = os.path.join(ckpt_dir, "best.pt")
                save_checkpoint(best_path, model, model_cfg, cfg, step, best_val)
                print(f"saved best checkpoint: {best_path} (val={best_val:.4f})")
            else:
                stale_evals += 1
                if patience > 0 and stale_evals >= patience:
                    print(f"early stop at step={step}, best_val={best_val:.4f}")
                    break

        if step > 0 and step % cfg["train"]["checkpoint_every"] == 0:
            ckpt_path = os.path.join(ckpt_dir, f"ckpt_step_{step}.pt")
            save_checkpoint(ckpt_path, model, model_cfg, cfg, step, best_val)
            print(f"saved checkpoint: {ckpt_path}")

    final_path = os.path.join(ckpt_dir, "final.pt")
    save_checkpoint(final_path, model, model_cfg, cfg, step, best_val)
    print(f"training complete, final checkpoint: {final_path}")
    print(f"best val loss: {best_val:.4f} -> {os.path.join(ckpt_dir, 'best.pt')}")


if __name__ == "__main__":
    main()
