import argparse
import math
import os

import torch
from torch.amp import GradScaler, autocast

from dataset import get_sft_batch, load_sft_pairs, split_sft_pairs
from model import GPT, GPTConfig
from tokenizer import CharTokenizer
from train import cosine_lr, save_checkpoint
from utils import load_config, resolve_device, setup_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--pretrain_checkpoint",
        type=str,
        default=None,
        help="Pretrain checkpoint (default: train.checkpoint_dir/best.pt)",
    )
    return parser.parse_args()


@torch.no_grad()
def estimate_sft_loss(model, pairs, cfg, device):
    model.eval()
    losses = torch.zeros(cfg["train"]["eval_steps"], device=device)
    for k in range(cfg["train"]["eval_steps"]):
        batch = get_sft_batch(
            pairs,
            cfg["train"]["batch_size"],
            cfg["model"]["block_size"],
            device,
        )
        _, loss = model(batch["x"], batch["y"], loss_mask=batch["loss_mask"])
        losses[k] = loss
    model.train()
    return losses.mean().item()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    setup_seed(cfg["seed"])
    device = resolve_device(cfg["device"])

    tok = CharTokenizer.load(cfg["data"]["vocab_path"])
    all_pairs = load_sft_pairs(cfg["data"]["sft_path"], tok)
    train_pairs, val_pairs = split_sft_pairs(
        all_pairs,
        val_ratio=float(cfg["data"].get("sft_val_ratio", 0.05)),
        seed=int(cfg["seed"]),
    )
    if not train_pairs:
        raise ValueError("No SFT training pairs found. Run prepare_data first.")

    pretrain_path = args.pretrain_checkpoint or os.path.join(
        cfg["train"]["pretrain_checkpoint_dir"],
        "best.pt",
    )
    if not os.path.exists(pretrain_path):
        pretrain_path = os.path.join(cfg["train"]["pretrain_checkpoint_dir"], "final.pt")
    ckpt = torch.load(pretrain_path, map_location=device)
    model_cfg = GPTConfig(**ckpt["model_cfg"])
    model = GPT(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"loaded pretrain weights: {pretrain_path}")

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
    best_val = float("inf")
    patience = int(cfg["train"].get("early_stop_patience", 0))
    stale_evals = 0

    for step in range(cfg["train"]["max_steps"]):
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
            batch = get_sft_batch(
                train_pairs,
                cfg["train"]["batch_size"],
                cfg["model"]["block_size"],
                device,
            )
            with autocast("cuda", enabled=use_amp):
                _, loss = model(batch["x"], batch["y"], loss_mask=batch["loss_mask"])
                loss = loss / grad_accum_steps
            running_loss += loss.item()
            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

        if step % cfg["train"]["log_interval"] == 0:
            print(f"[sft] step={step:5d} loss={running_loss:.4f} lr={lr:.6f}")

        if step % cfg["train"]["eval_interval"] == 0:
            train_loss = estimate_sft_loss(model, train_pairs, cfg, device)
            val_loss = estimate_sft_loss(model, val_pairs, cfg, device)
            print(
                f"[sft-eval] step={step:5d} train={train_loss:.4f} val={val_loss:.4f}"
            )
            if val_loss < best_val:
                best_val = val_loss
                stale_evals = 0
                best_path = os.path.join(ckpt_dir, "best.pt")
                save_checkpoint(best_path, model, model_cfg, cfg, step, best_val)
                print(f"saved sft best: {best_path} (val={best_val:.4f})")
            else:
                stale_evals += 1
                if patience > 0 and stale_evals >= patience:
                    print(f"sft early stop at step={step}, best_val={best_val:.4f}")
                    break

    final_path = os.path.join(ckpt_dir, "final.pt")
    save_checkpoint(final_path, model, model_cfg, cfg, step, best_val)
    print(f"sft complete: {final_path}, best_val={best_val:.4f}")


if __name__ == "__main__":
    main()
