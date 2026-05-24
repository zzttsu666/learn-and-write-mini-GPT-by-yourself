"""
One-click pipeline for learning runs:
  enrich (optional) -> prepare -> build QA -> pretrain -> [SFT if val ok] -> evaluate
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from dataset import build_data_bundle
from model import GPT, GPTConfig
from train import estimate_loss
from utils import load_config, resolve_device


def run(cmd: list[str], cwd: Path) -> None:
    print("\n>>", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain-config", default="project/configs/medium.yaml")
    parser.add_argument("--sft-config", default="project/configs/sft.yaml")
    parser.add_argument("--max-pretrain-val", type=float, default=3.2)
    parser.add_argument("--skip-enrich", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-qa", action="store_true")
    parser.add_argument("--skip-pretrain", action="store_true")
    parser.add_argument("--skip-sft", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    py = sys.executable
    src = root / "project" / "src"

    if not args.skip_enrich:
        run([py, str(src / "enrich_data.py")], root)

    if not args.skip_prepare:
        run([py, str(src / "prepare_data.py"), "--config", args.pretrain_config], root)

    if not args.skip_qa:
        run([py, str(src / "build_qa_data.py"), "--config", args.pretrain_config], root)

    if not args.skip_pretrain:
        run([py, str(src / "train.py"), "--config", args.pretrain_config], root)

    cfg = load_config(str(root / args.pretrain_config))
    ckpt_path = root / cfg["train"]["checkpoint_dir"] / "best.pt"
    run_sft = not args.skip_sft

    if not args.skip_pretrain and ckpt_path.exists():
        device = resolve_device(cfg["device"])
        bundle = build_data_bundle(
            str(root / cfg["data"]["train_path"]),
            str(root / cfg["data"]["val_path"]),
            str(root / cfg["data"]["vocab_path"]),
        )
        ckpt = torch.load(ckpt_path, map_location=device)
        model = GPT(GPTConfig(**ckpt["model_cfg"])).to(device)
        model.load_state_dict(ckpt["model_state"])
        metrics = estimate_loss(model, bundle.train_ids, bundle.val_ids, cfg, device)
        val_loss = metrics["val"]
        run_sft = run_sft and val_loss <= args.max_pretrain_val
        report_path = root / cfg["train"]["checkpoint_dir"] / "metrics.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(
                {
                    "pretrain_val_loss": val_loss,
                    "pretrain_train_loss": metrics["train"],
                    "threshold": args.max_pretrain_val,
                    "run_sft": run_sft,
                    "checkpoint": str(ckpt_path),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[gate] pretrain val={val_loss:.4f}, run_sft={run_sft}")

    if run_sft:
        run([py, str(src / "train_sft.py"), "--config", args.sft_config], root)

    if not args.skip_eval:
        sft_best = root / "project" / "checkpoints" / "sft" / "best.pt"
        pretrain_best = root / cfg["train"]["checkpoint_dir"] / "best.pt"
        ckpt = sft_best if sft_best.exists() else pretrain_best
        eval_cfg = args.sft_config if sft_best.exists() else args.pretrain_config
        if ckpt.exists():
            run(
                [
                    py,
                    str(src / "evaluate.py"),
                    "--config",
                    eval_cfg,
                    "--checkpoint",
                    str(ckpt),
                    "--split",
                    "test",
                ],
                root,
            )


if __name__ == "__main__":
    main()
