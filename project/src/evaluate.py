import argparse
import math
from pathlib import Path

import torch

from dataset import build_data_bundle, get_batch
from model import GPT, GPTConfig
from tokenizer import CharTokenizer
from utils import load_config, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="*",
        default=["中国革命", "实事求是", "为人民服务"],
    )
    return parser.parse_args()


@torch.no_grad()
def perplexity(model, data_ids, batch_size, block_size, device, eval_steps):
    losses = []
    for _ in range(eval_steps):
        batch = get_batch(data_ids, batch_size, block_size, device)
        _, loss = model(batch["x"], batch["y"])
        losses.append(loss.item())
    mean_loss = sum(losses) / len(losses)
    return math.exp(mean_loss), mean_loss


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg["device"])

    split_path = {
        "train": cfg["data"]["train_path"],
        "val": cfg["data"]["val_path"],
        "test": cfg["data"].get("test_path", "project/data/processed/test.txt"),
    }[args.split]

    tok = CharTokenizer.load(cfg["data"]["vocab_path"])
    data_ids = torch.tensor(
        tok.encode(Path(split_path).read_text(encoding="utf-8")),
        dtype=torch.long,
    )

    ckpt = torch.load(args.checkpoint, map_location=device)
    model_cfg = GPTConfig(**ckpt["model_cfg"])
    model = GPT(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ppl, loss = perplexity(
        model,
        data_ids,
        cfg["train"]["batch_size"],
        cfg["model"]["block_size"],
        device,
        args.eval_steps,
    )
    print(f"split={args.split} loss={loss:.4f} perplexity={ppl:.2f}")

    gen_cfg = cfg["generate"]
    for prompt in args.prompts:
        x = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
        out = model.generate(
            x,
            max_new_tokens=gen_cfg["max_new_tokens"],
            temperature=gen_cfg["temperature"],
            top_k=gen_cfg.get("top_k"),
            top_p=gen_cfg.get("top_p"),
            repetition_penalty=gen_cfg.get("repetition_penalty", 1.0),
        )
        text = tok.decode(out[0].tolist())
        print("-" * 40)
        print(f"prompt: {prompt}")
        print(text)


if __name__ == "__main__":
    main()
