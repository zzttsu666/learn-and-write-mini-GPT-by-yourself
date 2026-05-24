import argparse
from pathlib import Path

import torch

from data_utils import format_qa_prompt
from model import GPT, GPTConfig
from tokenizer import CharTokenizer
from utils import load_config, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="中国革命")
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--no_repeat_ngram", type=int, default=None)
    parser.add_argument("--complete_sentence", action="store_true")
    parser.add_argument(
        "--raw-prompt",
        action="store_true",
        help="Do not wrap prompt in 问/答 template (pretrain-style only)",
    )
    return parser.parse_args()


def build_prompt(user_text: str, cfg: dict, raw: bool) -> str:
    gen_cfg = cfg.get("generate", {})
    mode = gen_cfg.get("mode", "auto")
    if raw or mode == "pretrain":
        return user_text
    if mode == "qa" or (mode == "auto" and Path(cfg["data"].get("qa_path", "")).exists()):
        instruction = gen_cfg.get(
            "qa_instruction",
            "写出与主题相关的经典论述原句，只输出一句话。",
        )
        return format_qa_prompt(instruction, f"主题：{user_text}")
    return user_text


def extract_answer(full_text: str, prompt_text: str) -> str:
    if "答：" in full_text:
        return full_text.split("答：", 1)[-1].strip()
    if full_text.startswith(prompt_text):
        return full_text[len(prompt_text) :].strip()
    return full_text.strip()


def first_sentence(text: str) -> str:
    text = text.strip()
    for i, ch in enumerate(text):
        if ch in "。！？":
            return text[: i + 1]
    return text


def is_qa_mode(cfg: dict, raw: bool) -> bool:
    if raw:
        return False
    gen_cfg = cfg.get("generate", {})
    mode = gen_cfg.get("mode", "auto")
    return mode == "qa" or (mode == "auto" and Path(cfg["data"].get("qa_path", "")).exists())


def merge_theme_answer(user_prompt: str, answer: str) -> str:
    """If model only outputs the tail clause, prepend the theme for readability."""
    answer = answer.strip()
    user_prompt = user_prompt.strip()
    if not user_prompt or user_prompt in answer:
        return answer
    if answer.startswith(("是", "则", "要", "应", "必须", "才能")):
        return user_prompt + "，" + answer
    return answer


def resolve_checkpoint_path(args: argparse.Namespace, cfg: dict) -> str:
    if args.checkpoint:
        return args.checkpoint
    candidates = [
        "project/checkpoints/sft/best.pt",
        "project/checkpoints/pretrain/best.pt",
        "project/checkpoints/sft/final.pt",
        "project/checkpoints/pretrain/final.pt",
        "project/checkpoints/final.pt",
    ]
    for path in candidates:
        if Path(path).exists():
            return path
    return args.checkpoint or candidates[0]


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg["device"])

    tok = CharTokenizer.load(cfg["data"]["vocab_path"])
    ckpt_path = resolve_checkpoint_path(args, cfg)
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. Train first or pass --checkpoint."
        )
    print(f"checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model_cfg = GPTConfig(**ckpt["model_cfg"])
    model = GPT(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    user_prompt = args.prompt
    if not user_prompt:
        user_prompt = Path(cfg["data"]["train_path"]).read_text(encoding="utf-8")[:16]

    prompt = build_prompt(user_prompt, cfg, raw=args.raw_prompt)
    if prompt != user_prompt:
        print(f"qa_prompt: {prompt!r}")

    x = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    max_new = (
        args.max_new_tokens
        if args.max_new_tokens is not None
        else cfg["generate"]["max_new_tokens"]
    )
    temp = args.temperature if args.temperature is not None else cfg["generate"]["temperature"]
    top_k = args.top_k if args.top_k is not None else cfg["generate"]["top_k"]
    top_p = args.top_p if args.top_p is not None else cfg["generate"].get("top_p")
    repetition_penalty = (
        args.repetition_penalty
        if args.repetition_penalty is not None
        else cfg["generate"].get("repetition_penalty", 1.0)
    )
    no_repeat_ngram = (
        args.no_repeat_ngram
        if args.no_repeat_ngram is not None
        else cfg["generate"].get("no_repeat_ngram", 0)
    )

    gen_kw = dict(
        max_new_tokens=max_new,
        temperature=temp,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram=no_repeat_ngram,
    )

    gen_cfg = cfg.get("generate", {})
    qa = is_qa_mode(cfg, args.raw_prompt)

    out = model.generate(x, **gen_kw)
    text = extract_answer(tok.decode(out[0].tolist()), prompt)
    if qa and gen_cfg.get("first_sentence_only", True):
        text = first_sentence(text)
        text = merge_theme_answer(user_prompt, text)
    if args.complete_sentence:
        ending_marks = "。！？!?；;"
        extra_budget = 80
        while extra_budget > 0 and (not text or text[-1] not in ending_marks):
            out = model.generate(out, max_new_tokens=1, **gen_kw)
            text = extract_answer(tok.decode(out[0].tolist()), prompt)
            extra_budget -= 1
        if text and text[-1] not in ending_marks:
            text = text.rstrip("，、；：") + "。"
    print(text)


if __name__ == "__main__":
    main()
