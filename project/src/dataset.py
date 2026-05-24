import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from data_utils import format_qa_prompt
from tokenizer import CharTokenizer


@dataclass
class DataBundle:
    tokenizer: CharTokenizer
    train_ids: torch.Tensor
    val_ids: torch.Tensor


@dataclass
class SFTPair:
    prompt_ids: List[int]
    target_ids: List[int]


def load_text_ids(path: str, tokenizer: CharTokenizer) -> torch.Tensor:
    text = Path(path).read_text(encoding="utf-8")
    return torch.tensor(tokenizer.encode(text), dtype=torch.long)


def build_data_bundle(
    train_path: str,
    val_path: str,
    vocab_path: str,
) -> DataBundle:
    tokenizer = CharTokenizer.load(vocab_path)
    train_ids = load_text_ids(train_path, tokenizer)
    val_ids = load_text_ids(val_path, tokenizer)
    return DataBundle(tokenizer=tokenizer, train_ids=train_ids, val_ids=val_ids)


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    if len(data) < 3:
        raise ValueError("Data too short. Need at least 3 tokens.")
    seq_len = min(block_size, len(data) - 1)
    ix = torch.randint(0, len(data) - seq_len, (batch_size,))
    x = torch.stack([data[i : i + seq_len] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + seq_len] for i in ix])
    return {"x": x.to(device), "y": y.to(device)}


def row_to_prompt_completion(row: dict) -> tuple[str, str]:
    if "instruction" in row and "output" in row:
        prompt = format_qa_prompt(row["instruction"], row.get("input", ""))
        completion = row["output"]
    else:
        prompt = row["prompt"]
        completion = row["completion"]
    return prompt, completion


def load_sft_pairs(path: str, tokenizer: CharTokenizer) -> List[SFTPair]:
    pairs: List[SFTPair] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            prompt, completion = row_to_prompt_completion(row)
            prompt_ids = tokenizer.encode(prompt)
            target_ids = tokenizer.encode(completion)
            if len(prompt_ids) < 2 or len(target_ids) < 2:
                continue
            pairs.append(SFTPair(prompt_ids=prompt_ids, target_ids=target_ids))
    return pairs


def split_sft_pairs(
    pairs: Sequence[SFTPair],
    val_ratio: float,
    seed: int,
) -> tuple[List[SFTPair], List[SFTPair]]:
    rng = random.Random(seed)
    indices = list(range(len(pairs)))
    rng.shuffle(indices)
    val_count = max(1, int(len(pairs) * val_ratio)) if pairs else 0
    val_set = {indices[i] for i in range(val_count)}
    train_pairs = [pairs[i] for i in indices if i not in val_set]
    val_pairs = [pairs[i] for i in indices if i in val_set]
    return train_pairs, val_pairs


def _pad_1d(values: List[int], length: int, pad_id: int = 0) -> torch.Tensor:
    if len(values) >= length:
        return torch.tensor(values[:length], dtype=torch.long)
    padded = values + [pad_id] * (length - len(values))
    return torch.tensor(padded, dtype=torch.long)


def get_sft_batch(
    pairs: Sequence[SFTPair],
    batch_size: int,
    block_size: int,
    device: torch.device,
    pad_id: int = 0,
) -> Dict[str, torch.Tensor]:
    if not pairs:
        raise ValueError("No SFT pairs available.")
    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []

    for _ in range(batch_size):
        pair = pairs[random.randrange(len(pairs))]
        full_ids = pair.prompt_ids + pair.target_ids
        if len(full_ids) > block_size + 1:
            start = random.randint(0, len(full_ids) - block_size - 1)
            chunk = full_ids[start : start + block_size + 1]
            prompt_len = max(0, len(pair.prompt_ids) - start)
        else:
            chunk = full_ids
            prompt_len = len(pair.prompt_ids)

        seq = chunk[: block_size + 1]
        x = _pad_1d(seq[:-1], block_size, pad_id=pad_id)
        y = _pad_1d(seq[1:], block_size, pad_id=pad_id)
        mask = torch.zeros(block_size, dtype=torch.float)
        valid = min(block_size, len(seq) - 1)
        mask[:valid] = 1.0
        supervised = max(0, min(prompt_len - 1, valid))
        if supervised > 0:
            mask[:supervised] = 0.0
        xs.append(x)
        ys.append(y)
        masks.append(mask)

    return {
        "x": torch.stack(xs).to(device),
        "y": torch.stack(ys).to(device),
        "loss_mask": torch.stack(masks).to(device),
    }
