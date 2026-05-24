"""Shared text processing helpers for prepare_data and SFT export."""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def strip_control_chars(text: str) -> str:
    text = text.replace("\x00", "")
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or ch >= " ")
    return text


def clean_text(text: str) -> str:
    text = strip_control_chars(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"([。！？；，、,.!?;:：])\1{1,}", r"\1", text)

    cleaned_lines: List[str] = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if re.match(r"^目\s*录$", line):
            continue
        if re.match(r"^第[一二三四五六七八九十百千0-9]+[章节回部分篇]$", line):
            continue
        if re.match(r"^.{1,35}[.．·•…]{3,}\s*\d+\s*$", line):
            continue
        if re.match(r"^(第\s*)?\d+\s*页$", line):
            continue
        if re.match(r"^[-—_]*\s*\d+\s*[-—_]*$", line):
            continue
        if re.match(r"^《.+》\s*[（(].*[)）]\s*[,，]?\s*《.+》第.+页$", line):
            continue
        if re.match(r"^[0-9]{1,4}$", line):
            continue
        if line in {"毛主席语录", "《毛主席语录》"}:
            continue
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def deep_clean_line(line: str) -> str:
    """Rule-based line cleanup (citations, OCR noise, broken punctuation)."""
    line = strip_control_chars(line.strip())
    line = re.sub(r"[ \t]+", " ", line)
    line = re.sub(r"([。！？；，、])\1{1,}", r"\1", line)
    line = re.sub(r"([一二三四五六七八九十百千0-9]+)[—\-_]{2,}([一二三四五六七八九十百千0-9]+)", r"\1", line)
    line = re.sub(r"[（(][^）)]*《[^》]+》[^）)]*[)）]\s*$", "", line)
    line = re.sub(r"《[^》]{1,40}》[^。！？]*$", "", line)
    line = re.sub(r"[（(]\s*[一二三四五六七八九十百千0-9]+[^）)]*[)）]\s*$", "", line)
    line = re.sub(r"第\s*[0-9一二三四五六七八九十百千]+\s*页\s*$", "", line)
    line = re.sub(r"\s+", "", line)
    return line.strip()


def is_garbage_line(line: str, min_len: int = 12, max_len: int = 88) -> bool:
    if not line or len(line) < min_len or len(line) > max_len:
        return True
    if line_quality_score(line) < 0.58:
        return True
    if "……" in line or "..." in line or line.count("…") >= 2:
        return True
    if re.search(r"(.{{2,10}})\1{{2,}}", line):
        return True
    if sum(ch.isdigit() for ch in line) / len(line) > 0.12:
        return True
    if re.search(r"[=<>|\\[\]{}]{2,}", line):
        return True
    if re.fullmatch(r"[。，、；：！？\s]+", line):
        return True
    if "出版社" in line or ("选集" in line and "卷" in line):
        return True
    if line[-1] not in "。！？":
        return True
    if line.count("，") > 4 or line.count("。") > 1:
        return True
    if re.search(r"[\u4e00-\u9fff]{1,2}的[\u4e00-\u9fff]{1,2}的", line):
        return True
    return False


def filter_clean_lines(lines: Sequence[str]) -> List[str]:
    seen = set()
    cleaned: List[str] = []
    for raw in lines:
        line = deep_clean_line(raw)
        if not line or is_garbage_line(line):
            continue
        key = re.sub(r"[^\u4e00-\u9fff]", "", line)
        if len(key) < 8 or key in seen:
            continue
        seen.add(key)
        cleaned.append(line)
    return cleaned


def format_qa_prompt(instruction: str, input_text: str = "") -> str:
    instruction = instruction.strip()
    input_text = (input_text or "").strip()
    if input_text:
        return f"问：{instruction}\n{input_text}\n答："
    return f"问：{instruction}\n答："


def line_quality_score(line: str) -> float:
    if not line:
        return 0.0
    cjk = sum(1 for ch in line if "\u4e00" <= ch <= "\u9fff")
    digits = sum(ch.isdigit() for ch in line)
    weird = sum(ch in "—-_=|<>[]{}\\" for ch in line)
    ratio_cjk = cjk / len(line)
    ratio_digit = digits / len(line)
    penalty = 0.15 * weird + 0.25 * ratio_digit
    return max(0.0, ratio_cjk - penalty)


def split_paragraphs(corpus: str, min_chars: int = 8) -> List[str]:
    parts = re.split(r"\n\s*\n", corpus)
    docs: List[str] = []
    for part in parts:
        part = part.strip()
        if len(part) >= min_chars:
            docs.append(part)

    # Line-oriented corpora (e.g. quote collections) often have few blank-line breaks.
    if len(docs) < 20:
        line_docs: List[str] = []
        for line in corpus.split("\n"):
            line = line.strip()
            if len(line) >= min_chars and line_quality_score(line) >= 0.2:
                line_docs.append(line)
        if len(line_docs) > len(docs):
            docs = line_docs

    if not docs and corpus.strip():
        docs = [corpus.strip()]
    return docs


def split_documents(
    docs: Sequence[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Invalid split ratios. Require 0 < train_split and train+val < 1.")
    rng = random.Random(seed)
    indices = list(range(len(docs)))
    rng.shuffle(indices)
    n = len(docs)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train_docs = [docs[i] for i in indices[:train_end]]
    val_docs = [docs[i] for i in indices[train_end:val_end]]
    test_docs = [docs[i] for i in indices[val_end:]]
    return train_docs, val_docs, test_docs


def join_documents(docs: Sequence[str]) -> str:
    return "\n\n".join(docs).strip() + "\n"


def build_sft_pairs(
    lines: Sequence[str],
    min_line_len: int = 14,
    min_completion_len: int = 6,
    quality_threshold: float = 0.35,
    split_ratios: Sequence[float] = (0.35, 0.45, 0.55),
    max_pairs: int = 8000,
    seed: int = 42,
) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    pairs: List[Dict[str, str]] = []
    for line in lines:
        line = line.strip()
        if len(line) < min_line_len:
            continue
        if line_quality_score(line) < quality_threshold:
            continue
        ratios = list(split_ratios)
        rng.shuffle(ratios)
        for ratio in ratios:
            k = max(4, int(len(line) * ratio))
            if k >= len(line) - min_completion_len:
                continue
            prompt = line[:k]
            completion = line[k:]
            if len(completion) < min_completion_len:
                continue
            pairs.append({"prompt": prompt, "completion": completion})
            if len(pairs) >= max_pairs:
                return pairs
    rng.shuffle(pairs)
    return pairs[:max_pairs]


def build_qa_pairs(
    lines: Sequence[str],
    max_pairs: int = 12000,
    seed: int = 42,
) -> List[Dict[str, str]]:
    """Build instruction/input/output QA records from clean lines."""
    rng = random.Random(seed)
    records: List[Dict[str, str]] = []

    for line in lines:
        line = line.strip()
        if is_garbage_line(line):
            continue

        templates: List[Dict[str, str]] = []

        for ratio in (0.35, 0.45, 0.55):
            k = max(4, int(len(line) * ratio))
            if k < len(line) - 6:
                templates.append(
                    {
                        "instruction": "请续写下面的句子，只输出后半句，不要重复题干。",
                        "input": line[:k],
                        "output": line[k:],
                    }
                )

        if len(line) >= 12:
            head = line[: min(10, len(line) // 2)]
            templates.append(
                {
                    "instruction": "补全下列句子的完整表述，只输出整句。",
                    "input": f"句首：{head}",
                    "output": line,
                }
            )

        if "，" in line or "。" in line:
            parts = re.split(r"[，。]", line, maxsplit=1)
            if len(parts) == 2 and len(parts[0]) >= 4 and len(parts[1]) >= 4:
                templates.append(
                    {
                        "instruction": "写出与下面半句配套的后半句。",
                        "input": parts[0] + "，",
                        "output": parts[1].lstrip("，"),
                    }
                )

        theme = line[:6] if len(line) >= 6 else line
        templates.append(
            {
                "instruction": "写出与主题相关的经典论述原句，只输出一句话。",
                "input": f"主题：{theme}",
                "output": line,
            }
        )

        rng.shuffle(templates)
        for row in templates[:3]:
            records.append(row)
            if len(records) >= max_pairs:
                rng.shuffle(records)
                return records[:max_pairs]

    rng.shuffle(records)
    return records[:max_pairs]


def qa_records_to_sft_rows(records: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for row in records:
        rows.append(
            {
                "prompt": format_qa_prompt(row["instruction"], row.get("input", "")),
                "completion": row["output"],
                "instruction": row["instruction"],
                "input": row.get("input", ""),
                "output": row["output"],
            }
        )
    return rows


def save_sft_jsonl(path: Path, pairs: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in pairs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
