"""
Deep-clean corpus lines and export QA-style SFT data (问/答 format).

Usage:
  python project/src/build_qa_data.py --config project/configs/medium.yaml
"""
from __future__ import annotations

import json
from pathlib import Path

from data_utils import (
    build_qa_pairs,
    clean_text,
    filter_clean_lines,
    qa_records_to_sft_rows,
    save_sft_jsonl,
)
from utils import load_config, parse_common_args


def load_line_sources(raw_dir: Path, processed_train: Path) -> list[str]:
    lines: list[str] = []
    for path in sorted(raw_dir.rglob("*.txt")):
        if path.name.startswith("."):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if path.name.endswith("_quality_lines.txt") or "quotes" in path.name:
            lines.extend(ln.strip() for ln in text.splitlines() if ln.strip())
        else:
            lines.extend(ln.strip() for ln in clean_text(text).splitlines() if ln.strip())
    if processed_train.exists():
        lines.extend(
            ln.strip() for ln in processed_train.read_text(encoding="utf-8").splitlines() if ln.strip()
        )
    return lines


def main() -> None:
    args = parse_common_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    raw_dir = Path(data_cfg["raw_dir"])
    processed_dir = Path(data_cfg["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_path = Path(data_cfg["train_path"])
    raw_lines = load_line_sources(raw_dir, train_path)
    clean_lines = filter_clean_lines(raw_lines)

    cleaned_path = processed_dir / "cleaned_lines.txt"
    cleaned_path.write_text("\n".join(clean_lines) + "\n", encoding="utf-8")

    qa_records = build_qa_pairs(
        clean_lines,
        max_pairs=int(data_cfg.get("qa_max_pairs", 12000)),
        seed=int(cfg.get("seed", 42)),
    )
    qa_path = Path(data_cfg.get("qa_path", str(processed_dir / "qa.jsonl")))
    sft_rows = qa_records_to_sft_rows(qa_records)

    sft_path = Path(data_cfg.get("sft_path", str(processed_dir / "sft.jsonl")))
    save_sft_jsonl(qa_path, qa_records)
    save_sft_jsonl(sft_path, sft_rows)

    sample_path = processed_dir / "qa_samples.json"
    sample_path.write_text(
        json.dumps(qa_records[:20], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Raw lines      : {len(raw_lines)}")
    print(f"Clean lines    : {len(clean_lines)} -> {cleaned_path}")
    print(f"QA records     : {len(qa_records)} -> {qa_path}")
    print(f"SFT rows       : {len(sft_rows)} -> {sft_path}")
    print(f"Sample preview : {sample_path}")


if __name__ == "__main__":
    main()
