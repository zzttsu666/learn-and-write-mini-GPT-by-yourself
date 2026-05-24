from pathlib import Path

from data_utils import (
    build_sft_pairs,
    clean_text,
    join_documents,
    save_sft_jsonl,
    split_documents,
    split_paragraphs,
)
from tokenizer import CharTokenizer
from utils import load_config, parse_common_args


def main() -> None:
    args = parse_common_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]

    raw_dir = Path(data_cfg["raw_dir"])
    processed_dir = Path(data_cfg["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(raw_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(
            f"No .txt files found in {raw_dir}. Put source texts there first."
        )

    merged = []
    for p in txt_files:
        content = p.read_text(encoding="utf-8", errors="ignore")
        merged.append(clean_text(content))

    corpus = "\n\n".join(merged).strip() + "\n"
    docs = split_paragraphs(corpus, min_chars=int(data_cfg.get("min_paragraph_chars", 8)))

    train_docs, val_docs, test_docs = split_documents(
        docs,
        train_ratio=float(data_cfg.get("train_split", 0.9)),
        val_ratio=float(data_cfg.get("val_split", 0.05)),
        seed=int(cfg.get("seed", 42)),
    )

    train_text = join_documents(train_docs)
    val_text = join_documents(val_docs)
    test_text = join_documents(test_docs)

    corpus_path = processed_dir / "corpus.txt"
    train_path = Path(data_cfg["train_path"])
    val_path = Path(data_cfg["val_path"])
    test_path = Path(data_cfg.get("test_path", str(processed_dir / "test.txt")))
    vocab_path = Path(data_cfg["vocab_path"])
    sft_path = Path(data_cfg.get("sft_path", str(processed_dir / "sft.jsonl")))

    corpus_path.write_text(corpus, encoding="utf-8")
    train_path.write_text(train_text, encoding="utf-8")
    val_path.write_text(val_text, encoding="utf-8")
    test_path.write_text(test_text, encoding="utf-8")

    vocab_source = train_text + val_text
    tokenizer = CharTokenizer.from_text(vocab_source)
    tokenizer.save(str(vocab_path))

    train_lines = [ln.strip() for ln in train_text.split("\n") if ln.strip()]
    sft_pairs = build_sft_pairs(
        train_lines,
        min_line_len=int(data_cfg.get("sft_min_line_len", 14)),
        quality_threshold=float(data_cfg.get("sft_quality_threshold", 0.35)),
        max_pairs=int(data_cfg.get("sft_max_pairs", 8000)),
        seed=int(cfg.get("seed", 42)),
    )
    save_sft_jsonl(sft_path, sft_pairs)

    print(f"Loaded files : {len(txt_files)}")
    print(f"Paragraphs   : {len(docs)}")
    print(f"Corpus chars : {len(corpus)}")
    print(f"Train chars  : {len(train_text)} ({len(train_docs)} docs)")
    print(f"Val chars    : {len(val_text)} ({len(val_docs)} docs)")
    print(f"Test chars   : {len(test_text)} ({len(test_docs)} docs)")
    print(f"Vocab size   : {tokenizer.vocab_size}")
    print(f"SFT pairs    : {len(sft_pairs)} -> {sft_path}")
    print(f"Saved to     : {processed_dir}")


if __name__ == "__main__":
    main()
