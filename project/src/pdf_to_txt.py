import argparse
from pathlib import Path

from data_utils import clean_text
from pypdf import PdfReader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=str, required=True, help="Input PDF path")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output txt path. Defaults to same name beside PDF.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    out_path = Path(args.out) if args.out else pdf_path.with_suffix(".txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    reader = PdfReader(str(pdf_path))
    chunks = []
    for page in reader.pages:
        text = page.extract_text() or ""
        chunks.append(text.strip())

    merged = clean_text("\n\n".join(chunks).strip()) + "\n"
    out_path.write_text(merged, encoding="utf-8")

    print(f"pdf: {pdf_path}")
    print(f"pages: {len(reader.pages)}")
    print(f"chars: {len(merged)}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
