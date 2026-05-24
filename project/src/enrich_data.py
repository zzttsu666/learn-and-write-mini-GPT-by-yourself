"""Enrich project/data/raw from PDFs, local bundles, and optional online sources."""
from __future__ import annotations

import argparse
import html
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable, List, Optional

from data_utils import clean_text, line_quality_score
# Famous shorter 选集篇目 (readthedocs mirrors, educational use).
XUANJI_ARTICLE_SLUGS = [
    "005-%E6%98%9F%E6%98%9F%E4%B9%8B%E7%81%AB%EF%BC%8C%E5%8F%AF%E4%BB%A5%E7%87%8E%E5%8E%9F",
    "006-%E5%8F%8D%E5%AF%B9%E6%9C%AC%E6%9C%AC%E4%B8%BB%E4%B9%89",
    "020-%E5%8F%8D%E5%AF%B9%E8%87%AA%E7%94%B1%E4%B8%BB%E4%B9%89",
    "025-%E6%8A%97%E6%97%A5%E6%B8%B8%E5%87%BB%E6%88%98%E4%BA%89%E7%9A%84%E6%88%98%E7%95%A5%E9%97%AE%E9%A2%98",
    "030-%E4%BA%94%E5%9B%9B%E8%BF%90%E5%8A%A8",
    "042-%E7%BA%AA%E5%BF%B5%E7%99%BD%E6%B1%82%E6%81%A9",
    "059-%E6%94%B9%E9%80%A0%E6%88%91%E4%BB%AC%E7%9A%84%E5%AD%A6%E4%B9%A0",
    "063-%E6%95%B4%E9%A1%BF%E5%85%9A%E7%9A%84%E4%BD%9C%E9%A3%8E",
    "064-%E5%8F%8D%E5%AF%B9%E5%85%9A%E5%85%AB%E8%82%A1",
    "076-%E4%B8%BA%E4%BA%BA%E6%B0%91%E6%9C%8D%E5%8A%A1",
    "083-%E6%84%9A%E5%85%AC%E7%A7%BB%E5%B1%B1",
    "094-%E5%85%B3%E4%BA%8E%E9%87%8D%E5%BA%86%E8%B0%88%E5%88%A4",
    "101-%E5%92%8C%E7%BE%8E%E5%9B%BD%E8%AE%B0%E8%80%85%E5%AE%89%E5%A8%9C%C2%B7%E8%B7%AF%E6%98%93%E6%96%AF%C2%B7%E6%96%AF%E7%89%B9%E6%9C%97%E7%9A%84%E8%B0%88%E8%AF%9D",
    "102-%E9%9B%86%E4%B8%AD%E4%BC%98%E5%8A%BF%E5%85%B5%E5%8A%9B%EF%BC%8C%E5%90%84%E4%B8%AA%E6%AD%BC%E7%81%AD%E6%95%8C%E4%BA%BA",
    "124-%E5%AF%B9%E6%99%8B%E7%BB%A5%E6%97%A5%E6%8A%A5%E7%BC%96%E8%BE%91%E4%BA%BA%E5%91%98%E7%9A%84%E8%B0%88%E8%AF%9D",
    "136-%E5%B0%86%E9%9D%A9%E5%91%BD%E8%BF%9B%E8%A1%8C%E5%88%B0%E5%BA%95",
    "147-%E5%85%9A%E5%A7%94%E4%BC%9A%E7%9A%84%E5%B7%A5%E4%BD%9C%E6%96%B9%E6%B3%95",
    "154-%E4%B8%A2%E6%8E%89%E5%B9%BB%E6%83%B3%EF%BC%8C%E5%87%86%E5%A4%87%E6%96%97%E4%BA%89",
    "155-%E5%88%AB%E4%BA%86%EF%BC%8C%E5%8F%B8%E5%BE%92%E9%9B%B7%E7%99%BB",
    "159-%E4%B8%AD%E5%9B%BD%E4%BA%BA%E6%B0%91%E7%AB%99%E8%B5%B7%E6%9D%A5%E4%BA%86",
    "210-%E7%BE%8E%E5%B8%9D%E5%9B%BD%E4%B8%BB%E4%B9%89%E6%98%AF%E7%BA%B8%E8%80%81%E8%99%8E",
    "228-%E4%B8%80%E5%88%87%E5%8F%8D%E5%8A%A8%E6%B4%BE%E9%83%BD%E6%98%AF%E7%BA%B8%E8%80%81%E8%99%8E",
]

XUANJI_BASE = "https://maozedong-xuanji.readthedocs.io/zh-cn/src"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=str, default="project/data/raw")
    parser.add_argument("--fetch-online", action="store_true", default=False)
    parser.add_argument("--max-online-articles", type=int, default=22)
    return parser.parse_args()


def convert_pdfs(raw_dir: Path) -> None:
    from pypdf import PdfReader

    for pdf in sorted(raw_dir.glob("*.pdf")):
        out = pdf.with_suffix(".txt")
        reader = PdfReader(str(pdf))
        chunks = [(page.extract_text() or "").strip() for page in reader.pages]
        merged = clean_text("\n\n".join(chunks).strip()) + "\n"
        out.write_text(merged, encoding="utf-8")
        print(f"[pdf] {pdf.name} -> {out.name} ({len(merged)} chars)")


def dedupe_lines(text: str) -> str:
    seen = set()
    out: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line in seen:
            continue
        if line_quality_score(line) < 0.15:
            continue
        seen.add(line)
        out.append(line)
    return "\n".join(out) + "\n"


def extract_html_text(page_html: str) -> str:
    for pattern in (
        r'<div class="document"[^>]*>(.*?)</div>\s*<div class="clearer"',
        r'<div role="main"[^>]*>(.*?)</div>\s*</div>\s*<footer',
        r"<article[^>]*>(.*?)</article>",
    ):
        match = re.search(pattern, page_html, flags=re.S)
        if not match:
            continue
        body = match.group(1)
        body = re.sub(r"<script[^>]*>.*?</script>", "", body, flags=re.S | re.I)
        body = re.sub(r"<style[^>]*>.*?</style>", "", body, flags=re.S | re.I)
        body = re.sub(r"<[^>]+>", "\n", body)
        body = html.unescape(body)
        body = clean_text(body)
        if len(body) > 200:
            return body
    return ""


def fetch_xuanji_article(slug: str, timeout: int = 40) -> Optional[str]:
    for suffix in ("/index.html", "/"):
        url = f"{XUANJI_BASE}/{slug}{suffix}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; learn-gpt-enrich/1.0)"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                page = resp.read().decode("utf-8", errors="ignore")
            text = extract_html_text(page)
            if text:
                return text
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            continue
    return None


def fetch_online_corpus(raw_dir: Path, slugs: Iterable[str], max_chars_per_article: int = 12000) -> int:
    saved = 0
    out_dir = raw_dir / "xuanji_online"
    out_dir.mkdir(parents=True, exist_ok=True)
    for slug in slugs:
        text = fetch_xuanji_article(slug)
        time.sleep(0.35)
        if not text:
            print(f"[skip] online article failed: {slug}")
            continue
        if len(text) > max_chars_per_article:
            text = text[:max_chars_per_article]
        text = dedupe_lines(text)
        out_path = out_dir / f"{slug.split('-')[0]}_{slug[:48]}.txt"
        out_path.write_text(text, encoding="utf-8")
        saved += 1
        print(f"[online] saved {out_path.name} ({len(text)} chars)")
    return saved


def extract_quality_lines_from_corpus(raw_dir: Path, min_len: int = 12) -> None:
    """Mine high-quality single lines from the largest local txt source."""
    candidates = sorted(raw_dir.glob("*.txt"), key=lambda p: p.stat().st_size, reverse=True)
    if not candidates:
        return
    source = candidates[0]
    lines_out: List[str] = []
    seen = set()
    for line in source.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if len(line) < min_len or line in seen:
            continue
        if line_quality_score(line) < 0.45:
            continue
        seen.add(line)
        lines_out.append(line)
    out_path = raw_dir / "yulu_quality_lines.txt"
    out_path.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    print(f"[mine] {source.name} -> {out_path.name} ({len(lines_out)} lines)")


def write_local_bundles(raw_dir: Path) -> None:
    quotes_path = raw_dir / "mao_quotes_extended.txt"
    if not quotes_path.exists():
        quotes = """
中国革命的胜利，主要依靠统一战线、武装斗争、党的建设这三个法宝。
没有调查，就没有发言权。
实事求是，是马克思主义的根本观点。
为人民服务，是我们一切工作的出发点和落脚点。
星星之火，可以燎原。
人民，只有人民，才是创造世界历史的动力。
一切反动派都是纸老虎。
战略上藐视敌人，战术上重视敌人。
政策和策略是党的生命。
思想上政治上的路线正确与否是决定一切的。
自力更生，艰苦奋斗。
从战争中学习战争。
团结—批评—团结。
百花齐放，百家争鸣。
愚公移山，改造中国与世界。
调查就像十月怀胎，解决问题就像一朝分娩。
没有文化的军队是愚蠢的军队，而愚蠢的军队是不能战胜敌人的。
我们共产党人好比种子，人民好比土地。
青年是整个社会力量中的一部分最积极最有生气的力量。
世界是你们的，也是我们的，但是归根结底是你们的。
好好学习，天天向上。
下定决心，不怕牺牲，排除万难，去争取胜利。
宜将剩勇追穷寇，不可沽名学霸王。
枪杆子里面出政权。
政治工作是一切经济工作的生命线。
一切为了群众，一切依靠群众，从群众中来，到群众中去。
不打无准备之仗，不打无把握之仗。
集中优势兵力，各个歼灭敌人。
前途是光明的，道路是曲折的。
谦虚使人进步，骄傲使人落后。
敌进我退，敌驻我扰，敌疲我打，敌退我追。
军民团结如一人，试看天下谁能敌。
发展才是硬道理。
摸着石头过河。
不管黑猫白猫，捉到老鼠就是好猫。
空谈误国，实干兴邦。
绿水青山就是金山银山。
不忘初心，牢记使命。
为中华之崛起而读书。
苟利国家生死以，岂因祸福避趋之。
天下兴亡，匹夫有责。
生于忧患，死于安乐。
千里之行，始于足下。
知之为知之，不知为不知，是知也。
工欲善其事，必先利其器。
锲而不舍，金石可镂。
青，取之于蓝，而青于蓝。
博学之，审问之，慎思之，明辨之，笃行之。
""".strip()
        quotes_path.write_text(dedupe_lines(quotes), encoding="utf-8")
        print(f"[local] wrote {quotes_path.name}")

    classics_path = raw_dir / "classics_zh_public.txt"
    if not classics_path.exists():
        classics = """
学而时习之，不亦说乎。有朋自远方来，不亦乐乎。
温故而知新，可以为师矣。
学而不思则罔，思而不学则殆。
知之者不如好之者，好之者不如乐之者。
三人行，必有我师焉。择其善者而从之，其不善者而改之。
己所不欲，勿施于人。
君子坦荡荡，小人长戚戚。
不患人之不己知，患不知人也。
路漫漫其修远兮，吾将上下而求索。
长太息以掩涕兮，哀民生之多艰。
海内存知己，天涯若比邻。
落霞与孤鹜齐飞，秋水共长天一色。
欲穷千里目，更上一层楼。
会当凌绝顶，一览众山小。
春蚕到死丝方尽，蜡炬成灰泪始干。
先天下之忧而忧，后天下之乐而乐。
不以物喜，不以己悲。
问君能有几多愁，恰似一江春水向东流。
山重水复疑无路，柳暗花明又一村。
纸上得来终觉浅，绝知此事要躬行。
""".strip()
        classics_path.write_text(dedupe_lines(classics), encoding="utf-8")
        print(f"[local] wrote {classics_path.name}")


def summarize_raw(raw_dir: Path) -> None:
    total_chars = 0
    total_files = 0
    for path in sorted(raw_dir.rglob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        total_chars += len(text)
        total_files += 1
        print(f"  {path.relative_to(raw_dir)}: {len(text)} chars")
    print(f"[summary] txt files={total_files}, total_chars={total_chars}")


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("== enrich: convert PDFs ==")
    convert_pdfs(raw_dir)

    print("== enrich: local bundles ==")
    write_local_bundles(raw_dir)

    print("== enrich: mine quality lines ==")
    extract_quality_lines_from_corpus(raw_dir)

    if args.fetch_online:
        print("== enrich: fetch 选集 articles ==")
        slugs = XUANJI_ARTICLE_SLUGS[: args.max_online_articles]
        saved = fetch_online_corpus(raw_dir, slugs)
        print(f"[online] saved articles: {saved}/{len(slugs)}")

    print("== enrich: raw summary ==")
    summarize_raw(raw_dir)


if __name__ == "__main__":
    main()
