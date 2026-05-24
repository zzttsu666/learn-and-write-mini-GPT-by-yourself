# 原始语料目录

请把你**有权使用**的 `.txt` 文件放在这里（UTF-8 编码）。

仓库自带三份**极小示例**（共几百行），仅用于跑通流程、学习训练步骤：

- `sample_mao_public.txt` — 6 条示例短句
- `mao_quotes_extended.txt` — 扩展名句示例
- `classics_zh_public.txt` — 公版古文/诗词短句

**不要**把受版权保护的全书 PDF/全文提交到 GitHub。若你有 PDF，可在本地转换：

```bash
python project/src/pdf_to_txt.py --pdf "你的文件.pdf"
```

转换后的 `.txt` 会生成在本目录；默认已被 `.gitignore` 忽略大文件。
