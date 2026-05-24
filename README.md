# Mini GPT：从零理解「小模型是怎么炼出来的」

> 一个给 **零基础小白** 用的教学项目。  
> 你不会看到黑盒 API，只会看到：**数据 → 分词 → Transformer → 预训练 → 微调 → 生成** 的完整链路。

---

## 你将学到什么

| 阶段 | 你在学什么 | 对应脚本 |
|------|------------|----------|
| 1. 数据 | 文本清洗、划分训练/验证集、构建词表 | `prepare_data.py` |
| 2. 问答数据 | 把句子变成「问 / 答」监督样本 | `build_qa_data.py` |
| 3. 预训练 | 下一个字预测（语言模型） | `train.py` |
| 4. 微调 SFT | 只学「答」的部分，学会按题目回答 | `train_sft.py` |
| 5. 推理 | 采样、去重复、问答题格式 | `generate.py` |

**重要认知（请先读）：**

- 这是 **字符级** 小 GPT（每个汉字一个 token），不是 ChatGPT 那种大模型。
- 生成质量主要取决于 **数据量与数据质量**，不是 loss 越低越好（loss 很低也可能是「背答案」）。
- **BERT 不能拿来生成文本**；生成用 GPT 这类「从左到右」模型。本项目就是迷你版 GPT。

---

## 项目结构

```text
project/
├── configs/              # 超参数（模型大小、学习率、训练步数）
│   ├── tiny.yaml         # CPU 冒烟测试（几分钟）
│   ├── medium.yaml       # 推荐：4–8GB 显卡
│   ├── sft.yaml          # 问答微调配置
│   └── base.yaml         # 更大模型（语料够多时用）
├── data/
│   ├── raw/              # 放你自己的 .txt（仓库只带极小示例）
│   └── processed/        # 运行脚本后自动生成（已 gitignore）
├── checkpoints/          # 训练得到的 .pt 权重（已 gitignore）
├── src/
│   ├── model.py          # ★ GPT：Embedding + Transformer + 输出头
│   ├── tokenizer.py      # 字符词表
│   ├── dataset.py        # 随机切块、SFT 的 loss 掩码
│   ├── train.py          # 预训练循环
│   ├── train_sft.py      # 微调循环
│   ├── generate.py       # 生成（自动套「问 / 答」模板）
│   ├── build_qa_data.py  # 清洗 + 构造问答题
│   ├── prepare_data.py   # 语料合并与划分
│   ├── evaluate.py       # 困惑度 + 样例生成
│   ├── enrich_data.py    # 可选：PDF 转 txt、示例扩充
│   └── run_all.py        # 一键跑全流程
└── requirements.txt
```

---

## 环境准备

- Python 3.10+
- 建议：NVIDIA GPU + CUDA（`medium.yaml` 默认 `device: cuda`）
- 仅 CPU 也可跑通：用 `project/configs/tiny.yaml`

```bash
# 在仓库根目录执行
pip install -r project/requirements.txt
```

---

## 30 分钟上手（推荐顺序）

在仓库**根目录**执行（不要进 `project/` 再执行）：

```bash
# 0）可选：往 project/data/raw/ 放入你自己的 UTF-8 文本

# 1）一键：准备数据 → 问答题 → 预训练 →（条件满足则）SFT → 评估
python project/src/run_all.py

# 2）用微调后的模型生成（问答题格式会自动套上）
python project/src/generate.py --config project/configs/sft.yaml --prompt "为人民服务"
```

第一次没有权重时，`run_all.py` 会训练并写入：

- `project/checkpoints/pretrain/best.pt` — 预训练最优
- `project/checkpoints/sft/best.pt` — 微调最优（生成默认用这个）

---

## 分步执行（适合上课讲解）

### Step 1：准备语料

```bash
python project/src/prepare_data.py --config project/configs/medium.yaml
```

产出：

- `project/data/processed/train.txt` / `val.txt` / `test.txt`
- `project/data/processed/vocab.json` — 字符词表

**原理**：把 `project/data/raw/*.txt` 合并 → 清洗噪声行 → 按行/段随机划分，避免「前 90% 训练、后 10% 验证」造成分布不一致。

### Step 2：构造问答题（SFT 数据）

```bash
python project/src/build_qa_data.py --config project/configs/medium.yaml
```

产出：

- `project/data/processed/cleaned_lines.txt` — 清洗后的单句
- `project/data/processed/qa.jsonl` — 结构化 `instruction / input / output`
- `project/data/processed/sft.jsonl` — 训练用 `问：…\n答：…` 格式

**原理**：从高质量短句自动生成「续写 / 补全 / 主题句」等题型，让模型学会 **按问题作答**，而不是乱续写。

### Step 3：预训练（语言模型）

```bash
python project/src/train.py --config project/configs/medium.yaml
```

**原理**：给定前文，预测下一个字（交叉熵 loss）。相当于让模型「熟读」语料。

看日志：

- `train_loss` 与 `val_loss` 应同向下降
- 若 `train` 很低、`val` 很高 → 过拟合，需更多数据或更大 `dropout`

### Step 4：微调 SFT

```bash
python project/src/train_sft.py --config project/configs/sft.yaml
```

**原理**：加载 `pretrain/best.pt`，只在「答：」后面的 token 上算 loss（题干部分 mask 掉）。

### Step 5：生成

```bash
python project/src/generate.py --config project/configs/sft.yaml --prompt "实事求是"
```

SFT 模型会自动使用：

```text
问：写出与主题相关的经典论述原句，只输出一句话。
主题：实事求是
答：
```

并 **只打印第一句**，避免一长串胡言乱语。

预训练风格（不套问答题）：

```bash
python project/src/generate.py --config project/configs/medium.yaml \
  --checkpoint project/checkpoints/pretrain/best.pt \
  --prompt "实事求是" --raw-prompt
```

### Step 6：评估

```bash
python project/src/evaluate.py \
  --config project/configs/sft.yaml \
  --checkpoint project/checkpoints/sft/best.pt \
  --split test
```

输出验证集 **困惑度（perplexity）** 和几个固定 prompt 的生成样例。

---

## 模型长什么样？（`model.py` 一图流）

```text
输入 token 序列
    ↓
Token Embedding + Position Embedding
    ↓
× N 层 Block：
    LayerNorm → 因果自注意力（只能看左边）→ 残差
    LayerNorm → MLP（FFN）→ 残差
    ↓
LayerNorm → Linear → 词表大小 logits
    ↓
预测下一个字 / 计算 loss
```

核心代码不到 200 行，适合对照 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) 与 [nanoGPT](https://github.com/karpathy/nanoGPT) 阅读。

---

## 配置文件怎么选？

| 文件 | 适用场景 |
|------|----------|
| `tiny.yaml` | 没 GPU、只想验证代码能跑 |
| `medium.yaml` | **默认推荐**，约千万级参数量级 |
| `base.yaml` | 语料 > 百万字、显卡 ≥ 8GB |
| `sft.yaml` | 微调专用，学习率更小 |

显存不够：先把 `batch_size` 改为 `4` 或 `2`。

---

## 训练时看什么指标？

| 现象 | 含义 | 建议 |
|------|------|------|
| train ↓ val ↓ | 正常学习 | 继续训练 |
| train ↓ val 不动 | 开始过拟合 | 用 `best.pt`、加数据、早停 |
| train 很低 val 很高 | 严重过拟合 | 增 `dropout`、减 `max_steps`、清洗数据 |
| 生成重复一句 | 采样太长或 SFT 过拟合 | 用默认 `first_sentence_only`、调 `repetition_penalty` |

**务必使用 `best.pt`，不要用 `final.pt`。**

---

## 数据与版权（开源必读）

- 仓库 **不包含** 任何书籍全文 PDF，仅提供 **极短示例** 文本。
- 请自行准备 **有权使用** 的语料放入 `project/data/raw/`。
- 大文件、PDF、`checkpoints/*.pt`、`project/data/processed/*` 已在 `.gitignore` 中忽略，避免误传到 GitHub。

---

## 常见问题

**Q：loss 已经 0.0x 了，为什么生成还很差？**  
A：小数据 + 字符模型容易「背训练集」。看 **验证集 loss** 和真实生成，不要只看训练 loss。

**Q：能用 BERT 吗？**  
A：BERT 适合分类/理解，不适合续写。想更好效果请用 **更大的语料** 或 **中文预训练 GPT + 本项目的 SFT 数据格式** 做微调。

**Q：怎么让效果更好？**  
A：① 清洗数据 ② 语料量加大 ③ BPE 分词（进阶）④ 换 `base.yaml` ⑤ 使用预训练小模型微调。

---

## 实验记录

每次训练可在 `project/experiments/README.md` 模板里记录：配置、loss、生成样例，方便对比实验。

---

## 开源与贡献

- License: [MIT](LICENSE)
- 欢迎 Issue / PR：文档纠错、更清晰的注释、更好的示例数据（须确保可开源）

---

## 命令速查

```bash
# 全流程
python project/src/run_all.py

# 仅数据
python project/src/prepare_data.py --config project/configs/medium.yaml
python project/src/build_qa_data.py --config project/configs/medium.yaml

# 训练
python project/src/train.py --config project/configs/medium.yaml
python project/src/train_sft.py --config project/configs/sft.yaml

# 生成 / 评估
python project/src/generate.py --config project/configs/sft.yaml --prompt "你的主题"
python project/src/evaluate.py --config project/configs/sft.yaml \
  --checkpoint project/checkpoints/sft/best.pt --split test
```

祝学习顺利：亲手跑通一遍，比看十篇博客更有用。
