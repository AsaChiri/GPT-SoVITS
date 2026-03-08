# 批量语音合成 (batch_inference.py)

这个脚本可以把文本批量转换成语音。它会根据每行文字的情感，自动选择最合适的参考音频，让合成出来的语音更加自然。

---

## 开始之前，你需要准备

1. **训练好的语音模型** — 每个说话人需要一个 GPT 权重文件（`.ckpt`）和一个 SoVITS 权重文件（`.pth`）
2. **参考音频** — 每个说话人的短音频片段（3-10 秒），以及对应的文字稿
3. **输入文本文件** — 列出你想让角色说的台词

---

## 文件夹结构

把文件按如下结构放好：

```
GPT-SoVITS/                          （或 GPT-SoVITS-v2pro-20250604/）
├── batch_inference.py
├── run_batch.bat                     ← 双击即可运行
│
├── GPT_weights_v2ProPlus/            ← 训练好的 GPT 权重放这里
│   ├── alisa-e15.ckpt
│   └── hiro-e15.ckpt
│
├── SoVITS_weights_v2ProPlus/         ← 训练好的 SoVITS 权重放这里
│   ├── alisa_e8_s456.pth
│   └── hiro_e8_s2184.pth
│
├── inputs/
│   ├── speaker_config.yaml           ← 配置每个说话人的信息
│   ├── alisa.list                    ← alisa 的参考音频列表
│   ├── hiro.list                     ← hiro 的参考音频列表
│   ├── alisa_audio/                  ← alisa 的参考音频文件
│   │   ├── clip001.wav
│   │   └── clip002.wav
│   ├── hiro_audio/                   ← hiro 的参考音频文件
│   │   ├── clip001.wav
│   │   └── clip002.wav
│   └── mod_input/                    ← 要合成的文本文件
│       └── dialogue.list
│
└── output/                           ← 生成的音频会出现在这里
```

---

## 文件格式说明

### 参考音频列表（如 `inputs/alisa.list`）

每行描述一条参考音频：

```
音频路径|说话人名|语言|文字稿
```

示例：
```
alisa_audio/clip001.wav|alisa|ja|おはようございます、今日はいい天気ですね
alisa_audio/clip002.wav|alisa|ja|ありがとう、とても嬉しいです
```

- `音频路径` — 音频文件的路径，相对于 `inputs/` 文件夹
- `说话人名` — 说话人的名字
- `语言` — `ja`（日语）、`en`（英语）、`zh`（中文）、`ko`（韩语）
- `文字稿` — 音频中说话人说的内容

**注意：** 参考音频的时长必须在 **3 到 10 秒**之间，过短或过长的会被跳过。

### 输入文本列表（如 `inputs/mod_input/dialogue.list`）

每行描述一条要生成的语音：

```
输出路径|说话人名|语言|要合成的文本
```

示例：
```
alisa/line001.wav|alisa|ja|今日の冒険は楽しかったね
alisa/line002.wav|alisa|ja|気をつけてね、危ないよ
hiro/line001.wav|hiro|ja|俺に任せろ、絶対に守ってやる
hiro/line002.wav|hiro|ja|まあ、悪くないな
```

- `输出路径` — 生成的音频保存位置（相对于 `output/` 文件夹）
- `说话人名` — 必须和 `speaker_config.yaml` 中的名字一致
- `语言` — 文本的语言
- `要合成的文本` — 想让角色说的话

### 说话人配置（inputs/speaker_config.yaml）

告诉脚本每个说话人用哪个模型和哪些参考音频：

```yaml
speakers:
  alisa:
    gpt_path: GPT_weights_v2ProPlus/alisa-e15.ckpt
    sovits_path: SoVITS_weights_v2ProPlus/alisa_e8_s456.pth
    ref_list: inputs/alisa.list
    ref_audio_dir: inputs/
  hiro:
    gpt_path: GPT_weights_v2ProPlus/hiro-e15.ckpt
    sovits_path: SoVITS_weights_v2ProPlus/hiro_e8_s2184.pth
    ref_list: inputs/hiro.list
    ref_audio_dir: inputs/
```

- `gpt_path` / `sovits_path` — 训练好的模型文件路径
- `ref_list` — 参考音频列表文件的路径
- `ref_audio_dir` — 参考音频文件所在的父文件夹

---

## 如何运行

### 方式 A：双击运行（整合包）

1. 按上面的结构放好文件
2. 双击 **`run_batch.bat`**
3. 会弹出一个终端窗口，显示处理进度
4. 完成后，去 `output/` 文件夹查看生成的音频

### 方式 B：命令行运行（整合包）

在 GPT-SoVITS 文件夹中打开终端，输入：

```
chcp 65001
runtime\python.exe batch_inference.py --input_list inputs/mod_input/dialogue.list --output_dir output/ --speaker_config inputs/speaker_config.yaml --output_sr 44100 --output_channels 2
```

### 方式 C：开发环境（uv）

```bash
chcp 65001
uv run batch_inference.py \
  --input_list inputs/mod_input/dialogue.list \
  --output_dir output/ \
  --speaker_config inputs/speaker_config.yaml \
  --output_sr 44100 \
  --output_channels 2
```

### 常用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input_list` | （必填） | 输入的 .list 文件 |
| `--output_dir` | （必填） | 输出音频的文件夹 |
| `--speaker_config` | — | 说话人配置 YAML（多说话人模式） |
| `--mode` | `auto` | `auto`（情感匹配）或 `manual`（单一参考音频） |
| `--output_sr` | 原始采样率 | 输出采样率（如 `44100`） |
| `--output_channels` | 原始声道 | `1` 单声道，`2` 立体声 |
| `--speed_factor` | `1.0` | 语速倍率（0.5 = 半速，2.0 = 倍速） |
| `--temperature` | `1.0` | 随机性（越高越多变化） |
| `--top_k` | `5` | Top-K 采样 |
| `--top_p` | `1.0` | Top-P（核采样） |

---

## 完整示例

以下是一个最小的示例，包含 2 个说话人，各 2 句台词。

### 第 1 步：创建文件

**inputs/speaker_config.yaml**
```yaml
speakers:
  alisa:
    gpt_path: GPT_weights_v2ProPlus/alisa-e15.ckpt
    sovits_path: SoVITS_weights_v2ProPlus/alisa_e8_s456.pth
    ref_list: inputs/alisa.list
    ref_audio_dir: inputs/
  hiro:
    gpt_path: GPT_weights_v2ProPlus/hiro-e15.ckpt
    sovits_path: SoVITS_weights_v2ProPlus/hiro_e8_s2184.pth
    ref_list: inputs/hiro.list
    ref_audio_dir: inputs/
```

**inputs/alisa.list**（alisa 的参考音频列表 — 对应的 .wav 文件也要放好）
```
alisa_audio/happy.wav|alisa|ja|今日はとても楽しいです
alisa_audio/sad.wav|alisa|ja|悲しいことがありました
alisa_audio/neutral.wav|alisa|ja|明日の天気はどうですか
```

**inputs/hiro.list**（hiro 的参考音频列表）
```
hiro_audio/happy.wav|hiro|ja|よし、やったぞ
hiro_audio/angry.wav|hiro|ja|ふざけるな、許さないぞ
hiro_audio/neutral.wav|hiro|ja|そうだな、考えておこう
```

**inputs/mod_input/dialogue.list**（要合成的文本）
```
alisa/line001.wav|alisa|ja|今日の冒険は楽しかったね
alisa/line002.wav|alisa|ja|気をつけてね、危ないよ
hiro/line001.wav|hiro|ja|俺に任せろ、絶対に守ってやる
hiro/line002.wav|hiro|ja|まあ、悪くないな
```

### 第 2 步：运行

双击 `run_batch.bat`，或者在命令行中输入：

```
chcp 65001
runtime\python.exe batch_inference.py --input_list inputs/mod_input/dialogue.list --output_dir output/ --speaker_config inputs/speaker_config.yaml --output_sr 44100 --output_channels 2
```

### 第 3 步：查看结果

生成的文件会出现在：
```
output/
├── alisa/
│   ├── line001.wav
│   └── line002.wav
└── hiro/
    ├── line001.wav
    └── line002.wav
```

---

## 常见问题

### `UnicodeEncodeError: 'gbk' codec can't encode character`

Windows 终端默认使用 GBK 编码，无法显示某些特殊字符。解决方法：运行脚本前先执行 `chcp 65001`。`run_batch.bat` 已经自动处理了这个问题。

### `.ogg` 参考音频报错（`LibsndfileError`）

Windows 上的 libsndfile 不支持 OGG 格式。脚本会自动用 librosa 来处理。如果仍然报错，建议把参考音频转换为 `.wav` 格式。

### `LookupError: averaged_perceptron_tagger_eng`

英文文本处理需要这个资源包。修复方法 — 运行一次：
```
runtime\python.exe -c "import nltk; nltk.download('averaged_perceptron_tagger_eng')"
```

### 参考音频找不到 / 路径错误

如果参考列表中的路径是 `alisa_audio/clip001.wav` 这样的相对路径，那么 `ref_audio_dir` 应该设为 `inputs/`（父文件夹），**而不是** `inputs/alisa_audio/`。否则路径会变成 `inputs/alisa_audio/alisa_audio/clip001.wav`。

### `No reference audios passed the duration filter!`

所有参考音频的时长必须在 **3 到 10 秒**之间。检查你的音频，裁剪掉过短或过长的片段。

### `No emotion model for language 'yue'`

粤语（yue）没有情感模型。请使用 `--mode manual` 指定一条参考音频，而不是自动模式。

### 情感模型下载失败 / 没有网络

整合包可以离线使用。在有网络的环境下运行一次 `download_emotion_models.py` 来预下载模型：
```
runtime\python.exe download_emotion_models.py
```
模型会保存到 `GPT_SoVITS/pretrained_models/emotion-{ja,en,zh,ko}/`。之后就不再需要联网了。

---

# 开发者参考

`batch_inference.py` 技术细节文档。

## 架构概览

```
main()
 ├─ 解析参数、展开 glob、解析 .list 文件
 ├─ --mode manual  →  _run_manual()  →  synthesize_entries()
 └─ --mode auto（默认）
     ├─ 加载 speaker_config.yaml
     ├─ 按说话人分组输入条目
     ├─ TextEmotionAnalyzer（只加载一次，所有说话人共用）
     │   ├─ 对每个说话人的参考数据集打分（ReferenceDatasetManager）
     │   └─ 对输入文本打分
     ├─ --dry_run  →  _write_dry_run_lists()  →  退出
     ├─ 加载 TTS 管线（使用第一个说话人的权重）
     └─ 对每个说话人：
         ├─ 如果权重不同，热切换 GPT/SoVITS 权重
         ├─ clear_prompt_cache()
         └─ synthesize_entries()
```

## 模式

### Auto 模式（默认）

需要 `--speaker_config`。对每条输入：

1. speaker 列将该行路由到正确的模型权重。
2. 如果第 5 列（`ref_audio`）存在，直接使用该音频——跳过情感打分。
3. 否则，`TextEmotionAnalyzer` 对输入文本打分，`ReferenceDatasetManager.find_best_reference()` 通过余弦相似度选择最匹配的参考音频。

TTS 管线只初始化一次（使用第一个说话人的权重）。后续说话人通过 `init_vits_weights()` 和 `init_t2s_weights()` 热切换权重。每次切换后需调用 `clear_prompt_cache()`。

### Manual 模式

需要 `--ref_audio_path`。所有行使用同一条参考音频。不做情感分析，不需要 speaker_config。模型权重来自 `--gpt_path` / `--sovits_path` 或 TTS 配置文件的默认值。

### Dry run（`--dry_run`）

仅限 auto 模式。运行情感打分和参考音频选择，然后将更新后的 `.list` 文件（每个输入文件对应一个输出文件）写入 `--output_dir`，选中的参考音频路径追加为第 5 列。跳过 TTS 管线加载和推理。

使用 `--dry_run_topk K` 可在第 5 列输出前 K 个候选参考音频（逗号分隔，最佳匹配排在最前）。这样你可以查看备选项并手动替换参考音频。默认值为 5。

输出文件可以直接作为下次运行的 `--input_list` 使用——当第 5 列包含逗号分隔的路径时，只使用第一个路径作为参考音频。

## .list 文件格式

输入列表和参考列表使用相同的管道符分隔格式：

```
第1列|第2列|第3列|第4列[|第5列]
```

| 列 | 输入列表含义 | 参考列表含义 |
|----|------------|------------|
| 1 | 输出音频路径（相对于 `--output_dir`） | 参考音频路径（相对于 `ref_audio_dir`） |
| 2 | 说话人名（匹配 `speaker_config`） | 说话人名 |
| 3 | 语言代码（`ja`、`en`、`zh`、`ko`、`all_ja` 等） | 语言代码 |
| 4 | 要合成的文本 | 参考音频的文字稿 |
| 5（可选） | 参考音频路径覆盖（top-K 时逗号分隔） | — |

当输入列表的第 5 列存在时，直接使用该参考音频。如果包含逗号分隔的路径（来自 `--dry_run_topk`），只使用第一个路径；你可以重新排列或删除条目来选择不同的参考音频。脚本会在说话人的参考数据集中查找匹配条目（按路径或文件名匹配），以获取 TTS 所需的文字稿和语言。如未找到匹配，文字稿为空，语言回退到该条目的语言。

## 情感分析

### TextEmotionAnalyzer

封装了 HuggingFace `AutoModelForSequenceClassification`。一次运行中只加载一个实例，所有说话人共用。

- `get_scores(text)` → `np.ndarray`，形状 `(num_labels,)`——logits 的 softmax。
- `get_scores_batch(texts)` → 分数向量列表（逐条处理，非张量级批处理）。
- `unload()` — 删除模型和分词器，释放 GPU 显存。

### 各语言模型

| 语言 | 模型 | 标签 |
|------|------|------|
| `ja` | `Mizuiro-sakura/luke-japanese-large-sentiment-analysis-wrime` | joy, sadness, anticipation, surprise, anger, fear, disgust, trust |
| `en` | `j-hartmann/emotion-english-distilroberta-base` | anger, disgust, fear, joy, neutral, sadness, surprise |
| `zh` | `Johnson8187/Chinese-Emotion-Small` | neutral, concerned, happy, angry, sad, questioning, surprised, disgusted |
| `ko` | `rkdaldus/ko-sent5-classification`（分词器：`monologg/kobert`） | anger, fear, happy, tender, sad |

模型加载顺序：
1. 本地路径（`GPT_SoVITS/pretrained_models/emotion-{lang}/`）— 如果目录存在
2. `--emotion_model` 覆盖
3. HuggingFace Hub 下载
4. HuggingFace 本地缓存（`local_files_only=True`）— 离线时的回退

### 情感匹配

`find_best_reference()` 计算输入分数向量与所有参考分数向量之间的余弦相似度，返回相似度最高的参考音频。这是软匹配（基于分数向量），而非硬标签匹配，因此混合情感的文本也能得到细致的匹配。

### ReferenceDatasetManager

管理每个说话人的参考音频数据集，带缓存：

- **首次运行**：扫描音频时长（3–10 秒过滤），对所有参考文本打分，将缓存保存为 `.emotion_cache.npz`（与参考列表文件同目录）。
- **后续运行**：如果列表文件的 MD5 哈希匹配，从缓存加载。缓存存储音频路径、文本、语言和分数向量。
- **缓存失效**：参考列表文件内容变化时自动失效（MD5 不匹配）。

缓存文件：`{ref_list_path}.emotion_cache.npz`

## 输出

### 音频转换

指定 `--output_sr` 或 `--output_channels` 时，通过 `ffmpeg` 转换（使用 `ffmpeg-python` 通过管道传递原始 PCM）。未指定时，`soundfile.write()` 直接写入。

### 输出路径解析

输入列表的第 1 列作为输出路径。如果不是绝对路径，则与 `--output_dir` 拼接。扩展名强制为 `.wav`。

## 所有命令行参数

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|------|------|--------|------|------|
| `--input_list` | str (nargs=+) | — | 是 | 输入 .list 文件。支持 glob 通配符（Windows 上自动展开）。 |
| `--output_dir` | str | — | 是 | 输出目录。 |
| `--mode` | `auto` \| `manual` | `auto` | 否 | auto：多说话人情感匹配。manual：单一参考音频。 |
| `--speaker_config` | str | `None` | auto 模式 | 说话人配置 YAML。 |
| `--emotion_model` | str | `None` | 否 | 覆盖默认情感模型（HuggingFace 模型 ID 或本地路径）。 |
| `--ref_audio_path` | str | `None` | manual 模式 | 参考音频路径。 |
| `--ref_text` | str | `""` | 否 | 参考音频文字稿（manual 模式）。 |
| `--ref_lang` | str | `ja` | 否 | 参考音频语言（manual 模式）。 |
| `--dry_run` | 标志 | `False` | 否 | 跳过推理，输出带有选定 ref_audio 的 .list 文件。 |
| `--dry_run_topk` | int | `5` | 否 | dry run 输出中包含的前 K 个候选参考音频数量（第 5 列逗号分隔）。 |
| `--tts_config` | str | `GPT_SoVITS/configs/tts_infer.yaml` | 否 | TTS 配置文件路径。 |
| `--gpt_path` | str | `None` | 否 | 覆盖 GPT 模型路径。 |
| `--sovits_path` | str | `None` | 否 | 覆盖 SoVITS 模型路径。 |
| `--version` | str | `None` | 否 | 模型版本：`v1`、`v2`、`v3`、`v4`、`v2Pro`、`v2ProPlus`。未设置时自动检测。 |
| `--output_sr` | int | `None` | 否 | 输出采样率（通过 ffmpeg 重采样）。 |
| `--output_channels` | int (1\|2) | `None` | 否 | 转换为单声道 (1) 或立体声 (2)（通过 ffmpeg）。 |
| `--top_k` | int | `5` | 否 | Top-K 采样。 |
| `--top_p` | float | `1.0` | 否 | Top-P（核采样）。 |
| `--temperature` | float | `1.0` | 否 | 采样温度。 |
| `--speed_factor` | float | `1.0` | 否 | 语速倍率。 |
| `--text_split_method` | str | `cut0` | 否 | 文本分割方法（传递给 TTS）。 |
| `--batch_size` | int | `1` | 否 | TTS 批量大小。 |
| `--seed` | int | `-1` | 否 | 随机种子（-1 = 随机）。 |
| `--parallel_infer` | bool | `True` | 否 | 并行推理（传递给 TTS）。 |
| `--repetition_penalty` | float | `1.35` | 否 | 重复惩罚（传递给 TTS）。 |

## 关键函数

| 函数 | 说明 |
|------|------|
| `parse_list_file(path, base_dir)` | 解析 `.list` 文件为条目字典列表。将 `base_dir` 拼接到相对音频路径前。 |
| `get_audio_duration(path)` | 获取音频时长（先用 soundfile，失败后用 librosa）。文件缺失或不可读时抛出异常。 |
| `load_speaker_config(path)` | 加载说话人配置 YAML。返回 `{说话人名: 配置字典}`，键名小写。 |
| `synthesize_entries(tts, entries, ...)` | 核心合成循环。遍历条目，选择参考音频，调用 `tts.run()`，转换并保存输出。 |
| `clear_prompt_cache(tts)` | 重置 TTS 提示缓存。切换说话人时必须调用。 |
| `_write_dry_run_lists(args, per_speaker)` | 将带有选定 ref_audio（第 5 列）的 .list 文件写入输出目录。每个输入文件对应一个输出文件。 |
| `_run_manual(args, input_entries)` | Manual 模式入口。加载 TTS，使用单一参考音频合成所有条目。 |

## 依赖

核心（必需）：
- `torch`、`numpy`、`soundfile`、`tqdm`、`pyyaml`
- `transformers` — 情感模型（仅 auto 模式）
- `ffmpeg-python` + `ffmpeg` 命令行工具 — 输出重采样/声道转换

回退：
- `librosa` — 音频时长检测的回退方案（Windows 上的 OGG 文件）
