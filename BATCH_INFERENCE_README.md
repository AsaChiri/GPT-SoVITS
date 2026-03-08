# Batch Voice Synthesis (batch_inference.py)

This script takes text lines and converts them to speech using your trained voice models. It automatically picks the best reference audio clip for each line based on the emotion of the text.

---

## What You Need Before Starting

1. **Trained voice models** — a GPT weight file (`.ckpt`) and a SoVITS weight file (`.pth`) for each speaker
2. **Reference audio clips** — short (3-10 second) voice recordings of each speaker, with transcripts
3. **A text file** listing what you want the voices to say

---

## Folder Setup

Place your files in the GPT-SoVITS folder like this:

```
GPT-SoVITS/                          (or GPT-SoVITS-v2pro-20250604/)
├── batch_inference.py
├── run_batch.bat                     ← double-click to run
│
├── GPT_weights_v2ProPlus/            ← put trained GPT weights here
│   ├── alisa-e15.ckpt
│   └── hiro-e15.ckpt
│
├── SoVITS_weights_v2ProPlus/         ← put trained SoVITS weights here
│   ├── alisa_e8_s456.pth
│   └── hiro_e8_s2184.pth
│
├── inputs/
│   ├── speaker_config.yaml           ← tells the script about each speaker
│   ├── alisa.list                    ← reference audio list for alisa
│   ├── hiro.list                     ← reference audio list for hiro
│   ├── alisa_audio/                  ← reference audio files for alisa
│   │   ├── clip001.wav
│   │   └── clip002.wav
│   ├── hiro_audio/                   ← reference audio files for hiro
│   │   ├── clip001.wav
│   │   └── clip002.wav
│   └── mod_input/                    ← text files to synthesize
│       └── dialogue.list
│
└── output/                           ← generated audio appears here
```

---

## File Formats

### Reference Audio List (e.g. `inputs/alisa.list`)

Each line describes one reference audio clip:

```
audio_path|speaker_name|language|transcript
```

Example:
```
alisa_audio/clip001.wav|alisa|ja|おはようございます、今日はいい天気ですね
alisa_audio/clip002.wav|alisa|ja|ありがとう、とても嬉しいです
```

- `audio_path` — path to the audio file, relative to the `inputs/` folder
- `speaker_name` — the speaker's name
- `language` — `ja` (Japanese), `en` (English), `zh` (Chinese), or `ko` (Korean)
- `transcript` — what the speaker says in the clip

**Important:** Reference audio clips must be **3 to 10 seconds** long. Shorter or longer clips are skipped.

### Input Text List (e.g. `inputs/mod_input/dialogue.list`)

Each line describes one voice line to generate:

```
output_path|speaker_name|language|text_to_speak
```

Example:
```
alisa/line001.wav|alisa|ja|今日の冒険は楽しかったね
alisa/line002.wav|alisa|ja|気をつけてね、危ないよ
hiro/line001.wav|hiro|ja|俺に任せろ、絶対に守ってやる
hiro/line002.wav|hiro|ja|まあ、悪くないな
```

- `output_path` — where to save the generated audio (relative to `output/`)
- `speaker_name` — must match a speaker in `speaker_config.yaml`
- `language` — language of the text
- `text_to_speak` — the text to convert to speech

### Speaker Config (inputs/speaker_config.yaml)

Tells the script which models and reference audio to use for each speaker:

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

- `gpt_path` / `sovits_path` — paths to the trained model files
- `ref_list` — path to the reference audio list file
- `ref_audio_dir` — the parent folder containing the audio files referenced in ref_list

---

## How to Run

### Option A: Double-click (bundled distribution)

1. Make sure your files are set up as shown above
2. Double-click **`run_batch.bat`**
3. A terminal window will open showing progress
4. When finished, check the `output/` folder

### Option B: Command line (bundled distribution)

Open a terminal in the GPT-SoVITS folder and run:

```
chcp 65001
runtime\python.exe batch_inference.py --input_list inputs/mod_input/dialogue.list --output_dir output/ --speaker_config inputs/speaker_config.yaml --output_sr 44100 --output_channels 2
```

### Option C: Development environment (uv)

```bash
chcp 65001
uv run batch_inference.py \
  --input_list inputs/mod_input/dialogue.list \
  --output_dir output/ \
  --speaker_config inputs/speaker_config.yaml \
  --output_sr 44100 \
  --output_channels 2
```

### Command Options

| Option | Default | What it does |
|--------|---------|-------------|
| `--input_list` | (required) | Input .list file(s) to synthesize |
| `--output_dir` | (required) | Folder for generated audio |
| `--speaker_config` | — | Speaker config YAML (multi-speaker) |
| `--mode` | `auto` | `auto` (emotion matching) or `manual` (single ref) |
| `--output_sr` | original | Resample output to this sample rate (e.g. `44100`) |
| `--output_channels` | original | `1` for mono, `2` for stereo |
| `--speed_factor` | `1.0` | Speech speed (0.5 = half speed, 2.0 = double speed) |
| `--temperature` | `1.0` | Randomness of speech (higher = more varied) |
| `--top_k` | `5` | Top-K sampling |
| `--top_p` | `1.0` | Top-P (nucleus) sampling |

---

## Complete Worked Example

Here's a minimal example with 2 speakers and 2 lines each.

### Step 1: Create the files

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

**inputs/alisa.list** (reference audio for alisa — you need the actual .wav files too)
```
alisa_audio/happy.wav|alisa|ja|今日はとても楽しいです
alisa_audio/sad.wav|alisa|ja|悲しいことがありました
alisa_audio/neutral.wav|alisa|ja|明日の天気はどうですか
```

**inputs/hiro.list** (reference audio for hiro)
```
hiro_audio/happy.wav|hiro|ja|よし、やったぞ
hiro_audio/angry.wav|hiro|ja|ふざけるな、許さないぞ
hiro_audio/neutral.wav|hiro|ja|そうだな、考えておこう
```

**inputs/mod_input/dialogue.list** (text to synthesize)
```
alisa/line001.wav|alisa|ja|今日の冒険は楽しかったね
alisa/line002.wav|alisa|ja|気をつけてね、危ないよ
hiro/line001.wav|hiro|ja|俺に任せろ、絶対に守ってやる
hiro/line002.wav|hiro|ja|まあ、悪くないな
```

### Step 2: Run

Double-click `run_batch.bat`, or run:

```
chcp 65001
runtime\python.exe batch_inference.py --input_list inputs/mod_input/dialogue.list --output_dir output/ --speaker_config inputs/speaker_config.yaml --output_sr 44100 --output_channels 2
```

### Step 3: Check results

Generated files will appear in:
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

## Troubleshooting

### `UnicodeEncodeError: 'gbk' codec can't encode character`

Windows terminals default to GBK encoding, which can't display some characters. Fix: run `chcp 65001` before running the script. The `run_batch.bat` launcher does this automatically.

### `.ogg` reference audio errors (`LibsndfileError`)

Windows libsndfile lacks OGG support. The script automatically falls back to librosa. If you still get errors, convert your reference audio to `.wav` format.

### `LookupError: averaged_perceptron_tagger_eng`

Needed for English text processing. Fix — run this once:
```
runtime\python.exe -c "import nltk; nltk.download('averaged_perceptron_tagger_eng')"
```

### Reference audio paths are wrong / files not found

If your ref list has paths like `alisa_audio/clip001.wav`, set `ref_audio_dir` to `inputs/` (the parent folder), **not** `inputs/alisa_audio/`. Otherwise the path doubles up to `inputs/alisa_audio/alisa_audio/clip001.wav`.

### `No reference audios passed the duration filter!`

All reference audio clips must be **3 to 10 seconds** long. Check your clips and trim any that are too short or too long.

### `No emotion model for language 'yue'`

Cantonese (yue) doesn't have an emotion model. Use `--mode manual` with a single reference audio instead of auto mode.

### Emotion models fail to download / no internet

For offline use (bundled distribution), run `download_emotion_models.py` once with internet access to save the models locally:
```
runtime\python.exe download_emotion_models.py
```
This saves models to `GPT_SoVITS/pretrained_models/emotion-{ja,en,zh,ko}/`. After that, no internet is needed.

---

# Developer Reference

Detailed technical documentation for `batch_inference.py`.

## Architecture Overview

```
main()
 ├─ parse args, expand globs, parse .list files
 ├─ if --mode manual  →  _run_manual()  →  synthesize_entries()
 └─ if --mode auto (default)
     ├─ load speaker_config.yaml
     ├─ group input entries by speaker
     ├─ TextEmotionAnalyzer (loaded once, shared across speakers)
     │   ├─ score each speaker's ref dataset  (ReferenceDatasetManager)
     │   └─ score input texts
     ├─ if --dry_run  →  _write_dry_run_lists()  →  exit
     ├─ load TTS pipeline once (first speaker's weights)
     └─ for each speaker:
         ├─ hot-swap GPT/SoVITS weights if different from current
         ├─ clear_prompt_cache()
         └─ synthesize_entries()
```

## Modes

### Auto mode (default)

Requires `--speaker_config`. For each input line:

1. The speaker column routes the line to the correct model weights.
2. If the 5th column (`ref_audio`) is present, that audio is used directly — emotion scoring is skipped for that line.
3. Otherwise, `TextEmotionAnalyzer` scores the input text and `ReferenceDatasetManager.find_best_reference()` picks the ref audio with the highest cosine similarity between emotion score vectors.

The TTS pipeline is initialized once with the first speaker's weights. For subsequent speakers, `init_vits_weights()` and `init_t2s_weights()` hot-swap only the changed weights. `clear_prompt_cache()` is called after every speaker switch.

### Manual mode

Requires `--ref_audio_path`. All lines use the same reference audio. No emotion analysis, no speaker config needed. Model weights come from `--gpt_path` / `--sovits_path` or the TTS config defaults.

### Dry run (`--dry_run`)

Auto mode only. Runs emotion scoring and reference selection, then writes updated `.list` files (one per input file) to `--output_dir` with the chosen ref audio path appended as the 5th column. Skips TTS pipeline loading and inference entirely.

The output files are directly usable as `--input_list` for a subsequent run — entries with the 5th column will use that ref audio without re-scoring.

## .list File Format

Both input and reference lists use the same pipe-delimited format:

```
column1|column2|column3|column4[|column5]
```

| Column | Input list meaning | Reference list meaning |
|--------|-------------------|----------------------|
| 1 | Output audio path (relative to `--output_dir`) | Reference audio path (relative to `ref_audio_dir`) |
| 2 | Speaker name (matched against `speaker_config`) | Speaker name |
| 3 | Language code (`ja`, `en`, `zh`, `ko`, `all_ja`, etc.) | Language code |
| 4 | Text to synthesize | Transcript of the reference audio |
| 5 (optional) | Path to reference audio override | — |

When column 5 is present in an input list, that ref audio is used directly. The script looks up the matching entry in the speaker's ref dataset (by path or basename) to retrieve the transcript and language for the TTS prompt. If no match is found, prompt text is empty and prompt language falls back to the entry's language.

## Emotion Analysis

### TextEmotionAnalyzer

Wraps a HuggingFace `AutoModelForSequenceClassification`. One instance is shared across all speakers in a run.

- `get_scores(text)` → `np.ndarray` of shape `(num_labels,)` — softmax over logits.
- `get_scores_batch(texts)` → list of score vectors (sequential, not batched at the tensor level).
- `unload()` — deletes model and tokenizer, frees GPU memory.

### Per-language models

| Language | Model | Labels |
|----------|-------|--------|
| `ja` | `Mizuiro-sakura/luke-japanese-large-sentiment-analysis-wrime` | joy, sadness, anticipation, surprise, anger, fear, disgust, trust |
| `en` | `j-hartmann/emotion-english-distilroberta-base` | anger, disgust, fear, joy, neutral, sadness, surprise |
| `zh` | `Johnson8187/Chinese-Emotion-Small` | neutral, concerned, happy, angry, sad, questioning, surprised, disgusted |
| `ko` | `rkdaldus/ko-sent5-classification` (tokenizer: `monologg/kobert`) | anger, fear, happy, tender, sad |

Model resolution order:
1. Local path (`GPT_SoVITS/pretrained_models/emotion-{lang}/`) — if the directory exists
2. `--emotion_model` override
3. HuggingFace hub download
4. HuggingFace local cache (`local_files_only=True`) — fallback when offline

### Emotion matching

`find_best_reference()` computes cosine similarity between the input's score vector and all reference score vectors, then returns the ref with the highest similarity. This is a soft match — not hard emotion labels — so mixed-emotion texts get nuanced matches.

### ReferenceDatasetManager

Manages a per-speaker reference audio dataset with caching:

- **First run**: scans audio durations (3–10s filter), scores all ref texts, saves `.emotion_cache.npz` next to the ref list file.
- **Subsequent runs**: loads from cache if the list file's MD5 hash matches. Cache stores audio paths, texts, languages, and score vectors.
- **Cache invalidation**: automatic when the ref list file content changes (MD5 mismatch).

Cache file: `{ref_list_path}.emotion_cache.npz`

## Output

### Audio conversion

When `--output_sr` or `--output_channels` is specified, conversion is done via `ffmpeg` (piping raw PCM through `ffmpeg-python`). When neither is specified, `soundfile.write()` writes directly.

### Output path resolution

The first column of the input list is used as the output path. If not absolute, it's joined with `--output_dir`. The extension is forced to `.wav`.

## All CLI Arguments

| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--input_list` | str (nargs=+) | — | yes | Input .list file(s). Glob patterns supported (expanded on Windows). |
| `--output_dir` | str | — | yes | Output directory for synthesized WAVs. |
| `--mode` | `auto` \| `manual` | `auto` | no | Auto: multi-speaker emotion-based ref selection. Manual: single ref. |
| `--speaker_config` | str | `None` | auto mode | YAML mapping speakers to model paths and ref data. |
| `--emotion_model` | str | `None` | no | Override the default emotion model (HuggingFace model ID or local path). |
| `--ref_audio_path` | str | `None` | manual mode | Reference audio path. |
| `--ref_text` | str | `""` | no | Reference audio transcript (manual mode). |
| `--ref_lang` | str | `ja` | no | Reference audio language (manual mode). |
| `--dry_run` | flag | `False` | no | Skip inference; write updated .list files with chosen ref_audio. |
| `--tts_config` | str | `GPT_SoVITS/configs/tts_infer.yaml` | no | TTS config YAML path. |
| `--gpt_path` | str | `None` | no | Override GPT model path (manual mode, or overrides speaker_config). |
| `--sovits_path` | str | `None` | no | Override SoVITS model path. |
| `--version` | str | `None` | no | Model version: `v1`, `v2`, `v3`, `v4`, `v2Pro`, `v2ProPlus`. Auto-detected if not set. |
| `--output_sr` | int | `None` | no | Resample output to this sample rate (via ffmpeg). |
| `--output_channels` | int (1\|2) | `None` | no | Convert to mono (1) or stereo (2) (via ffmpeg). |
| `--top_k` | int | `5` | no | Top-K sampling. |
| `--top_p` | float | `1.0` | no | Top-P (nucleus) sampling. |
| `--temperature` | float | `1.0` | no | Sampling temperature. |
| `--speed_factor` | float | `1.0` | no | Speech speed multiplier. |
| `--text_split_method` | str | `cut0` | no | Text splitting method (passed to TTS). |
| `--batch_size` | int | `1` | no | TTS batch size. |
| `--seed` | int | `-1` | no | Random seed (-1 = random). |
| `--parallel_infer` | bool | `True` | no | Parallel inference (passed to TTS). |
| `--repetition_penalty` | float | `1.35` | no | Repetition penalty (passed to TTS). |

## Key Functions

| Function | Description |
|----------|-------------|
| `parse_list_file(path, base_dir)` | Parse a `.list` file into a list of entry dicts. Prepends `base_dir` to relative audio paths. |
| `get_audio_duration(path)` | Get audio duration via soundfile, falling back to librosa. Raises on missing/unreadable files. |
| `load_speaker_config(path)` | Load speaker config YAML. Returns `{speaker_name: config_dict}`, keys lowercased. |
| `synthesize_entries(tts, entries, ...)` | Core synthesis loop. Iterates entries, picks ref audio, calls `tts.run()`, converts and saves output. |
| `clear_prompt_cache(tts)` | Reset TTS prompt cache. Required when switching speakers. |
| `_write_dry_run_lists(args, per_speaker)` | Write updated .list files with chosen ref_audio in column 5. One output file per input file. |
| `_run_manual(args, input_entries)` | Manual mode entry point. Loads TTS, synthesizes all entries with a single ref audio. |

## Dependencies

Core (always needed):
- `torch`, `numpy`, `soundfile`, `tqdm`, `pyyaml`
- `transformers` — for emotion models (auto mode only)
- `ffmpeg-python` + `ffmpeg` CLI — for output resampling/channel conversion

Fallback:
- `librosa` — fallback for audio duration detection (OGG files on Windows)
