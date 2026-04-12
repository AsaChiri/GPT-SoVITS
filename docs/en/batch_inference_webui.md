# Batch Inference WebUI

A Gradio front-end for `batch_inference.py`. Use it to configure batch runs, pick reference audio by ear after a dry-run, and kick off the full synthesis — without editing `.list` files or `.bat` files by hand.

## Launching

Double-click `go-batch-webui.bat` in the repo root, or run:

```
runtime\python.exe -I webui_batch_inference.py zh_CN   :: Chinese UI
runtime\python.exe -I webui_batch_inference.py en_US   :: English UI
```

The UI listens on port `9870` by default and auto-opens your browser.

## Tabs

### 1. Config

| Field | Notes |
|---|---|
| **Input list pattern** | Glob pattern for `.list` files — same string you'd pass to `--input_list`. Default: `inputs/mod_input/*.list`. Click **Preview matched files** to see exactly which files the pattern resolves to. |
| **Speaker config** | Dropdown populated from `inputs/*.yaml`. Click **View speakers** to expand the file in a table (speaker → GPT / SoVITS / ref list paths) without opening an editor. |
| **Output directory** | Where `.wav` output (full run) and updated `.list` files (dry-run) are written. |
| **Top-K reference candidates** | Number of candidates to keep per line during dry-run. Default 5. |
| **Mode** | `auto` = multi-speaker emotion-based ref selection via `--speaker_config`. `manual` = single reference audio for every line. Switching to `manual` reveals the ref audio picker / text / language inputs. |
| **Advanced inference params** | `top_k`, `top_p`, `temperature`, `speed_factor`, `batch_size`, `seed`, `text_split_method`, `output_sr`, `output_channels`. Seeded from `batch_inference.py` defaults. |

Two action buttons sit below the form:

- **Run dry-run** — launches `batch_inference.py --dry_run --dry_run_topk N ...` as a subprocess and streams its stdout/stderr into the **Live log** box. No audio is synthesized; the top-K candidates are written to column 5 of each output `.list`.
- **Run full inference** — same arguments, no `--dry_run`. Synthesizes `.wav` files to the output directory.

### 2. Review dry-run output

After a dry-run, switch here to audition the candidates.

1. Click **Rescan output/** and pick a `.list` file from the dropdown. `.curated.list` files are hidden so you don't re-review your own saves.
2. The UI shows 5 lines per page. Each line displays:
   - the target text, speaker, and language,
   - up to `Top-K` inline audio players (click to play each candidate in-browser),
   - a radio button to mark the best candidate.
3. Use **◀ Previous page** / **Next page ▶** to walk through all lines. Your radio selections are remembered as you page.
4. Click **Save curated list**. A new file `<name>.curated.list` is written next to the original with column 5 collapsed to a single path per line (the one you picked).
5. Click **Run full inference with curated list** to synthesize using the curated file immediately. Internally this re-runs `batch_inference.py` with `--input_list <curated>` and no `--dry_run`.

### 3. Help

Inline summary of the workflow above.

## Typical workflow

```
edit inputs/mod_input/*.list   (write the text you want synthesized)
  ↓
go-batch-webui.bat → Config tab → Run dry-run
  ↓
Review tab → listen to candidates → pick one per line → Save curated list
  ↓
Run full inference with curated list
  ↓
output/**/*.wav
```

## Troubleshooting

- **Unicode / GBK errors in the log box** — the launcher already sets `chcp 65001` and `PYTHONIOENCODING=utf-8`. If you run the Python file directly from another shell, set those yourself first.
- **NLTK tagger missing (English G2P)** — `python -m nltk.downloader averaged_perceptron_tagger_eng`.
- **Emotion model download fails on first run** — ensure you can reach Hugging Face, or pre-populate `GPT_SoVITS/pretrained_models/emotion-*`.
- **Dropdown empty under Speaker config** — the scanner only looks at `inputs/*.yaml` / `inputs/*.yml`. Place your YAML there or type a custom path (the dropdown is editable).
- **"No candidates" on a line** — the dry-run only writes rows that matched. Confirm your `speaker` column in the input list matches a key in `speaker_config.yaml` (case-sensitive, lowercase).
- **Port already in use** — edit `DEFAULT_PORT` in `webui_batch_inference.py` or free port `9870`.

## Related

- `batch_inference.py` — the underlying CLI the WebUI wraps. Run `python batch_inference.py --help` for the full argument list.
