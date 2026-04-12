"""Batch Inference WebUI.

A Gradio front-end for batch_inference.py. Lets the user:
  - Configure the common arguments via a form
  - Run dry-run (emotion-based reference selection) and watch live logs
  - Review the top-K candidates for each line and pick the best one by ear
  - Save the curated list and kick off the full inference

Launch via: go-batch-webui.bat   (or: python webui_batch_inference.py [locale])
"""

import glob
import os
import subprocess
import sys

import gradio as gr
import yaml

now_dir = os.getcwd()
sys.path.append(now_dir)

from tools.assets import css, js, top_html
from tools.i18n.i18n import I18nAuto, scan_language_list

language = sys.argv[-1] if len(sys.argv) > 1 and sys.argv[-1] in scan_language_list() else "Auto"
i18n = I18nAuto(language=language)

try:
    from config import webui_port_subfix
    DEFAULT_PORT = webui_port_subfix - 1  # 9870 by default, adjacent to the subfix tool
except Exception:
    DEFAULT_PORT = 9870

PAGE_SIZE = 5
MAX_K = 10

# ───────────────────── Review state (module globals) ─────────────────────
g_review_source = None   # currently loaded .list path
g_review_rows = []       # list[dict(audio_path, speaker, language, text, candidates, chosen_idx)]
g_review_page = 0        # current page index (0-based)


# ───────────────────── Subprocess runner ─────────────────────

def _stream_batch_inference(args_list):
    """Run batch_inference.py as a subprocess and yield accumulated stdout."""
    cmd = [sys.executable, "-u", "-I", "batch_inference.py", *args_list]
    header = "$ " + " ".join(cmd) + "\n\n"
    yield header

    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=now_dir,
            env=env,
            bufsize=1,
        )
    except Exception as e:
        yield header + f"[ERROR] Failed to start subprocess: {e}\n"
        return

    buf = header
    assert proc.stdout is not None
    for line in iter(proc.stdout.readline, ""):
        buf += line
        yield buf
    proc.stdout.close()
    proc.wait()
    yield buf + f"\n[exit code: {proc.returncode}]\n"


def _build_common_args(
    mode,
    input_pattern,
    output_dir,
    speaker_config,
    manual_ref,
    manual_ref_text,
    manual_ref_lang,
    top_k,
    top_p,
    temperature,
    speed_factor,
    batch_size,
    seed,
    text_split_method,
    output_sr,
    output_channels,
):
    args = [
        "--input_list", input_pattern,
        "--output_dir", output_dir,
        "--mode", mode,
        "--top_k", str(int(top_k)),
        "--top_p", str(float(top_p)),
        "--temperature", str(float(temperature)),
        "--speed_factor", str(float(speed_factor)),
        "--batch_size", str(int(batch_size)),
        "--seed", str(int(seed)),
        "--text_split_method", text_split_method,
    ]
    if mode == "auto":
        if speaker_config:
            args += ["--speaker_config", speaker_config]
    else:
        if manual_ref:
            args += ["--ref_audio_path", manual_ref]
        if manual_ref_text:
            args += ["--ref_text", manual_ref_text]
        if manual_ref_lang:
            args += ["--ref_lang", manual_ref_lang]
    if output_sr and int(output_sr) > 0:
        args += ["--output_sr", str(int(output_sr))]
    if output_channels and int(output_channels) in (1, 2):
        args += ["--output_channels", str(int(output_channels))]
    return args


def run_dry_run(
    mode, input_pattern, output_dir, speaker_config,
    manual_ref, manual_ref_text, manual_ref_lang,
    dry_run_topk,
    top_k, top_p, temperature, speed_factor,
    batch_size, seed, text_split_method,
    output_sr, output_channels,
):
    args = _build_common_args(
        mode, input_pattern, output_dir, speaker_config,
        manual_ref, manual_ref_text, manual_ref_lang,
        top_k, top_p, temperature, speed_factor,
        batch_size, seed, text_split_method,
        output_sr, output_channels,
    )
    args += ["--dry_run", "--dry_run_topk", str(int(dry_run_topk))]
    yield from _stream_batch_inference(args)


def run_full(
    mode, input_pattern, output_dir, speaker_config,
    manual_ref, manual_ref_text, manual_ref_lang,
    dry_run_topk,
    top_k, top_p, temperature, speed_factor,
    batch_size, seed, text_split_method,
    output_sr, output_channels,
):
    args = _build_common_args(
        mode, input_pattern, output_dir, speaker_config,
        manual_ref, manual_ref_text, manual_ref_lang,
        top_k, top_p, temperature, speed_factor,
        batch_size, seed, text_split_method,
        output_sr, output_channels,
    )
    yield from _stream_batch_inference(args)


# ───────────────────── Config tab helpers ─────────────────────

def preview_input_pattern(pattern):
    try:
        matches = sorted(glob.glob(pattern))
    except Exception as e:
        return [[f"[ERROR] {e}"]]
    if not matches:
        return [[i18n("No files matched")]]
    return [[m] for m in matches]


def scan_speaker_configs():
    candidates = sorted(glob.glob("inputs/*.yaml")) + sorted(glob.glob("inputs/*.yml"))
    return candidates


def refresh_speaker_configs():
    choices = scan_speaker_configs()
    default = "inputs/speaker_config.yaml" if "inputs/speaker_config.yaml" in choices else (choices[0] if choices else None)
    return gr.update(choices=choices, value=default)


def view_speaker_config(path):
    if not path or not os.path.exists(path):
        return [[i18n("File not found"), "", "", ""]]
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        return [[f"[ERROR] {e}", "", "", ""]]
    rows = []
    for name, cfg in (data.get("speakers") or {}).items():
        rows.append([
            name,
            str(cfg.get("gpt_path", "")),
            str(cfg.get("sovits_path", "")),
            str(cfg.get("ref_list", "")),
        ])
    return rows or [[i18n("No speakers defined"), "", "", ""]]


# ───────────────────── Review tab logic ─────────────────────

def scan_output_list_files(output_dir):
    if not output_dir or not os.path.isdir(output_dir):
        return []
    files = sorted(
        p for p in glob.glob(os.path.join(output_dir, "*.list"))
        if not p.endswith(".curated.list")
    )
    return files


def refresh_output_lists(output_dir):
    files = scan_output_list_files(output_dir)
    return gr.update(choices=files, value=(files[0] if files else None))


def load_list_file(path):
    """Parse a dry-run output .list and reset review state."""
    global g_review_source, g_review_rows, g_review_page
    g_review_source = path
    g_review_rows = []
    g_review_page = 0

    if not path or not os.path.exists(path):
        return _render_current_page() + [i18n("No file loaded")]

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|")
            if len(parts) < 5:
                continue
            audio_path, speaker, lang, text, ref_col = parts[0], parts[1], parts[2], parts[3], parts[4]
            candidates = [c.strip() for c in ref_col.split(",") if c.strip()]
            if not candidates:
                continue
            g_review_rows.append({
                "audio_path": audio_path,
                "speaker": speaker,
                "language": lang,
                "text": text,
                "candidates": candidates[:MAX_K],
                "chosen_idx": 0,
            })

    return _render_current_page() + [_status_text()]


def _status_text():
    if not g_review_rows:
        return i18n("No file loaded")
    total_pages = max(1, (len(g_review_rows) + PAGE_SIZE - 1) // PAGE_SIZE)
    return f"{i18n('Page')} {g_review_page + 1}/{total_pages} · {len(g_review_rows)} {i18n('lines')}"


def _render_current_page():
    """Return a flat list of gr.update(...) values for all review components on the page."""
    updates = []
    start = g_review_page * PAGE_SIZE
    for row_i in range(PAGE_SIZE):
        global_i = start + row_i
        if global_i < len(g_review_rows):
            row = g_review_rows[global_i]
            header_md = (
                f"### {i18n('Line')} {global_i + 1} · `{row['speaker']}` · `{row['language']}`\n\n"
                f"**{i18n('Text')}:** {row['text']}"
            )
            updates.append(gr.update(visible=True, value=header_md))
            for slot_i in range(MAX_K):
                if slot_i < len(row["candidates"]):
                    cand_path = row["candidates"][slot_i]
                    updates.append(gr.update(
                        visible=True,
                        value=cand_path if os.path.exists(cand_path) else None,
                        label=f"{i18n('Candidate')} {slot_i + 1}",
                    ))
                else:
                    updates.append(gr.update(visible=False, value=None))
            choices = [(f"{i18n('Candidate')} {i + 1}", i) for i in range(len(row["candidates"]))]
            updates.append(gr.update(visible=True, choices=choices, value=row["chosen_idx"]))
        else:
            updates.append(gr.update(visible=False))
            for _ in range(MAX_K):
                updates.append(gr.update(visible=False, value=None))
            updates.append(gr.update(visible=False))
    return updates


def go_prev_page():
    global g_review_page
    if g_review_page > 0:
        g_review_page -= 1
    return _render_current_page() + [_status_text()]


def go_next_page():
    global g_review_page
    total_pages = max(1, (len(g_review_rows) + PAGE_SIZE - 1) // PAGE_SIZE)
    if g_review_page + 1 < total_pages:
        g_review_page += 1
    return _render_current_page() + [_status_text()]


def _make_radio_handler(row_i):
    def handler(value):
        global_i = g_review_page * PAGE_SIZE + row_i
        if 0 <= global_i < len(g_review_rows) and value is not None:
            try:
                g_review_rows[global_i]["chosen_idx"] = int(value)
            except (TypeError, ValueError):
                pass
        return gr.update()
    return handler


def save_curated_list():
    if not g_review_source or not g_review_rows:
        return i18n("Nothing to save — load a dry-run list first.")
    base, _ = os.path.splitext(g_review_source)
    out_path = base + ".curated.list"
    with open(out_path, "w", encoding="utf-8") as f:
        for row in g_review_rows:
            idx = max(0, min(row["chosen_idx"], len(row["candidates"]) - 1))
            chosen = row["candidates"][idx]
            f.write(f"{row['audio_path']}|{row['speaker']}|{row['language']}|{row['text']}|{chosen}\n")
    return f"{i18n('Saved')}: {out_path}"


def run_with_curated(
    mode, output_dir, speaker_config,
    manual_ref, manual_ref_text, manual_ref_lang,
    top_k, top_p, temperature, speed_factor,
    batch_size, seed, text_split_method,
    output_sr, output_channels,
):
    """Run full inference using the most recently saved curated list (or save it first)."""
    if not g_review_source or not g_review_rows:
        yield i18n("Nothing to run — load a dry-run list first.")
        return
    save_curated_list()
    base, _ = os.path.splitext(g_review_source)
    curated = base + ".curated.list"
    args = _build_common_args(
        mode, curated, output_dir, speaker_config,
        manual_ref, manual_ref_text, manual_ref_lang,
        top_k, top_p, temperature, speed_factor,
        batch_size, seed, text_split_method,
        output_sr, output_channels,
    )
    yield from _stream_batch_inference(args)


# ───────────────────── UI ─────────────────────

def build_ui():
    initial_speaker_configs = scan_speaker_configs()
    default_speaker_config = (
        "inputs/speaker_config.yaml"
        if "inputs/speaker_config.yaml" in initial_speaker_configs
        else (initial_speaker_configs[0] if initial_speaker_configs else None)
    )

    with gr.Blocks(title="GPT-SoVITS Batch Inference WebUI", analytics_enabled=False, js=js, css=css) as app:
        gr.HTML(top_html.format(i18n("Batch Inference WebUI")))

        # ───── Shared config state (used by both Config and Review tabs) ─────
        with gr.Tabs():
            # ═════════════ Config tab ═════════════
            with gr.TabItem(i18n("Config")):
                with gr.Row():
                    with gr.Column(scale=3):
                        input_pattern = gr.Textbox(
                            label=i18n("Input list pattern"),
                            value="inputs/mod_input/*.list",
                            info=i18n("Glob pattern for .list files. Passed directly to batch_inference.py."),
                        )
                    with gr.Column(scale=1, min_width=160):
                        preview_btn = gr.Button(i18n("Preview matched files"))
                matched_df = gr.Dataframe(
                    headers=[i18n("Matched .list files")],
                    datatype=["str"],
                    row_count=(1, "dynamic"),
                    interactive=False,
                )
                preview_btn.click(preview_input_pattern, inputs=input_pattern, outputs=matched_df)

                with gr.Row():
                    with gr.Column(scale=3):
                        speaker_config = gr.Dropdown(
                            label=i18n("Speaker config"),
                            choices=initial_speaker_configs,
                            value=default_speaker_config,
                            allow_custom_value=True,
                            info=i18n("YAML mapping speakers to GPT/SoVITS weights and ref lists."),
                        )
                    with gr.Column(scale=1, min_width=160):
                        refresh_cfg_btn = gr.Button(i18n("Rescan inputs/"))
                        view_cfg_btn = gr.Button(i18n("View speakers"))
                speakers_df = gr.Dataframe(
                    headers=["speaker", "gpt_path", "sovits_path", "ref_list"],
                    datatype=["str", "str", "str", "str"],
                    row_count=(1, "dynamic"),
                    interactive=False,
                )
                refresh_cfg_btn.click(refresh_speaker_configs, outputs=speaker_config)
                view_cfg_btn.click(view_speaker_config, inputs=speaker_config, outputs=speakers_df)

                with gr.Row():
                    output_dir = gr.Textbox(label=i18n("Output directory"), value="output/")
                    dry_run_topk = gr.Slider(
                        label=i18n("Top-K reference candidates"),
                        minimum=1, maximum=MAX_K, step=1, value=5,
                    )

                mode = gr.Radio(
                    label=i18n("Mode"),
                    choices=[("Auto (multi-speaker)", "auto"), ("Manual (single ref)", "manual")],
                    value="auto",
                )

                with gr.Group(visible=False) as manual_group:
                    with gr.Row():
                        manual_ref = gr.Audio(
                            label=i18n("Reference audio (manual mode)"),
                            type="filepath",
                        )
                        with gr.Column():
                            manual_ref_text = gr.Textbox(label=i18n("Reference text"), value="")
                            manual_ref_lang = gr.Textbox(label=i18n("Reference language"), value="ja")

                def _toggle_manual(m):
                    return gr.update(visible=(m == "manual"))
                mode.change(_toggle_manual, inputs=mode, outputs=manual_group)

                with gr.Accordion(i18n("Advanced inference params"), open=False):
                    with gr.Row():
                        top_k = gr.Slider(label="top_k", minimum=1, maximum=50, step=1, value=5)
                        top_p = gr.Slider(label="top_p", minimum=0.0, maximum=1.0, step=0.05, value=1.0)
                        temperature = gr.Slider(label="temperature", minimum=0.0, maximum=2.0, step=0.05, value=1.0)
                        speed_factor = gr.Slider(label="speed_factor", minimum=0.5, maximum=2.0, step=0.05, value=1.0)
                    with gr.Row():
                        batch_size = gr.Slider(label="batch_size", minimum=1, maximum=32, step=1, value=1)
                        seed = gr.Number(label="seed", value=-1, precision=0)
                        text_split_method = gr.Textbox(label="text_split_method", value="cut0")
                    with gr.Row():
                        output_sr = gr.Number(label=i18n("Output sample rate (0 = keep)"), value=44100, precision=0)
                        output_channels = gr.Radio(
                            label=i18n("Output channels"),
                            choices=[("keep", 0), ("mono", 1), ("stereo", 2)],
                            value=2,
                        )

                with gr.Row():
                    dry_btn = gr.Button(i18n("Run dry-run"), variant="primary")
                    full_btn = gr.Button(i18n("Run full inference"), variant="primary")

                log_box = gr.Textbox(
                    label=i18n("Live log"),
                    lines=20,
                    max_lines=40,
                    interactive=False,
                    autoscroll=True,
                )

                common_inputs = [
                    mode, input_pattern, output_dir, speaker_config,
                    manual_ref, manual_ref_text, manual_ref_lang,
                    dry_run_topk,
                    top_k, top_p, temperature, speed_factor,
                    batch_size, seed, text_split_method,
                    output_sr, output_channels,
                ]
                dry_btn.click(run_dry_run, inputs=common_inputs, outputs=log_box)
                full_btn.click(run_full, inputs=common_inputs, outputs=log_box)

            # ═════════════ Review tab ═════════════
            with gr.TabItem(i18n("Review dry-run output")):
                with gr.Row():
                    with gr.Column(scale=3):
                        list_picker = gr.Dropdown(
                            label=i18n("Load list file"),
                            choices=scan_output_list_files("output/"),
                            value=None,
                            allow_custom_value=True,
                        )
                    with gr.Column(scale=1, min_width=160):
                        review_refresh_btn = gr.Button(i18n("Rescan output/"))

                status_md = gr.Markdown(i18n("No file loaded"))

                # Build PAGE_SIZE rows; each row has MAX_K audio slots + 1 radio + 1 header markdown.
                row_headers = []
                row_audios = []   # flat list of all audio components (PAGE_SIZE * MAX_K)
                row_radios = []

                for row_i in range(PAGE_SIZE):
                    header = gr.Markdown(visible=False)
                    row_headers.append(header)
                    with gr.Row():
                        audios_this_row = []
                        for slot_i in range(MAX_K):
                            a = gr.Audio(
                                label=f"Candidate {slot_i + 1}",
                                type="filepath",
                                interactive=False,
                                visible=False,
                                show_download_button=False,
                            )
                            audios_this_row.append(a)
                            row_audios.append(a)
                    radio = gr.Radio(
                        label=i18n("Pick best candidate"),
                        choices=[],
                        visible=False,
                        type="value",
                    )
                    row_radios.append(radio)
                    radio.change(_make_radio_handler(row_i), inputs=radio, outputs=radio)

                # Flat output list in the exact order _render_current_page builds updates:
                #   [header_0, audio_0_0 ... audio_0_{K-1}, radio_0, header_1, ...]
                review_outputs = []
                for row_i in range(PAGE_SIZE):
                    review_outputs.append(row_headers[row_i])
                    for slot_i in range(MAX_K):
                        review_outputs.append(row_audios[row_i * MAX_K + slot_i])
                    review_outputs.append(row_radios[row_i])

                with gr.Row():
                    prev_btn = gr.Button(i18n("◀ Previous page"))
                    next_btn = gr.Button(i18n("Next page ▶"))

                with gr.Row():
                    save_btn = gr.Button(i18n("Save curated list"), variant="primary")
                    run_curated_btn = gr.Button(i18n("Run full inference with curated list"), variant="primary")

                save_status = gr.Markdown("")
                review_log_box = gr.Textbox(
                    label=i18n("Live log"),
                    lines=15,
                    max_lines=40,
                    interactive=False,
                    autoscroll=True,
                )

                review_refresh_btn.click(
                    lambda od: refresh_output_lists(od),
                    inputs=output_dir,
                    outputs=list_picker,
                )
                list_picker.change(
                    load_list_file, inputs=list_picker,
                    outputs=review_outputs + [status_md],
                )
                prev_btn.click(go_prev_page, outputs=review_outputs + [status_md])
                next_btn.click(go_next_page, outputs=review_outputs + [status_md])
                save_btn.click(save_curated_list, outputs=save_status)

                curated_inputs = [
                    mode, output_dir, speaker_config,
                    manual_ref, manual_ref_text, manual_ref_lang,
                    top_k, top_p, temperature, speed_factor,
                    batch_size, seed, text_split_method,
                    output_sr, output_channels,
                ]
                run_curated_btn.click(run_with_curated, inputs=curated_inputs, outputs=review_log_box)

            # ═════════════ Help tab ═════════════
            with gr.TabItem(i18n("Help")):
                gr.Markdown(i18n(
                    "## Workflow\n\n"
                    "1. **Config tab** → set `Input list pattern` and `Speaker config`, pick a `Top-K`, click **Run dry-run**.\n"
                    "2. **Review tab** → click **Rescan output/**, pick a `.list` file, listen to each candidate, "
                    "choose the best with the radio button, then click **Save curated list**.\n"
                    "3. Click **Run full inference with curated list** (or return to the Config tab, point `Input list pattern` "
                    "at the `.curated.list` file, and click **Run full inference**).\n\n"
                    "Output WAVs land in the folder set as `Output directory`. "
                    "See `docs/en/batch_inference_webui.md` / `docs/cn/批量推理WebUI.md` for full docs."
                ))

    return app


if __name__ == "__main__":
    os.environ.setdefault("no_proxy", "localhost, 127.0.0.1, ::1")
    app = build_ui()
    app.queue().launch(
        server_name="0.0.0.0",
        server_port=DEFAULT_PORT,
        inbrowser=True,
        share=False,
        show_error=True,
    )
