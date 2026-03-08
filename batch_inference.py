"""
Batch inference with emotion-based reference audio selection.

Supports two modes:
  - auto (default): Multi-speaker. Requires --speaker_config YAML mapping speakers to
    model weights and ref data. Each input line is routed to the correct speaker, and the
    best reference audio is selected per line based on emotion similarity. The TTS pipeline
    loads once and hot-swaps weights per speaker.
  - manual: Use a single reference audio for all lines.

Uses the modern TTS class from GPT_SoVITS/TTS_infer_pack/TTS.py.
"""

import argparse
import gc
import os
import re
import sys
import time
import traceback
from collections import Counter

import ffmpeg
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append(os.path.join(now_dir, "GPT_SoVITS"))

# ───────────────────── Per-language emotion model registry ─────────────────────
EMOTION_MODEL_REGISTRY = {
    "ja": {
        "model_id": "Mizuiro-sakura/luke-japanese-large-sentiment-analysis-wrime",
        "local_path": "GPT_SoVITS/pretrained_models/emotion-ja",
        "labels": ["joy", "sadness", "anticipation", "surprise", "anger", "fear", "disgust", "trust"],
    },
    "en": {
        "model_id": "j-hartmann/emotion-english-distilroberta-base",
        "local_path": "GPT_SoVITS/pretrained_models/emotion-en",
        "labels": ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"],
    },
    "zh": {
        "model_id": "Johnson8187/Chinese-Emotion-Small",
        "local_path": "GPT_SoVITS/pretrained_models/emotion-zh",
        "labels": ["neutral", "concerned", "happy", "angry", "sad", "questioning", "surprised", "disgusted"],
    },
    "ko": {
        "model_id": "rkdaldus/ko-sent5-classification",
        "local_path": "GPT_SoVITS/pretrained_models/emotion-ko",
        "labels": ["anger", "fear", "happy", "tender", "sad"],
        "tokenizer_id": "monologg/kobert",
        "trust_remote_code": True,
    },
}


# ───────────────────── Utilities ─────────────────────

def sanitize_filename(s: str, max_len: int = 50) -> str:
    s = re.sub(r'[\\/:*?"<>|\r\n]+', '_', s)
    return s[:max_len].rstrip('_. ')


def parse_list_file(path: str, base_dir: str = None):
    """Parse a .list file. Returns list of dicts with keys:
       audio_path, speaker, language, text, ref_audio (optional)."""
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 4:
                print(f"[WARN] Skipping line {lineno}: expected at least 4 columns, got {len(parts)}")
                continue
            audio_path = parts[0]
            if base_dir and not os.path.isabs(audio_path):
                audio_path = os.path.join(base_dir, audio_path)
            entry = {
                "audio_path": audio_path,
                "speaker": parts[1],
                "language": parts[2].strip().lower(),
                "text": parts[3],
                "ref_audio": parts[4].split(",")[0].strip() if len(parts) > 4 and parts[4].strip() else None,
            }
            entries.append(entry)
    return entries


def get_audio_duration(path: str) -> float:
    """Get audio duration in seconds. Raises if the file cannot be read."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Audio file not found: {path}")
    try:
        info = sf.info(path)
        return info.duration
    except Exception:
        pass
    try:
        import librosa
        return librosa.get_duration(path=path)
    except Exception:
        raise OSError(f"Could not read audio duration for: {path}  "
                      "(file exists but neither soundfile nor librosa could read it)")


def load_speaker_config(path: str) -> dict:
    """Load speaker config YAML. Returns dict of speaker_key -> config."""
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    speakers = data.get("speakers", data)
    # Normalize keys to lowercase
    return {k.lower(): v for k, v in speakers.items()}


# ───────────────────── TextEmotionAnalyzer ─────────────────────

class TextEmotionAnalyzer:
    """Get emotion score vectors using a language-specific emotion model."""

    def __init__(self, lang: str, model_override: str = None, device: str = None):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        if lang not in EMOTION_MODEL_REGISTRY and not model_override:
            raise ValueError(f"No emotion model for language '{lang}'. "
                             f"Supported: {list(EMOTION_MODEL_REGISTRY.keys())}. "
                             f"Use --emotion_model to provide a custom model.")
        cfg = EMOTION_MODEL_REGISTRY.get(lang, {})
        self.labels = cfg.get("labels", [])
        trust = cfg.get("trust_remote_code", False)

        # Prefer local pre-downloaded models; fall back to HuggingFace download
        local_path = cfg.get("local_path", "")
        if local_path and os.path.isdir(local_path):
            model_id = local_path
            tokenizer_id = local_path
        else:
            model_id = model_override or cfg["model_id"]
            tokenizer_id = cfg.get("tokenizer_id", model_id)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        print(f"[TextEmotionAnalyzer] Loading {model_id} for '{lang}' on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=trust)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device)
        self.model.eval()
        # Auto-detect labels from model config if not in registry
        if not self.labels:
            id2label = self.model.config.id2label
            self.labels = [id2label[i] for i in range(len(id2label))]
        print(f"[TextEmotionAnalyzer] Ready. Labels: {self.labels}")

    @torch.no_grad()
    def get_scores(self, text: str) -> np.ndarray:
        """Return emotion score vector (softmax over logits)."""
        tokens = self.tokenizer(text, truncation=True, max_length=512,
                                padding=True, return_tensors="pt")
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        logits = self.model(**tokens).logits[0]
        return torch.softmax(logits, dim=-1).cpu().numpy()

    def get_scores_batch(self, texts: list) -> list:
        """Returns list of score vectors."""
        return [self.get_scores(t) for t in texts]

    def get_dominant(self, text: str) -> str:
        scores = self.get_scores(text)
        return self.labels[int(np.argmax(scores))]

    def unload(self):
        del self.model, self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ───────────────────── ReferenceDatasetManager ─────────────────────

class ReferenceDatasetManager:
    """Manage a dataset of reference audios with text-based emotion score vectors.

    On first run: filters refs by audio duration, scores texts, saves cache.
    On subsequent runs with unchanged list file: restores everything from cache,
    completely skipping the expensive duration scan and emotion scoring.
    """

    def __init__(self, ref_entries: list, cache_path: str = None,
                 list_file_path: str = None,
                 min_duration: float = 3.0, max_duration: float = 10.0):
        self._raw_entries = ref_entries
        self.cache_path = cache_path
        self.list_file_path = list_file_path
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.ref_entries = None   # set by cache load or _filter_by_duration
        self.score_vectors = None  # np.ndarray shape (N, num_labels)

    def _compute_list_hash(self) -> str:
        """MD5 hash of the list file content for cache invalidation."""
        if not self.list_file_path:
            return ""
        import hashlib
        with open(self.list_file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def _filter_by_duration(self):
        """Filter raw entries by audio duration. Expensive (reads every file)."""
        filtered = []
        errors = []
        for entry in self._raw_entries:
            try:
                dur = get_audio_duration(entry["audio_path"])
            except (FileNotFoundError, OSError) as e:
                errors.append(str(e))
                continue
            if self.min_duration <= dur <= self.max_duration:
                entry["duration"] = dur
                filtered.append(entry)
            else:
                print(f"[RefManager] Skipping {entry['audio_path']} (duration={dur:.1f}s, need {self.min_duration}-{self.max_duration}s)")
        if errors:
            print(f"[ERROR] {len(errors)} reference audio(s) could not be read:")
            for err in errors:
                print(f"  - {err}")
        self.ref_entries = filtered
        if not self.ref_entries:
            raise ValueError("No reference audios passed the duration filter!")
        print(f"[RefManager] {len(self.ref_entries)} reference audios after duration filtering.")

    def extract_emotions(self, analyzer: TextEmotionAnalyzer):
        """Score all reference texts. Tries cache first; filters+scores on miss."""
        if self._try_load_cache():
            return

        # Cache miss — do the expensive duration scan + emotion scoring
        self._filter_by_duration()

        print("[RefManager] Scoring reference texts...")
        vectors = []
        for entry in tqdm(self.ref_entries, desc="Scoring ref texts"):
            scores = analyzer.get_scores(entry["text"])
            vectors.append(scores)
        self.score_vectors = np.array(vectors, dtype=np.float32)

        self._save_cache()

    def _try_load_cache(self) -> bool:
        if not self.cache_path or not os.path.exists(self.cache_path):
            return False
        try:
            data = np.load(self.cache_path, allow_pickle=True)
            # Validate by list file hash
            if "list_hash" in data:
                current_hash = self._compute_list_hash()
                if current_hash and str(data["list_hash"]) != current_hash:
                    print("[RefManager] Cache invalid (list file changed), recomputing...")
                    return False
            for key in ("audio_paths", "texts", "languages", "score_vectors"):
                if key not in data:
                    print(f"[RefManager] Cache missing '{key}', recomputing...")
                    return False
            # Restore ref_entries from cache (skip duration scan entirely)
            cached_paths = list(data["audio_paths"])
            cached_texts = list(data["texts"])
            cached_langs = list(data["languages"])
            self.ref_entries = [
                {"audio_path": str(p), "text": str(t), "language": str(l)}
                for p, t, l in zip(cached_paths, cached_texts, cached_langs)
            ]
            self.score_vectors = data["score_vectors"]
            print(f"[RefManager] Loaded cache ({len(self.ref_entries)} refs from {self.cache_path})")
            return True
        except Exception as e:
            print(f"[RefManager] Failed to load cache: {e}")
        return False

    def _save_cache(self):
        if not self.cache_path:
            return
        try:
            audio_paths = np.array([e["audio_path"] for e in self.ref_entries], dtype=object)
            texts = np.array([e["text"] for e in self.ref_entries], dtype=object)
            languages = np.array([e["language"] for e in self.ref_entries], dtype=object)
            list_hash = np.array(self._compute_list_hash())
            np.savez(self.cache_path, audio_paths=audio_paths, texts=texts,
                     languages=languages, score_vectors=self.score_vectors,
                     list_hash=list_hash)
            print(f"[RefManager] Saved emotion cache to {self.cache_path}")
        except Exception as e:
            print(f"[WARN] Failed to save cache: {e}")

    def find_entry_by_audio(self, audio_path: str):
        """Look up a ref entry by audio path (exact or basename match)."""
        basename = os.path.basename(audio_path)
        for entry in self.ref_entries:
            if entry["audio_path"] == audio_path or os.path.basename(entry["audio_path"]) == basename:
                return entry
        return None

    def find_topk_references(self, input_scores: np.ndarray, k: int = 1) -> list:
        """Find the top-k refs with highest cosine similarity to input_scores."""
        if self.score_vectors is None:
            raise RuntimeError("Call extract_emotions() first")

        input_norm = input_scores / (np.linalg.norm(input_scores) + 1e-8)
        ref_norms = self.score_vectors / (np.linalg.norm(self.score_vectors, axis=1, keepdims=True) + 1e-8)
        similarities = ref_norms @ input_norm
        k = min(k, len(self.ref_entries))
        top_indices = np.argsort(similarities)[::-1][:k]
        return [self.ref_entries[int(idx)] for idx in top_indices]

    def find_best_reference(self, input_scores: np.ndarray) -> dict:
        """Find the ref with highest cosine similarity to input_scores."""
        return self.find_topk_references(input_scores, k=1)[0]

    def unload(self):
        self.score_vectors = None
        gc.collect()


# ───────────────────── CLI ─────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch TTS inference with emotion-based reference audio selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example speaker_config.yaml:
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
""",
    )

    # Required
    parser.add_argument("--input_list", required=True, nargs="+",
                        help="Input .list file(s) (vocal_path|speaker|lang|text[|ref_audio])")
    parser.add_argument("--output_dir", required=True, help="Output directory for synthesized WAVs")

    # Mode
    parser.add_argument("--mode", choices=["auto", "manual"], default="auto",
                        help="auto: multi-speaker emotion-based ref selection; manual: single ref (default: auto)")

    # Auto mode (multi-speaker)
    parser.add_argument("--speaker_config", default=None,
                        help="YAML file mapping speakers to model paths and ref data (required for auto mode)")
    parser.add_argument("--emotion_model", default=None,
                        help="Override the default emotion model for the detected language")

    # Manual mode
    parser.add_argument("--ref_audio_path", help="Reference audio path (for manual mode)")
    parser.add_argument("--ref_text", default="", help="Reference audio transcript")
    parser.add_argument("--ref_lang", default="ja", help="Reference audio language")

    # Dry run — score emotions and pick refs, then write updated list file and exit
    parser.add_argument("--dry_run", action="store_true",
                        help="Skip inference; write updated .list with chosen ref_audio in 5th column")
    parser.add_argument("--dry_run_topk", type=int, default=5,
                        help="Number of top reference candidates to include in dry run output (comma-separated in column 5)")

    # TTS config
    parser.add_argument("--tts_config", default="GPT_SoVITS/configs/tts_infer.yaml", help="TTS config YAML")
    parser.add_argument("--gpt_path", default=None, help="Override GPT model path")
    parser.add_argument("--sovits_path", default=None, help="Override SoVITS model path")
    parser.add_argument("--version", default=None,
                        choices=["v1", "v2", "v3", "v4", "v2Pro", "v2ProPlus"],
                        help="Model version (auto-detected from weights if not set)")

    # Output format
    parser.add_argument("--output_sr", type=int, default=None,
                        help="Resample output to this sample rate (default: keep original)")
    parser.add_argument("--output_channels", type=int, default=None, choices=[1, 2],
                        help="Convert output to mono (1) or stereo (2) (default: keep original)")

    # Inference params
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--speed_factor", type=float, default=1.0)
    parser.add_argument("--text_split_method", default="cut0", help="Text splitting method")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--parallel_infer", type=bool, default=True)
    parser.add_argument("--repetition_penalty", type=float, default=1.35)

    return parser


# ───────────────────── Synthesis helpers ─────────────────────

def synthesize_entries(tts, entries, ref_manager, input_scores, input_emotions,
                       args, mode="auto", global_offset=0):
    """Synthesize a batch of entries. Returns (succeeded, failed) counts."""
    failed = 0
    for i, entry in enumerate(tqdm(entries, desc="Synthesizing")):
        text = entry["text"]
        text_lang = entry["language"]
        emotion = input_emotions[i] if (input_emotions and input_emotions[i]) else ""

        # Determine reference audio
        if mode == "manual":
            ref_audio = args.ref_audio_path
            ref_text = args.ref_text
            ref_lang = args.ref_lang.lower()
        elif entry.get("ref_audio"):
            # Explicit ref audio override from input list 5th column
            ref_audio = entry["ref_audio"]
            ref_entry = ref_manager.find_entry_by_audio(ref_audio) if ref_manager else None
            ref_text = ref_entry["text"] if ref_entry else ""
            ref_lang = ref_entry["language"] if ref_entry else entry["language"]
        else:
            ref_entry = ref_manager.find_best_reference(input_scores[i])
            ref_audio = ref_entry["audio_path"]
            ref_text = ref_entry["text"]
            ref_lang = ref_entry["language"]

        text_preview = text[:40] + "..." if len(text) > 40 else text
        global_i = global_offset + i
        emo_tag = f" emotion={emotion} |" if emotion else ""
        print(f"\n[{global_i+1}]{emo_tag} ref={os.path.basename(ref_audio)} | {text_preview}")

        try:
            inputs_dict = {
                "text": text,
                "text_lang": text_lang,
                "ref_audio_path": ref_audio,
                "prompt_text": ref_text,
                "prompt_lang": ref_lang,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "temperature": args.temperature,
                "text_split_method": args.text_split_method,
                "batch_size": args.batch_size,
                "speed_factor": args.speed_factor,
                "seed": args.seed,
                "parallel_infer": args.parallel_infer,
                "repetition_penalty": args.repetition_penalty,
                "return_fragment": False,
            }

            # TTS.run() is a generator
            sr, audio_data = next(tts.run(inputs_dict))

            # Save output — use first column as output path
            out_path = entry["audio_path"]
            if not os.path.isabs(out_path):
                out_path = os.path.join(args.output_dir, out_path)
            # Ensure .wav extension
            base, ext = os.path.splitext(out_path)
            if ext.lower() != ".wav":
                out_path = base + ".wav"
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

            # Resample / channel conversion via ffmpeg if needed
            need_ffmpeg = ((args.output_sr and args.output_sr != sr)
                           or args.output_channels)
            if need_ffmpeg:
                raw_audio = audio_data.astype(np.int16).tobytes()
                is_mono = audio_data.ndim == 1
                input_ac = 1 if is_mono else audio_data.shape[-1]
                stream = ffmpeg.input(
                    "pipe:", format="s16le", acodec="pcm_s16le",
                    ar=str(sr), ac=input_ac)
                out_kwargs = {}
                if args.output_sr:
                    out_kwargs["ar"] = str(args.output_sr)
                if args.output_channels:
                    out_kwargs["ac"] = str(args.output_channels)
                out_bytes, _ = (
                    stream.output(out_path, **out_kwargs)
                    .overwrite_output()
                    .run(input=raw_audio, capture_stdout=True, capture_stderr=True)
                )
            else:
                sf.write(out_path, audio_data, sr)

        except Exception as e:
            print(f"[ERROR] Failed on line {global_i+1}: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1
            continue

    return len(entries) - failed, failed


def clear_prompt_cache(tts):
    """Reset TTS prompt cache (needed when switching speakers)."""
    tts.prompt_cache = {
        "ref_audio_path": None,
        "prompt_semantic": None,
        "refer_spec": [],
        "prompt_text": None,
        "prompt_lang": None,
        "phones": None,
        "bert_features": None,
        "norm_text": None,
        "aux_ref_audio_paths": [],
    }


# ───────────────────── Main ─────────────────────

def main():
    parser = build_parser()
    args = parser.parse_args()

    # Validate args
    if args.mode == "auto" and not args.speaker_config:
        parser.error("--speaker_config is required for auto mode")
    if args.mode == "manual" and not args.ref_audio_path:
        parser.error("--ref_audio_path is required for manual mode")

    os.makedirs(args.output_dir, exist_ok=True)

    # Validate manual ref audio duration
    if args.mode == "manual":
        dur = get_audio_duration(args.ref_audio_path)
        if dur < 3.0 or dur > 10.0:
            print(f"[ERROR] Reference audio duration is {dur:.1f}s, must be 3-10s.")
            return

    # Parse all input lists (expand globs for Windows cmd.exe compatibility)
    import glob as _glob
    expanded_lists = []
    for pattern in args.input_list:
        matched = sorted(_glob.glob(pattern))
        if matched:
            expanded_lists.extend(matched)
        else:
            print(f"[ERROR] No files matched: {pattern}")
            return
    if not expanded_lists:
        print("[ERROR] No input list files found!")
        return
    input_entries = []
    for list_path in expanded_lists:
        entries = parse_list_file(list_path)
        for entry in entries:
            entry["_source_list"] = list_path
        input_entries.extend(entries)
    if not input_entries:
        print("[ERROR] No valid entries in input list(s)!")
        return
    print(f"[INFO] {len(input_entries)} lines to synthesize.")

    # ──────── Route: manual vs auto ────────
    if args.mode == "manual":
        _run_manual(args, input_entries)
        return

    # Load speaker config and group entries by speaker
    speaker_configs = load_speaker_config(args.speaker_config)

    speaker_groups = {}
    unknown_speakers = set()
    for entry in input_entries:
        speaker = entry["speaker"].lower()
        if speaker not in speaker_configs:
            unknown_speakers.add(speaker)
            continue
        speaker_groups.setdefault(speaker, []).append(entry)

    if unknown_speakers:
        print(f"[WARN] Speakers not in config (skipped): {sorted(unknown_speakers)}")
    if not speaker_groups:
        print("[ERROR] No input entries matched any speaker in config!")
        return

    total_lines = sum(len(v) for v in speaker_groups.values())
    print(f"[INFO] {len(speaker_groups)} speakers, {total_lines} lines total.")
    for spk, entries in speaker_groups.items():
        print(f"  {spk}: {len(entries)} lines")

    # Determine language for emotion analysis (from first entry)
    first_entry = next(iter(speaker_groups.values()))[0]
    emotion_lang = first_entry["language"].replace("all_", "").replace("auto", "")

    # Load emotion analyzer once
    analyzer = TextEmotionAnalyzer(lang=emotion_lang, model_override=args.emotion_model)

    # Pre-score all ref datasets and input texts
    per_speaker = {}  # speaker -> {ref_manager, input_scores, input_emotions, entries}
    for speaker, entries in speaker_groups.items():
        cfg = speaker_configs[speaker]
        ref_list = cfg["ref_list"]
        ref_audio_dir = cfg.get("ref_audio_dir", None)

        print(f"\n[INFO] Preparing {speaker} ({len(entries)} lines)...")
        ref_entries = parse_list_file(ref_list, base_dir=ref_audio_dir)
        if not ref_entries:
            print(f"[WARN] No ref entries for {speaker}, skipping!")
            continue

        cache_path = ref_list + ".emotion_cache.npz"
        ref_manager = ReferenceDatasetManager(
            ref_entries, cache_path=cache_path, list_file_path=ref_list)
        ref_manager.extract_emotions(analyzer)

        # Score input texts (skip entries with explicit ref_audio override)
        input_scores = [None] * len(entries)
        input_emotions = [None] * len(entries)
        texts_to_score = []
        score_indices = []
        for i, entry in enumerate(entries):
            if entry.get("ref_audio") and not args.dry_run:
                continue  # explicit ref audio, no scoring needed
            texts_to_score.append(entry["text"])
            score_indices.append(i)

        if texts_to_score:
            batch_scores = analyzer.get_scores_batch(texts_to_score)
            for idx, scores in zip(score_indices, batch_scores):
                input_scores[idx] = scores

        for i, scores in enumerate(input_scores):
            if scores is not None:
                input_emotions[i] = analyzer.labels[int(np.argmax(scores))]

        print(f"  Emotion distribution: {dict(Counter(input_emotions).most_common())}")

        per_speaker[speaker] = {
            "ref_manager": ref_manager,
            "input_scores": input_scores,
            "input_emotions": input_emotions,
            "entries": entries,
        }

    # Free emotion analyzer
    analyzer.unload()

    # Dry run: write updated list files with chosen ref_audio and exit
    if args.dry_run:
        _write_dry_run_lists(args, per_speaker)
        return

    # Initialize TTS pipeline once (with first speaker's weights)
    print(f"\n[INFO] Loading TTS pipeline...")
    from TTS_infer_pack.TTS import TTS, TTS_Config

    first_speaker = next(iter(per_speaker))
    first_cfg = speaker_configs[first_speaker]
    tts_config = TTS_Config(args.tts_config)
    if args.version:
        tts_config.version = args.version
    tts_config.t2s_weights_path = first_cfg["gpt_path"]
    tts_config.vits_weights_path = first_cfg["sovits_path"]
    tts = TTS(tts_config)
    current_gpt = first_cfg["gpt_path"]
    current_sovits = first_cfg["sovits_path"]
    print("[INFO] TTS pipeline ready.")

    # Synthesize per speaker, swapping weights as needed
    start_time = time.time()
    total_ok = 0
    total_failed = 0
    global_offset = 0

    for speaker_idx, (speaker, data) in enumerate(per_speaker.items()):
        cfg = speaker_configs[speaker]
        gpt_path = cfg["gpt_path"]
        sovits_path = cfg["sovits_path"]

        print(f"\n{'='*60}")
        print(f"[{speaker_idx+1}/{len(per_speaker)}] Speaker: {speaker} ({len(data['entries'])} lines)")

        # Swap weights if needed
        if sovits_path != current_sovits:
            tts.init_vits_weights(sovits_path)
            current_sovits = sovits_path
        if gpt_path != current_gpt:
            tts.init_t2s_weights(gpt_path)
            current_gpt = gpt_path
        clear_prompt_cache(tts)

        ok, failed = synthesize_entries(
            tts, data["entries"], data["ref_manager"],
            data["input_scores"], data["input_emotions"],
            args, global_offset=global_offset,
        )
        total_ok += ok
        total_failed += failed
        global_offset += len(data["entries"])
        print(f"  -> {speaker}: {ok}/{len(data['entries'])} succeeded")

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"All done! {total_ok}/{total_ok+total_failed} succeeded, {total_failed} failed.")
    print(f"Time elapsed: {elapsed:.1f}s ({elapsed/max(total_ok+total_failed,1):.1f}s per line)")
    print(f"Output directory: {args.output_dir}")


def _write_dry_run_lists(args, per_speaker):
    """Write updated .list files with chosen ref_audio in the 5th column.
    One output file per original input list, placed in output_dir."""
    os.makedirs(args.output_dir, exist_ok=True)

    # Collect all entries with their chosen ref_audio, grouped by source file
    topk = getattr(args, "dry_run_topk", 1) or 1
    by_source = {}  # source_list_path -> list of output lines
    for speaker, data in per_speaker.items():
        ref_manager = data["ref_manager"]
        input_scores = data["input_scores"]
        for i, entry in enumerate(data["entries"]):
            if topk > 1:
                top_refs = ref_manager.find_topk_references(input_scores[i], k=topk)
                ref_audio_col = ",".join(r["audio_path"] for r in top_refs)
            else:
                ref_entry = ref_manager.find_best_reference(input_scores[i])
                ref_audio_col = ref_entry["audio_path"]
            line = f"{entry['audio_path']}|{entry['speaker']}|{entry['language']}|{entry['text']}|{ref_audio_col}"
            source = entry.get("_source_list", "unknown")
            by_source.setdefault(source, []).append(line)

    total = 0
    for source_path, lines in by_source.items():
        out_name = os.path.basename(source_path)
        out_path = os.path.join(args.output_dir, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
        total += len(lines)
        print(f"[INFO] Wrote {len(lines)} lines to {out_path}")
    print(f"[INFO] Dry run complete. {total} lines across {len(by_source)} file(s).")


def _run_manual(args, input_entries):
    """Manual mode: single reference audio for all lines."""
    print("[INFO] Loading TTS pipeline...")
    from TTS_infer_pack.TTS import TTS, TTS_Config

    tts_config = TTS_Config(args.tts_config)
    if args.version:
        tts_config.version = args.version
    if args.gpt_path:
        tts_config.t2s_weights_path = args.gpt_path
    if args.sovits_path:
        tts_config.vits_weights_path = args.sovits_path

    tts = TTS(tts_config)
    print("[INFO] TTS pipeline ready.")

    input_emotions = [None] * len(input_entries)

    start_time = time.time()
    ok, failed = synthesize_entries(
        tts, input_entries, None, None, input_emotions,
        args, mode="manual",
    )

    elapsed = time.time() - start_time
    total = len(input_entries)
    print(f"\n{'='*60}")
    print(f"Done! {ok}/{total} succeeded, {failed} failed.")
    print(f"Time elapsed: {elapsed:.1f}s ({elapsed/max(total,1):.1f}s per line)")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
