"""Discussion analyzer: Moonshine transcription + speaker enrollment via pyannote embeddings.

Usage:
  1. Place ~10s enrollment audio per speaker in data/enrollment/<name>.wav
     (file stem = speaker name, e.g. 本田.wav, 渡辺.wav)
  2. Set AUDIO_IN below to the target recording.
  3. Set HF_TOKEN environment variable for the pyannote embedding model.
  4. python scripts/moonshine_discussion_analyzer.py

If enrollment audio is found, speakers are identified by voiceprint matching.
Otherwise, falls back to moonshine's built-in (less reliable) speaker identification.
"""

import json
import os
import platform
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial.distance import cosine as cosine_dist

from moonshine_voice import (
    Transcriber,
    TranscriptEventListener,
    get_model_for_language,
    load_wav_file,
)

# ===================== Configuration =====================
AUDIO_IN = "data/recordings/2026-03-04_12.51.m4a"
ENROLLMENT_DIR = Path("data/enrollment")
ENROLLMENT_THRESHOLD = 0.35  # cosine similarity; raise for stricter matching
MIN_DURATION_FOR_ID = 0.5  # seconds – segments shorter than this skip embedding
# =========================================================

_AUDIO_EXTS = {".wav", ".m4a", ".mp3", ".flac", ".ogg"}


# -----------------------------
# Audio utilities
# -----------------------------
def ensure_wav_16k_mono(in_path: str) -> str:
    """Convert audio to 16 kHz mono PCM WAV. On macOS prefer afconvert."""
    p = Path(in_path)
    if p.suffix.lower() == ".wav":
        return str(p)

    out = p.with_suffix(".wav")
    if platform.system() == "Darwin":
        cmd = [
            "afconvert", str(p),
            "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(out),
        ]
    else:
        cmd = [
            "ffmpeg", "-y", "-i", str(p),
            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out),
        ]
    subprocess.run(cmd, check=True)
    return str(out)


def chunk_audio(audio_data, sample_rate: int, chunk_duration_sec: float = 0.1):
    chunk_size = int(chunk_duration_sec * sample_rate)
    for i in range(0, len(audio_data), chunk_size):
        yield audio_data[i : i + chunk_size], sample_rate


# -----------------------------
# Text cleanup
# -----------------------------
_re_spaced_jp = re.compile(r"(?:(?<=\S)\s+(?=\S))")


def normalize_text(text: str) -> str:
    """Collapse the spaced-character artifact common in Moonshine Japanese output."""
    t = (text or "").strip()
    if not t:
        return t
    toks = t.split()
    if len(toks) >= 6:
        single = sum(1 for x in toks if len(x) == 1)
        if single / len(toks) >= 0.7:
            return "".join(toks)
    t = _re_spaced_jp.sub(" ", t)
    return t


# -----------------------------
# Speaker Enrollment
# -----------------------------
def _load_embedding_model():
    """Load pyannote speaker-embedding model (requires HF_TOKEN)."""
    from pyannote.audio import Inference, Model

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError(
            "Set the HF_TOKEN environment variable "
            "(https://huggingface.co/settings/tokens) "
            "for the pyannote embedding model."
        )
    model = Model.from_pretrained("pyannote/embedding", use_auth_token=token)
    return Inference(model, window="whole")


def build_enrollment(
    enrollment_dir: Path, inference,
) -> Dict[str, np.ndarray]:
    """Compute a reference embedding per enrolled speaker.

    Each audio file in *enrollment_dir* should contain ~10 s of speech from a
    single speaker.  The file stem is used as the speaker name.
    """
    files = sorted(
        f for f in enrollment_dir.iterdir() if f.suffix.lower() in _AUDIO_EXTS
    )
    enrollments: Dict[str, np.ndarray] = {}
    for f in files:
        wav = ensure_wav_16k_mono(str(f))
        emb = np.array(inference(wav)).flatten()
        enrollments[f.stem] = emb
        print(f"  {f.stem}: embedding dim={emb.shape[0]}")
    return enrollments


def _match_speaker(
    embedding: np.ndarray,
    enrollments: Dict[str, np.ndarray],
    threshold: float,
) -> Tuple[str, float]:
    """Return (speaker_name, similarity) for the best-matching enrollment."""
    scores = {
        name: 1.0 - cosine_dist(embedding, ref)
        for name, ref in enrollments.items()
    }
    best = max(scores, key=scores.get)
    sc = scores[best]
    return (best if sc >= threshold else "unknown"), float(sc)


def relabel_with_enrollment(
    lines: List[Dict],
    wav_path: str,
    enrollments: Dict[str, np.ndarray],
    inference,
    threshold: float,
    min_dur: float,
) -> None:
    """Add *speaker_name* and *speaker_confidence* fields to each line."""
    from pyannote.core import Segment

    n = len(lines)
    for i, ln in enumerate(lines):
        if ln.get("duration", 0) < min_dur:
            ln["speaker_name"] = "unknown"
            ln["speaker_confidence"] = 0.0
            continue
        try:
            seg = Segment(ln["start"], ln["end"])
            emb = np.array(inference.crop(wav_path, seg)).flatten()
            name, score = _match_speaker(emb, enrollments, threshold)
            ln["speaker_name"] = name
            ln["speaker_confidence"] = round(score, 4)
        except Exception as exc:
            print(f"  [warn] segment {i} ({ln['start']:.1f}–{ln['end']:.1f}s): {exc}")
            ln["speaker_name"] = "unknown"
            ln["speaker_confidence"] = 0.0

        if (i + 1) % 50 == 0 or i + 1 == n:
            print(f"  relabeled {i + 1}/{n}")


def _assign_speaker_labels(lines: List[Dict], enrolled: bool) -> None:
    """Set a unified *speaker_label* field used by all downstream analytics."""
    for ln in lines:
        if enrolled and ln.get("speaker_name"):
            ln["speaker_label"] = ln["speaker_name"]
        else:
            idx = ln.get("speaker_index", -1)
            ln["speaker_label"] = (
                f"SPEAKER_{idx}" if idx is not None and idx >= 0 else "unknown"
            )


# -----------------------------
# GraphML export
# -----------------------------
def write_graphml(edges: Counter, speakers: list, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<graphml xmlns="http://graphml.graphdrawing.org/xmlns">\n')
        f.write('  <key id="w" for="edge" attr.name="weight" attr.type="int"/>\n')
        f.write('  <graph id="G" edgedefault="directed">\n')
        for spk in speakers:
            f.write(f'    <node id="{spk}"/>\n')
        eid = 0
        for (a, b), w in edges.items():
            f.write(
                f'    <edge id="e{eid}" source="{a}" target="{b}">\n'
                f'      <data key="w">{int(w)}</data>\n'
                f'    </edge>\n'
            )
            eid += 1
        f.write("  </graph>\n</graphml>\n")


# -----------------------------
# Moonshine listener
# -----------------------------
class Collector(TranscriptEventListener):
    def __init__(self, progress_every: int = 20):
        self.completed_lines: List[Dict] = []
        self.progress_every = progress_every

    def on_line_completed(self, event):
        line = event.line
        start = float(getattr(line, "start_time", 0.0))
        dur = float(getattr(line, "duration", 0.0))
        end = start + dur

        has_spk = bool(getattr(line, "has_speaker_id", False))
        speaker_id = int(getattr(line, "speaker_id", -1)) if has_spk else -1
        speaker_index = int(getattr(line, "speaker_index", -1)) if has_spk else -1

        text = normalize_text(getattr(line, "text", "") or "")

        self.completed_lines.append(
            {
                "start": start,
                "duration": dur,
                "end": end,
                "text": text,
                "speaker_id": speaker_id,
                "speaker_index": speaker_index,
            }
        )

        if self.progress_every and len(self.completed_lines) % self.progress_every == 0:
            last = self.completed_lines[-1]
            print(
                f"[moonshine] lines={len(self.completed_lines)} "
                f"t={last['end']:.1f}s spk={last['speaker_index']} "
                f"text={last['text'][:30]}"
            )


# -----------------------------
# Analytics
# -----------------------------
def speaker_stats(lines: List[Dict]) -> List[Dict]:
    talk_sec = defaultdict(float)
    turns: Counter = Counter()

    for ln in lines:
        spk = ln.get("speaker_label", "unknown")
        talk_sec[spk] += max(0.0, float(ln.get("duration", 0.0)))
        turns[spk] += 1

    total = sum(talk_sec.values()) or 1.0
    return [
        {
            "speaker": spk,
            "talk_seconds": float(sec),
            "talk_ratio": float(sec / total),
            "turns": int(turns[spk]),
        }
        for spk, sec in sorted(talk_sec.items(), key=lambda kv: kv[1], reverse=True)
    ]


def interaction_edges(lines: List[Dict]) -> Counter:
    edges: Counter = Counter()
    seq = [
        ln["speaker_label"]
        for ln in lines
        if ln.get("speaker_label", "unknown") != "unknown"
    ]
    for a, b in zip(seq, seq[1:]):
        if a != b:
            edges[(a, b)] += 1
    return edges


def overlap_counts(lines: List[Dict]) -> Counter:
    sorted_lines = sorted(lines, key=lambda x: x["start"])
    overlaps: Counter = Counter()
    active: List[Tuple[float, str]] = []

    for ln in sorted_lines:
        s, e = float(ln["start"]), float(ln["end"])
        spk = ln.get("speaker_label", "unknown")
        if spk == "unknown":
            continue
        active = [(ae, asp) for ae, asp in active if ae > s]
        for ae, asp in active:
            if asp != spk:
                overlaps[(asp, spk)] += 1
        active.append((e, spk))

    return overlaps


# -----------------------------
# Main
# -----------------------------
def main():
    # ---- input ----
    wav_path = ensure_wav_16k_mono(AUDIO_IN)

    # ---- enrollment (optional) ----
    enrollments: Dict[str, np.ndarray] = {}
    emb_inference = None
    has_enrollment = ENROLLMENT_DIR.is_dir() and any(
        f for f in ENROLLMENT_DIR.iterdir() if f.suffix.lower() in _AUDIO_EXTS
    )

    if has_enrollment:
        print("== Speaker Enrollment ==")
        try:
            emb_inference = _load_embedding_model()
            enrollments = build_enrollment(ENROLLMENT_DIR, emb_inference)
            print(f"Enrolled {len(enrollments)} speakers: {list(enrollments.keys())}\n")
        except Exception as exc:
            print(f"[enrollment] Failed to load embedding model: {exc}")
            print("[enrollment] Falling back to moonshine speaker IDs.\n")
            enrollments = {}
    else:
        print(f"[info] No enrollment audio in {ENROLLMENT_DIR}/")
        print("[info] Falling back to moonshine built-in speaker ID.\n")

    # ---- moonshine transcription ----
    print("== Moonshine Transcription ==")
    model_path, model_arch = get_model_for_language("ja")

    transcriber = Transcriber(
        model_path=model_path,
        model_arch=model_arch,
        options={
            "identify_speakers": "true",
            "return_audio_data": "false",
            "log_output_text": "false",
        },
    )

    stream = transcriber.create_stream(update_interval=0.5)
    collector = Collector(progress_every=20)
    stream.add_listener(collector)

    stream.start()

    audio_data, sample_rate = load_wav_file(wav_path)
    for chunk, sr in chunk_audio(audio_data, sample_rate, chunk_duration_sec=0.1):
        stream.add_audio(chunk, sr)

    stream.stop()
    stream.close()

    lines = sorted(collector.completed_lines, key=lambda x: x["start"])

    # ---- re-label with enrollment embeddings ----
    if enrollments and emb_inference:
        print("\n== Speaker Identification (Enrollment) ==")
        relabel_with_enrollment(
            lines, wav_path, enrollments, emb_inference,
            ENROLLMENT_THRESHOLD, MIN_DURATION_FOR_ID,
        )

    _assign_speaker_labels(lines, enrolled=bool(enrollments))

    # ---- analytics ----
    stats = speaker_stats(lines)
    edges = interaction_edges(lines)
    overlaps = overlap_counts(lines)

    # ---- output ----
    stem = Path(AUDIO_IN).stem
    out_dir = Path("data/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_labeled = out_dir / f"{stem}.moonshine.labeled.json"
    out_labeled.write_text(
        json.dumps(lines, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    out_stats = out_dir / f"{stem}.moonshine.speaker_stats.tsv"
    with out_stats.open("w", encoding="utf-8") as f:
        f.write("speaker\ttalk_seconds\ttalk_ratio\tturns\n")
        for r in stats:
            f.write(
                f"{r['speaker']}\t{r['talk_seconds']:.6f}"
                f"\t{r['talk_ratio']:.10f}\t{r['turns']}\n"
            )

    out_edges = out_dir / f"{stem}.moonshine.edges.tsv"
    with out_edges.open("w", encoding="utf-8") as f:
        f.write("from\tto\tcount\n")
        for (a, b), c in edges.most_common():
            f.write(f"{a}\t{b}\t{c}\n")

    out_overlaps = out_dir / f"{stem}.moonshine.overlaps.tsv"
    with out_overlaps.open("w", encoding="utf-8") as f:
        f.write("from\tto\tcount\n")
        for (a, b), c in overlaps.most_common():
            f.write(f"{a}\t{b}\t{c}\n")

    speakers_list = sorted(
        r["speaker"] for r in stats if r["speaker"] != "unknown"
    )
    out_graphml = out_dir / f"{stem}.moonshine.interaction.graphml"
    write_graphml(edges, speakers_list, out_graphml)

    # ---- summary ----
    print("\n== Summary ==")
    print("n_lines:", len(lines))
    print("n_speakers:", len(speakers_list))
    if enrollments:
        identified = sum(
            1 for ln in lines if ln.get("speaker_name", "unknown") != "unknown"
        )
        print(
            f"identified_segments: {identified}/{len(lines)} "
            f"({identified / max(len(lines), 1) * 100:.1f}%)"
        )
    print("top speakers:")
    for r in stats[:5]:
        print(
            f"  {r['speaker']}: {r['talk_seconds']:.1f}s "
            f"({r['talk_ratio'] * 100:.1f}%), turns={r['turns']}"
        )
    print("top edges:", edges.most_common(10))
    print("top overlaps:", overlaps.most_common(10))
    print("\nSaved:")
    for p in (out_labeled, out_stats, out_edges, out_overlaps, out_graphml):
        print(" ", p)


if __name__ == "__main__":
    main()
