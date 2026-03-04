import os
import json
import subprocess
from pathlib import Path
from collections import Counter, defaultdict

import whisper
from pyannote.audio import Pipeline


def ensure_wav(input_path: str) -> str:
    p = Path(input_path)
    if p.suffix.lower() == ".wav":
        return str(p)
    out = p.with_suffix(".wav")
    cmd = ["ffmpeg", "-y", "-i", str(p), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out)]
    subprocess.run(cmd, check=True)
    return str(out)


def assign_speaker(seg_start: float, seg_end: float, speaker_segments):
    best_spk, best = None, 0.0
    for s0, s1, spk in speaker_segments:
        ov = max(0.0, min(seg_end, s1) - max(seg_start, s0))
        if ov > best:
            best, best_spk = ov, spk
    return best_spk


def write_graphml(edges_counter: Counter, speakers: set, out_path: Path) -> None:
    """Minimal GraphML for directed weighted graph."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<graphml xmlns="http://graphml.graphdrawing.org/xmlns">\n')
        f.write('  <key id="w" for="edge" attr.name="weight" attr.type="int"/>\n')
        f.write('  <graph id="G" edgedefault="directed">\n')
        for spk in sorted(speakers):
            f.write(f'    <node id="{spk}"/>\n')
        edge_id = 0
        for (a, b), w in edges_counter.items():
            f.write(f'    <edge id="e{edge_id}" source="{a}" target="{b}">\n')
            f.write(f'      <data key="w">{int(w)}</data>\n')
            f.write('    </edge>\n')
            edge_id += 1
        f.write("  </graph>\n</graphml>\n")


def main():
    audio_in = "data/recordings/2026-03-04_12.51.m4a"  # ここを適宜変更
    audio_path = ensure_wav(audio_in)

    out_dir = Path("data/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) diarization
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=os.environ["HF_TOKEN"],
    )
    diarization = pipeline(audio_path)

    speaker_segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_segments.append((turn.start, turn.end, speaker))

    print("n_speaker_segments:", len(speaker_segments))
    print("speaker_head:", speaker_segments[:10])

    # ---- (1) speaker stats from diarization (talk time / turn count) ----
    spk_seconds = defaultdict(float)
    spk_turns = Counter()
    for s0, s1, spk in speaker_segments:
        dur = max(0.0, s1 - s0)
        spk_seconds[spk] += dur
        spk_turns[spk] += 1

    total_spk_time = sum(spk_seconds.values()) or 1.0

    speaker_stats = []
    for spk in sorted(spk_seconds.keys()):
        speaker_stats.append(
            {
                "speaker": spk,
                "talk_seconds": spk_seconds[spk],
                "talk_ratio": spk_seconds[spk] / total_spk_time,
                "turns": int(spk_turns[spk]),
            }
        )

    print("\nSpeaker talk-time (top):")
    for row in sorted(speaker_stats, key=lambda x: x["talk_seconds"], reverse=True)[:10]:
        print(
            f'{row["speaker"]}: {row["talk_seconds"]:.1f}s '
            f'({row["talk_ratio"]*100:.1f}%), turns={row["turns"]}'
        )

    out_stats_tsv = out_dir / (Path(audio_in).stem + ".speaker_stats.tsv")
    with out_stats_tsv.open("w", encoding="utf-8") as f:
        f.write("speaker\ttalk_seconds\ttalk_ratio\tturns\n")
        for row in sorted(speaker_stats, key=lambda x: x["talk_seconds"], reverse=True):
            f.write(
                f'{row["speaker"]}\t{row["talk_seconds"]:.6f}\t'
                f'{row["talk_ratio"]:.10f}\t{row["turns"]}\n'
            )

    # 2) whisper transcription
    model = whisper.load_model("medium")  # まずは small でもOK
    w = model.transcribe(audio_path, language="ja", verbose=False)

    # 3) attach speaker label to whisper segments
    labeled = []
    for seg in w["segments"]:
        spk = assign_speaker(seg["start"], seg["end"], speaker_segments)
        labeled.append(
            {
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "speaker": spk,
                "text": seg["text"].strip(),
            }
        )

    out_json = out_dir / (Path(audio_in).stem + ".labeled.json")
    out_json.write_text(json.dumps(labeled, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- (3) interaction graph (speaker switches) ----
    edges = Counter()
    for prev, cur in zip(labeled, labeled[1:]):
        a, b = prev["speaker"], cur["speaker"]
        if a and b and a != b:
            edges[(a, b)] += 1

    print("\nTop speaker-switch edges:")
    for (a, b), c in edges.most_common(10):
        print(f"{a} -> {b}: {c}")

    out_edges = out_dir / (Path(audio_in).stem + ".edges.tsv")
    with out_edges.open("w", encoding="utf-8") as f:
        f.write("from\to\tcount\n")
        for (a, b), c in edges.most_common():
            f.write(f"{a}\t{b}\t{c}\n")

    # GraphML export (open in Gephi / Cytoscape / yEd)
    speakers = set(spk_seconds.keys())
    out_graphml = out_dir / (Path(audio_in).stem + ".interaction.graphml")
    write_graphml(edges, speakers, out_graphml)

    print("\nSaved:")
    print(" ", out_json)
    print(" ", out_edges)
    print(" ", out_stats_tsv)
    print(" ", out_graphml)


if __name__ == "__main__":
    main()