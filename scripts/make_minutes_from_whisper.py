import json
import subprocess
from pathlib import Path

import whisper


def ensure_wav_16k_mono(in_path: str) -> str:
    p = Path(in_path)
    if p.suffix.lower() == ".wav":
        return str(p)
    out = p.with_suffix(".wav")
    cmd = ["ffmpeg", "-y", "-i", str(p), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out)]
    subprocess.run(cmd, check=True)
    return str(out)


def assign_speaker(seg_start: float, seg_end: float, speaker_lines):
    # speaker_lines: list of dict with start/end/speaker_index
    best_spk, best = None, 0.0
    for ln in speaker_lines:
        s0, s1, spk = ln["start"], ln["end"], ln["speaker_index"]
        ov = max(0.0, min(seg_end, s1) - max(seg_start, s0))
        if ov > best:
            best = ov
            best_spk = spk
    return best_spk


def main():
    audio_in = "data/recordings/2026-03-04_12.51.m4a"
    stem = Path(audio_in).stem
    out_dir = Path("data/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    labeled_path = out_dir / f"{stem}.moonshine.labeled.json"
    speaker_lines = json.loads(labeled_path.read_text(encoding="utf-8"))

    wav_path = ensure_wav_16k_mono(audio_in)

    # 高品質を狙うなら large / large-v3 など（環境によりDLに時間）
    model = whisper.load_model("large-v3")  # まず heavy。重ければ medium に落としてOK
    result = model.transcribe(wav_path, language="ja", verbose=False)

    # 出力：話者付き議事録テキスト
    out_txt = out_dir / f"{stem}.minutes.txt"
    with out_txt.open("w", encoding="utf-8") as f:
        current_spk = None
        for seg in result["segments"]:
            spk = assign_speaker(seg["start"], seg["end"], speaker_lines)
            text = (seg["text"] or "").strip()
            if not text:
                continue

            # 話者が変わったら見出しを入れる
            if spk != current_spk:
                f.write("\n")
                f.write(f"SPEAKER_{spk}\n")
                current_spk = spk

            f.write(text + "\n")

    print("saved:", out_txt)


if __name__ == "__main__":
    main()