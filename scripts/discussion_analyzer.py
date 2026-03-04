from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="HF_TOKEN"
)

diarization = pipeline("data/recordings/2026-03-04_12.51.wav")

# 例: (start, end, speaker) を列挙
segments = []
for turn, _, speaker in diarization.itertracks(yield_label=True):
    segments.append((turn.start, turn.end, speaker))

print("n_segments:", len(segments))
print("head:", segments[:10])