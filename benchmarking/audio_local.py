import json
import os

import pandas as pd
import torch
import whisperx
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from pydub import AudioSegment

load_dotenv()

# Convert MP3 to WAV for pyannote compatibility
audio_mp3 = AudioSegment.from_mp3("oyez/trimmed.mp3")
audio_mp3.export("oyez/trimmed.wav", format="wav")

asr = whisperx.load_model("large-v3", device="cpu", compute_type="int8")
audio = whisperx.load_audio("oyez/trimmed.wav")
result = asr.transcribe(audio, batch_size=8)

# Alignment (optional but improves word timings)
align_model, align_metadata = whisperx.load_align_model(
    language_code="en", device="cpu"
)
result = whisperx.align(
    result["segments"], align_model, align_metadata, audio, device="cpu"
)

# Diarization (pyannote 3.1)
diar = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1", use_auth_token=os.getenv("HF_READ_TOKEN")
).to(torch.device("cpu"))
diar_result = diar("oyez/trimmed.wav")

# Convert pyannote output to whisperx format (DataFrame)
diar_data = []
for turn, _, speaker in diar_result.itertracks(yield_label=True):
    diar_data.append({"start": turn.start, "end": turn.end, "speaker": speaker})
diar_segments = pd.DataFrame(diar_data)

# Stitch text + speakers
final = whisperx.assign_word_speakers(diar_segments, result)

# Save final object to JSON
with open("benchmarking/data/transcript_with_speakers.json", "w") as f:
    json.dump(final, f, indent=2)

# Parse into clean diarized transcript
transcript = []
current_speaker = None
current_text = []

for segment in final["segments"]:
    speaker = segment.get("speaker", "UNKNOWN")
    text = segment["text"].strip()

    if speaker != current_speaker:
        # Save previous speaker's text
        if current_speaker is not None and current_text:
            transcript.append(f"{current_speaker}: {' '.join(current_text)}")

        # Start new speaker
        current_speaker = speaker
        current_text = [text]
    else:
        current_text.append(text)

# Add the last speaker's text
if current_speaker is not None and current_text:
    transcript.append(f"{current_speaker}: {' '.join(current_text)}")

# Print clean transcript
print("\n=== Diarized Transcript ===\n")
for line in transcript:
    print(line)
    print()
