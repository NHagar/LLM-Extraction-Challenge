import json
import os
from pathlib import Path

import pandas as pd
import torch
import whisperx
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from pydub import AudioSegment

load_dotenv()

# Configuration
INPUT_FILE = "oyez/trimmed.mp3"
OUTPUT_DIR = "oyez"
CHUNK_LENGTH_MS = 10 * 60 * 1000  # 10 minutes in milliseconds
OVERLAP_MS = 30 * 1000  # 30 seconds overlap
BATCH_SIZE = 16

Path(OUTPUT_DIR).mkdir(exist_ok=True)

print("Loading audio file...")
audio_mp3 = AudioSegment.from_mp3(INPUT_FILE)
duration_ms = len(audio_mp3)
duration_minutes = duration_ms / (60 * 1000)

print(f"Audio duration: {duration_minutes:.2f} minutes")

# Convert to WAV for pyannote compatibility
wav_path = f"{OUTPUT_DIR}/full_audio.wav"
print("Converting to WAV...")
audio_mp3.export(wav_path, format="wav")

# Load models once
print("Loading Whisper model...")
asr = whisperx.load_model("large-v3", device="cpu", compute_type="int8")

print("Loading alignment model...")
align_model, align_metadata = whisperx.load_align_model(
    language_code="en", device="cpu"
)

print("Loading diarization model...")
diar_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1", use_auth_token=os.getenv("HF_READ_TOKEN")
).to(torch.device("cpu"))

# Process in chunks if file is longer than chunk length
all_segments = []

if duration_ms <= CHUNK_LENGTH_MS:
    print("Processing single file (no chunking needed)...")

    audio = whisperx.load_audio(wav_path)
    print("Transcribing...")
    result = asr.transcribe(audio, batch_size=BATCH_SIZE)

    print("Aligning...")
    result = whisperx.align(
        result["segments"], align_model, align_metadata, audio, device="cpu"
    )

    print("Diarizing...")
    diar_result = diar_pipeline(wav_path)
    diar_data = []
    for turn, _, speaker in diar_result.itertracks(yield_label=True):
        diar_data.append({"start": turn.start, "end": turn.end, "speaker": speaker})
    diar_segments = pd.DataFrame(diar_data)

    print("Assigning speakers...")
    final = whisperx.assign_word_speakers(diar_segments, result)
    all_segments = final["segments"]

else:
    print(
        f"Processing in chunks of {CHUNK_LENGTH_MS / 60000:.1f} minutes with {OVERLAP_MS / 1000:.1f}s overlap..."
    )

    num_chunks = 0
    start_ms = 0

    while start_ms < duration_ms:
        end_ms = min(start_ms + CHUNK_LENGTH_MS, duration_ms)
        chunk_num = num_chunks + 1

        print(f"\n{'=' * 60}")
        print(f"Chunk {chunk_num}: {start_ms / 60000:.2f} - {end_ms / 60000:.2f} min")
        print(f"{'=' * 60}")

        # Extract chunk
        print("  Extracting chunk...")
        chunk = audio_mp3[start_ms:end_ms]
        chunk_path = f"{OUTPUT_DIR}/chunk_{chunk_num}.wav"
        chunk.export(chunk_path, format="wav")

        # Transcribe chunk
        print("  Transcribing...")
        audio = whisperx.load_audio(chunk_path)
        result = asr.transcribe(audio, batch_size=BATCH_SIZE)

        print("  Aligning...")
        result = whisperx.align(
            result["segments"], align_model, align_metadata, audio, device="cpu"
        )

        # Diarize chunk
        print("  Diarizing...")
        diar_result = diar_pipeline(chunk_path)
        diar_data = []
        for turn, _, speaker in diar_result.itertracks(yield_label=True):
            diar_data.append({"start": turn.start, "end": turn.end, "speaker": speaker})
        diar_segments = pd.DataFrame(diar_data)

        # Assign speakers to words
        print("  Assigning speakers...")
        chunk_result = whisperx.assign_word_speakers(diar_segments, result)

        # Adjust timestamps to global timeline
        time_offset = start_ms / 1000.0
        for segment in chunk_result["segments"]:
            segment["start"] += time_offset
            segment["end"] += time_offset
            if "words" in segment:
                for word in segment["words"]:
                    word["start"] += time_offset
                    word["end"] += time_offset

        # Add to all segments (skip overlapping parts except for first chunk)
        if num_chunks == 0:
            all_segments.extend(chunk_result["segments"])
        else:
            # Skip segments in the overlap region
            overlap_threshold = time_offset + (OVERLAP_MS / 2000.0)
            for segment in chunk_result["segments"]:
                if segment["start"] >= overlap_threshold:
                    all_segments.append(segment)

        # Cleanup chunk file
        os.remove(chunk_path)
        print(f"  ✓ Chunk {chunk_num} complete")

        num_chunks += 1

        # Move to next chunk with overlap
        if end_ms < duration_ms:
            start_ms = end_ms - OVERLAP_MS
        else:
            break

# Create final result
final = {"segments": all_segments}

print(f"\n{'=' * 60}")
print("Saving results...")
print(f"{'=' * 60}")

# Save final object to JSON
json_path = f"{OUTPUT_DIR}/transcript_with_speakers.json"
with open(json_path, "w") as f:
    json.dump(final, f, indent=2)
print(f"✓ Saved detailed JSON to {json_path}")

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

# Save clean transcript
transcript_path = f"{OUTPUT_DIR}/transcript_clean.txt"
with open(transcript_path, "w") as f:
    f.write("\n\n".join(transcript))
print(f"✓ Saved clean transcript to {transcript_path}")

# Print clean transcript
print("\n" + "=" * 60)
print("DIARIZED TRANSCRIPT")
print("=" * 60 + "\n")
for line in transcript:
    print(line)
    print()

print(f"\n{'=' * 60}")
print("✓ Processing complete!")
print(f"  Total segments: {len(final['segments'])}")
print(
    f"  Total speakers: {len(set(s.get('speaker', 'UNKNOWN') for s in final['segments']))}"
)
print(f"{'=' * 60}")
