import io
import os
import requests
import torch
import torchaudio
import json
import threading
import time
import websocket
import urllib.parse


# Download audio file
url = "https://storage.googleapis.com/fireworks-public/test/3.5m.flac"
response = requests.get(url)
audio_bytes = response.content
print(f"Downloaded audio file size: {len(audio_bytes)} bytes")

# Load to torch tensor
audio_tensor, sr = torchaudio.load(io.BytesIO(audio_bytes))
print(f"Loaded audio. shape={audio_tensor.shape} sample_rate={sr}")

# Resample to 16000 Hz
target_sr = 16000
audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
print(f"Resampled audio. shape={audio_tensor.shape} sample_rate={target_sr}")

# Convert to mono
audio_tensor = audio_tensor.mean(dim=0, keepdim=True)
print(f"Mono audio. shape={audio_tensor.shape}")

# Split into chunks of 50ms
chunk_size_ms = 50
audio_chunk_tensors = torch.split(audio_tensor, int(
    chunk_size_ms * target_sr / 1000), dim=1
)
print(f"Split into {len(audio_chunk_tensors)} chunks each {chunk_size_ms}ms")

# Convert to bytes
audio_chunk_bytes = []
for audio_chunk_tensor in audio_chunk_tensors:
    audio_chunk_bytes.append(
        (audio_chunk_tensor * 32768.0).to(torch.int16).numpy().tobytes())


# Segments state dictionary works helps to keep track of the segments and their updates. For example:
#
# 1. Segments before an update:
#
#     {
#         "0": "sentence0",
#         "1": "sentence1",
#         "2": "sentence2",
#     }
#
# 2. A new message with an update is received (on_message):
#
#     {
#         "1": "sentence1",
#         "2": "sentence2_updated",
#     }
#
# 3. Segments state after the update:
#
#     {
#         "0": "sentence0",
#         "1": "sentence1",
#         "2": "sentence2_updated",
#     }

lock = threading.Lock()
segments = {}


def on_open(ws):
    def stream_audio(ws):
        for chunk in audio_chunk_bytes:
            ws.send(chunk, opcode=websocket.ABNF.OPCODE_BINARY)
            time.sleep(chunk_size_ms / 1000)

        time.sleep(2)  # Custom post-processing
        ws.close()

    threading.Thread(target=stream_audio, args=(ws,)).start()


def on_error(ws, error):
    print(f"Error: {error}")


def on_message(ws, message):
    updated_segments = {
        segment["id"]: segment["text"]
        for segment in json.loads(message)["segments"]
    }
    with lock:
        segments.update(updated_segments)
        print("\n".join(f" - {k}: {v}" for k, v in segments.items()))


url = "ws://audio-streaming.us-virginia-1.direct.fireworks.ai/v1/audio/transcriptions/streaming"
params = urllib.parse.urlencode({
    "language": "en",
})
ws = websocket.WebSocketApp(
    f"{url}?{params}",
    header={
        "Authorization": os.environ["FIREWORKS_API_KEY"],
    },
    on_open=on_open,
    on_message=on_message,
    on_error=on_error,
)
ws.run_forever()
