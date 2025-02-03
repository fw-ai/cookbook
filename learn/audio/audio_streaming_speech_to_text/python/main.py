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


# The client maintains a state dictionary, starting with an empty
# dictionary `{}`. When the server sends the first transcription message,
# it contains a list of segments. Each segment has an `id` and `text`:
#
# Server initial message:
# {
#     "segments": [
#         {"id": "0", "text": "This is the first sentence"},
#         {"id": "1", "text": "This is the second sentence"}
#     ]
# }
#
# Client initial state:
# {
#     "0": "This is the first sentence",
#     "1": "This is the second sentence",
# }
#
# When the server sends the next updates to the transcription, the client
# updates the state dictionary based on the segment `id`:
#
# Server continuous message:
# {
#     "segments": [
#         {"id": "1", "text": "This is the second sentence modified"},
#         {"id": "2", "text": "This is the third sentence"}
#     ]
# }
#
# Client updated state:
# {
#     "0": "This is the first sentence",
#     "1": "This is the second sentence modified",   # overwritten
#     "2": "This is the third sentence",             # new
# }

lock = threading.Lock()
state = {}


def on_open(ws):
    def stream_audio(ws):
        for chunk in audio_chunk_bytes:
            ws.send(chunk, opcode=websocket.ABNF.OPCODE_BINARY)
            time.sleep(chunk_size_ms / 1000)

        final_checkpoint = json.dumps({"checkpoint_id": "final"})
        ws.send(final_checkpoint, opcode=websocket.ABNF.OPCODE_TEXT)

    threading.Thread(target=stream_audio, args=(ws,)).start()


def on_error(ws, error):
    print(f"Error: {error}")


def on_message(ws, message):
    message = json.loads(message)
    if message.get("checkpoint_id") == "final":
        ws.close()
        return

    update = {s["id"]: s["text"] for s in message["segments"]}
    with lock:
        state.update(update)
        print("\n".join(f" - {k}: {v}" for k, v in state.items()))


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
