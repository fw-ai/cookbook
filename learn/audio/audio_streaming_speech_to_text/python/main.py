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
from dotenv import load_dotenv

load_dotenv()

# Configuration
AUDIO_URL = "https://storage.googleapis.com/fireworks-public/test/3.5m.flac"
WEBSOCKET_URL = "wss://audio-streaming.us-virginia-1.direct.fireworks.ai/v1/audio/transcriptions/streaming"
TARGET_SAMPLE_RATE = 16000
CHUNK_SIZE_MS = 50
LANGUAGE = "en"


class TranscriptionClient:
    """Handles real-time audio transcription via WebSocket streaming."""

    def __init__(self):
        self.state = {}
        self.lock = threading.Lock()
        self.audio_chunks = []

    @staticmethod
    def download_audio(url):
        """Download audio file from URL."""
        print(f"Downloading audio from: {url}")
        response = requests.get(url)
        response.raise_for_status()

        audio_bytes = response.content
        print(f"Downloaded audio file size: {len(audio_bytes)} bytes")
        return audio_bytes

    def process_audio(self, audio_bytes):
        """Convert audio to the required format and chunk it."""
        print("Processing audio...")

        audio_tensor, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))
        print(f"Loaded audio: shape={audio_tensor.shape}, sample_rate={sample_rate}")

        if sample_rate != TARGET_SAMPLE_RATE:
            audio_tensor = torchaudio.functional.resample(
                audio_tensor, sample_rate, TARGET_SAMPLE_RATE
            )
            print(f"Resampled to {TARGET_SAMPLE_RATE}Hz: shape={audio_tensor.shape}")

        if audio_tensor.shape[0] > 1:
            audio_tensor = audio_tensor.mean(dim=0, keepdim=True)
            print(f"Converted to mono: shape={audio_tensor.shape}")

        samples_per_chunk = int(CHUNK_SIZE_MS * TARGET_SAMPLE_RATE / 1000)
        chunk_tensors = torch.split(audio_tensor, samples_per_chunk, dim=1)
        print(f"Split into {len(chunk_tensors)} chunks of {CHUNK_SIZE_MS}ms each")

        # Convert chunks to bytes
        audio_chunks = []
        for chunk_tensor in chunk_tensors:
            # Convert to 16-bit integers and then to bytes
            chunk_bytes = (chunk_tensor * 32768.0).to(torch.int16).numpy().tobytes()
            audio_chunks.append(chunk_bytes)

        self.audio_chunks = audio_chunks
        return audio_chunks

    def stream_audio(self, ws):
        """Stream audio chunks to the WebSocket."""
        print("Starting audio stream...")

        for i, chunk in enumerate(self.audio_chunks):
            ws.send(chunk, opcode=websocket.ABNF.OPCODE_BINARY)
            time.sleep(CHUNK_SIZE_MS / 1000)  # Maintain real-time pace

            if (i + 1) % 20 == 0:  # Progress update every second
                print(f"Streamed {i + 1}/{len(self.audio_chunks)} chunks")

        final_trace = json.dumps({
            "event_id": "streaming_complete",
            "object": "stt.input.trace",
            "trace_id": "final"
        })
        ws.send(final_trace, opcode=websocket.ABNF.OPCODE_TEXT)
        print("Audio streaming complete - final trace sent")

    def on_websocket_open(self, ws):
        """Handle WebSocket connection opening."""
        print("WebSocket connected - starting audio stream")
        # Start streaming in a separate thread
        streaming_thread = threading.Thread(
            target=self.stream_audio,
            args=(ws,),
            daemon=True
        )
        streaming_thread.start()

    def on_websocket_message(self, ws, message):
        """Handle incoming transcription messages."""
        try:
            data = json.loads(message)

            # Check for final trace completion
            if data.get("trace_id") == "final":
                print("\nTranscription complete!")
                ws.close()
                return

            # Update transcription state
            if "segments" in data:
                with self.lock:
                    self.state = {segment["id"]: segment["text"] for segment in data["segments"]}
                    self.display_transcription()

        except json.JSONDecodeError:
            print(f"Failed to parse message: {message}")

    @staticmethod
    def on_websocket_error(_, error):
        """Handle WebSocket errors."""
        print(f"WebSocket error: {error}")

    def display_transcription(self):
        """Display the current transcription state."""
        print("\n--- Current Transcription ---")
        for segment_id in sorted(self.state.keys(), key=int):
            print(f"{segment_id}: {self.state[segment_id]}")
        print("----------------------------\n")

    def create_websocket_connection(self):
        """Create and configure the WebSocket connection."""
        # Build WebSocket URL with parameters
        params = urllib.parse.urlencode({"language": LANGUAGE})
        full_url = f"{WEBSOCKET_URL}?{params}"

        api_key = os.environ.get("FIREWORKS_API_KEY")
        if not api_key:
            raise ValueError("FIREWORKS_API_KEY environment variable not set")

        websocket_client = websocket.WebSocketApp(
            full_url,
            header={"Authorization": api_key},
            on_open=self.on_websocket_open,
            on_message=self.on_websocket_message,
            on_error=self.on_websocket_error,
        )

        return websocket_client

    def run(self):
        """Main execution flow."""
        try:
            audio_bytes = self.download_audio(AUDIO_URL)
            self.process_audio(audio_bytes)
            websocket_client = self.create_websocket_connection()

            print("Connecting to transcription service...")
            websocket_client.run_forever()

        except Exception as e:
            print(f"Error: {e}")
            return 1

        return 0


def main():
    """Entry point for the transcription demo."""
    print("🎤 Fireworks Audio Transcription Demo")
    print("=" * 40)

    client = TranscriptionClient()
    return client.run()


if __name__ == "__main__":
    exit(main())