import asyncio
import json
import os
import websockets
import sounddevice as sd
import numpy as np
import threading
from queue import Queue
from dotenv import load_dotenv

load_dotenv()

# Audio configuration
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
TTS_SAMPLE_RATE = 44100

# Try the correct endpoint
ENDPOINT = "wss://audio-agent.link.fireworks.ai/v1/audio/agent"

PROMPT = """
You are a professional healthcare office assistant at Acme Medical Center. Your role is to help patients by:

1. Triaging calls into new or existing patients
2. Answering common questions about appointments, office hours, and procedures
3. Collecting basic information for scheduling

For new patients, ask for: name, date of birth, insurance, and reason for visit.
For existing patients, ask for: name, date of birth, and what they need help with.

Be friendly, professional, and efficient. Keep responses brief and focused. Always ask one question at a time.

If you cannot help with something, politely direct them to speak with a nurse or doctor.
"""


class VoiceAgent:
    def __init__(self):
        self.audio_queue = Queue()
        self.recording = True

    def record_audio(self):
        """Record audio from microphone"""

        def audio_callback(indata, frames, time, status):
            if self.recording:
                # Convert float32 to int16 and put in queue
                audio_data = (indata[:, 0] * 32767).astype(np.int16).tobytes()
                self.audio_queue.put(audio_data)

        with sd.InputStream(
                channels=1,
                samplerate=SAMPLE_RATE,
                blocksize=CHUNK_SIZE,
                callback=audio_callback,
                dtype=np.float32
        ):
            while self.recording:
                sd.sleep(100)

    def play_audio(self, audio_data):
        """Play audio response"""
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        audio_float = audio_array.astype(np.float32) / 32767.0

        tts_sample_rate = TTS_SAMPLE_RATE
        sd.play(audio_float, samplerate=tts_sample_rate)
        sd.wait()

    async def run(self):
        """Connect to voice agent and handle conversation"""
        url = ENDPOINT

        try:
            print(f"🔗 Connecting to: {url}")

            # Start recording in background
            threading.Thread(target=self.record_audio, daemon=True).start()

            async with websockets.connect(
                    url,
                    additional_headers={"Authorization": f"Bearer {os.environ['FIREWORKS_API_KEY']}"}
            ) as ws:
                # Send configuration
                config = {
                    "object": "agent.state.configure",
                    "system_prompt": PROMPT,
                    "tts_voice": "af_bella",
                    "tts_speed": 1.2
                }
                await ws.send(json.dumps(config))
                print("🎤 Connected! Start speaking...")

                # Handle messages
                async def send_audio():
                    while True:
                        if not self.audio_queue.empty():
                            audio_data = self.audio_queue.get()
                            await ws.send(audio_data)
                        await asyncio.sleep(0.01)

                async def receive_messages():
                    async for message in ws:
                        if isinstance(message, bytes):
                            # Audio response - play it
                            threading.Thread(target=self.play_audio, args=(message,), daemon=True).start()
                        else:
                            # Text message - just log it
                            print(f"Received: {message}")

                # Run both send and receive
                await asyncio.gather(send_audio(), receive_messages())

        except Exception as e:
            print(f"❌ Connection error: {e}")


# Run the voice agent
if __name__ == "__main__":
    api_key = os.environ.get('FIREWORKS_API_KEY')
    if not api_key:
        print("❌ Please set FIREWORKS_API_KEY environment variable")
        exit(1)

    agent = VoiceAgent()
    try:
        asyncio.run(agent.run())
    except KeyboardInterrupt:
        agent.recording = False
        print("\nGoodbye!")