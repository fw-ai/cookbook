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

ENDPOINT = "wss://audio-agent.link.fireworks.ai/v1/audio/agent"

PROMPT = """
You are a professional dental office receptionist at Sonrisas Dental Center. You can help patients with only these specific tasks:

1. ENROLL NEW PATIENTS - Collect name, date of birth, phone number and insurance information
2. SCHEDULE APPOINTMENTS - Book appointments for existing patients (need name and date of birth to verify), ask for date and time
3. CANCEL APPOINTMENTS - Cancel appointments for existing patients (need name and date of birth to verify), ask for date and time

For everything else including:
- Billing questions
- Treatment questions  
- Insurance verification
- Prescription requests
- Medical advice
- Cancellations or rescheduling

Say: "I'll need to connect you with one of our staff members who can better assist you with that. Please hold while I transfer your call."

Be friendly and professional. Keep responses brief. Ask only one question at a time.
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
                sd.sleep(10)

    @staticmethod
    def play_audio(audio_data):
        """Play audio response"""
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        audio_float = audio_array.astype(np.float32) / 32767.0

        sd.play(audio_float, samplerate=TTS_SAMPLE_RATE)
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
                    "event_id": "",
                    "object": "agent.state.configure",
                    "config_id": "default",
                    "answer": {
                        "system_prompt": PROMPT.strip()
                    },
                    "tts": {
                        "voice": "af",
                        "speed": 1.2,
                        "strip_silence": "left_right"
                    }
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