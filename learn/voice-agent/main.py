import asyncio
import json
import os
import websockets
import sounddevice as sd
import numpy as np
import threading
from queue import Queue
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Audio configuration
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
TTS_SAMPLE_RATE = 44100

ENDPOINT = "wss://audio-agent.link.fireworks.ai/v1/audio/agent"

PROMPT = """
You are a professional dental office receptionist at Sonrisas Dental Center. You can help patients with:

1. ENROLL NEW PATIENTS - Collect name and phone number
2. SCHEDULE APPOINTMENTS - Book appointments for existing patients, ask for preferred date and time
3. CANCEL APPOINTMENTS - Cancel appointments for existing patients
4. CHECK AVAILABILITY - Show available appointment slots

Use the available functions to help patients. Be friendly and professional. Keep responses brief and conversational.
"""


def cancel_appointment(patient_name: str, date: str, time: str) -> dict:
    """Cancel an appointment for a patient"""
    return {
        "patient_name": patient_name,
        "date": date,
        "time": time,
        "status": "cancelled",
        "message": f"Appointment for {patient_name} on {date} at {time} has been cancelled successfully"
    }


def schedule_appointment(patient_name: str, date: str, time: str, service: str = "General Checkup") -> dict:
    """Schedule an appointment for a patient"""
    try:
        appointment_date = datetime.strptime(date, "%Y-%m-%d").date()

        # Mock scheduling - in real implementation, this would update your scheduling system
        appointment_id = f"APT-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        return {
            "appointment_id": appointment_id,
            "patient_name": patient_name,
            "date": appointment_date.strftime("%Y-%m-%d"),
            "time": time,
            "service": service,
            "status": "confirmed",
            "message": f"Appointment scheduled successfully for {patient_name} on {date} at {time}"
        }
    except ValueError:
        return {"error": "Invalid date format. Please use YYYY-MM-DD format."}


def check_availability(date: str) -> dict:
    """Check available appointment slots for a given date"""
    try:
        # Parse the date
        target_date = datetime.strptime(date, "%Y-%m-%d").date()

        # Mock availability - in real implementation, this would query your scheduling system
        available_slots = [
            "9:00 AM", "10:30 AM", "2:00 PM", "3:30 PM", "4:45 PM"
        ]

        return {
            "date": target_date.strftime("%Y-%m-%d"),
            "available_slots": available_slots,
            "total_slots": len(available_slots)
        }
    except ValueError:
        return {"error": "Invalid date format. Please use YYYY-MM-DD format."}


class VoiceAgent:
    def __init__(self):
        self.audio_queue = Queue()
        self.recording = True
        self.websocket = None
        self.current_transcript = ""
        self.last_response = ""

    def record_audio(self):
        """Record audio from microphone with error handling"""

        def audio_callback(indata, frames, time, status):
            if status:
                return  # Skip problematic frames
            if self.recording:
                # Convert float32 to int16 and put in queue
                audio_data = (indata[:, 0] * 32767).astype(np.int16).tobytes()
                self.audio_queue.put(audio_data)

        try:
            with sd.InputStream(
                    channels=1,
                    samplerate=SAMPLE_RATE,
                    blocksize=CHUNK_SIZE,
                    callback=audio_callback,
                    dtype=np.float32,
                    latency='low'  # Reduce latency issues
            ):
                while self.recording:
                    sd.sleep(10)
        except Exception as e:
            print(f"⚠️  Audio recording error: {e}")
            self.recording = False

    @staticmethod
    def play_audio(audio_data):
        """Play audio response with error suppression"""
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_float = audio_array.astype(np.float32) / 32767.0

            # Suppress Core Audio errors by reducing device interaction
            with sd.OutputStream(samplerate=TTS_SAMPLE_RATE, channels=1, dtype='float32') as stream:
                stream.write(audio_float.reshape(-1, 1))
        except Exception as e:
            print(f"⚠️  Audio playback error: {e}")
            try:
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                audio_float = audio_array.astype(np.float32) / 32767.0
                sd.play(audio_float, samplerate=TTS_SAMPLE_RATE)
                sd.wait()
            except:
                pass  # Silently handle audio errors

    @staticmethod
    def print_conversation_turn(user_text=None, assistant_text=None):
        """Print clean conversation turns"""
        if user_text:
            print(f"\n👤 User: {user_text}")
        if assistant_text:
            print(f"🤖 Assistant: {assistant_text}")

    # Tool implementations

    @staticmethod
    async def handle_tool_call(tool_calls: list) -> dict:
        """Handle function calls from the agent"""
        results = {}

        for tool_call in tool_calls:
            call_id = tool_call.get("id")
            function_info = tool_call.get("function", {})
            function_name = function_info.get("name")

            try:
                # Parse function arguments
                arguments = json.loads(function_info.get("arguments", "{}"))

                # Route to appropriate function
                if function_name == "check_availability":
                    result = check_availability(arguments.get("date"))
                elif function_name == "schedule_appointment":
                    result = schedule_appointment(
                        arguments.get("patient_name"),
                        arguments.get("date"),
                        arguments.get("time"),
                        arguments.get("service", "General Checkup")
                    )
                elif function_name == "cancel_appointment":
                    result = cancel_appointment(
                        arguments.get("patient_name"),
                        arguments.get("date"),
                        arguments.get("time")
                    )
                else:
                    result = {"error": f"Unknown function: {function_name}"}

                results[call_id] = result
                print(f"🔧 Function called: {function_name} -> {result}")

            except Exception as e:
                results[call_id] = {"error": f"Function execution error: {str(e)}"}
                print(f"❌ Error executing {function_name}: {e}")

        return results

    async def send_tool_result(self, tool_results: dict):
        """Send tool execution results back to the agent"""
        if self.websocket:
            tool_result_message = {
                "event_id": "",
                "object": "agent.input.tool_result",
                "tool_results": tool_results
            }
            await self.websocket.send(json.dumps(tool_result_message))

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
                self.websocket = ws

                # Send configuration with optimized settings for responsiveness
                config = {
                    "event_id": "",
                    "object": "agent.state.configure",
                    "config_id": "default",
                    "answer": {
                        "system_prompt": PROMPT.strip(),
                        "max_tokens": 150,  # Shorter responses for better flow
                        "temperature": 0.7,  # More natural but consistent
                        "tool_config": {
                            "system_prompt": "Use the available functions when patients need scheduling help. Be efficient and direct.",
                            "tools": [
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "check_availability",
                                        "description": "Check available appointment slots for a specific date",
                                        "parameters": {
                                            "type": "object",
                                            "properties": {
                                                "date": {
                                                    "type": "string",
                                                    "description": "Date to check availability for (YYYY-MM-DD format)"
                                                }
                                            },
                                            "required": ["date"]
                                        }
                                    }
                                },
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "schedule_appointment",
                                        "description": "Schedule an appointment for a patient",
                                        "parameters": {
                                            "type": "object",
                                            "properties": {
                                                "patient_name": {
                                                    "type": "string",
                                                    "description": "Full name of the patient"
                                                },
                                                "date": {
                                                    "type": "string",
                                                    "description": "Appointment date (YYYY-MM-DD format)"
                                                },
                                                "time": {
                                                    "type": "string",
                                                    "description": "Appointment time (e.g., '10:30 AM')"
                                                },
                                                "service": {
                                                    "type": "string",
                                                    "description": "Type of dental service",
                                                    "default": "General Checkup"
                                                }
                                            },
                                            "required": ["patient_name", "date", "time"]
                                        }
                                    }
                                },
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "cancel_appointment",
                                        "description": "Cancel an existing appointment",
                                        "parameters": {
                                            "type": "object",
                                            "properties": {
                                                "patient_name": {
                                                    "type": "string",
                                                    "description": "Full name of the patient"
                                                },
                                                "date": {
                                                    "type": "string",
                                                    "description": "Appointment date (YYYY-MM-DD format)"
                                                },
                                                "time": {
                                                    "type": "string",
                                                    "description": "Appointment time (e.g., '10:30 AM')"
                                                }
                                            },
                                            "required": ["patient_name", "date", "time"]
                                        }
                                    }
                                }
                            ]
                        }
                    },
                    "intent": {
                        "min_delay": 0.5,  # Faster response trigger
                        "max_interrupt_delay": 1.5,  # Allow quicker interruptions
                        "max_follow_up_delay": 8.0  # Shorter follow-up window
                    },
                    "audio": {
                        "audio_processing_config": {
                            "echo_cancellation": {"enabled": True},  # Enable for better audio
                            "high_pass_filter": {"enabled": True},
                            "gain_controller2": {"enabled": True, "fixed_digital_gain_db": 6},
                            "noise_suppression": {"enabled": True, "level": 3}  # Max noise suppression
                        }
                    },
                    "tts": {
                        "voice": "af_heart",  # More natural voice
                        "speed": 1.25,  # Slightly faster for efficiency
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
                            # Text message - parse and handle
                            try:
                                data = json.loads(message)
                                message_type = data.get("object")

                                # Handle key message types with clean logging
                                if message_type == "agent.output.transcript":
                                    # Store transcript but don't print immediately (wait for final version)
                                    transcript = data.get("transcript", "")
                                    if transcript:
                                        self.current_transcript = transcript

                                elif message_type == "agent.output.generating":
                                    # Print final user transcript when agent starts generating
                                    if self.current_transcript and self.current_transcript.strip():
                                        self.print_conversation_turn(user_text=self.current_transcript)
                                        self.current_transcript = ""  # Clear after printing

                                elif message_type == "agent.output.done":
                                    # Print final assistant response
                                    response_text = data.get("text", "")
                                    if response_text != self.last_response:
                                        self.last_response = response_text
                                        self.print_conversation_turn(assistant_text=response_text)

                                elif message_type == "agent.output.tool_call":
                                    # Handle function calls
                                    tool_calls = data.get("tool_calls", [])
                                    tool_results = await self.handle_tool_call(tool_calls)
                                    await self.send_tool_result(tool_results)

                                elif message_type == "agent.state.configured":
                                    print("✅ Agent configured and ready!")

                                # Skip logging for noisy message types
                                elif message_type not in [
                                    "agent.output.waiting",
                                    "agent.output.generating",
                                    "agent.output.delta.metadata"
                                ]:
                                    print(f"📨 {message_type}")

                            except json.JSONDecodeError:
                                print(f"📨 Non-JSON message received")

                # Run both send and receive
                await asyncio.gather(send_audio(), receive_messages())

        except Exception as e:
            print(f"❌ Connection error: {e}")
        finally:
            self.websocket = None


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
        print("\n👋 Goodbye!")