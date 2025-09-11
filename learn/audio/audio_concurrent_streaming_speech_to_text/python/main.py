import io
import os
import requests
import threading
import concurrent.futures
import queue
import torch
import torchaudio
import json
import websocket
import time
import urllib.parse
import signal
import sys
from typing import Dict
from dotenv import load_dotenv

load_dotenv()


# Configuration
AUDIO_URL = "https://storage.googleapis.com/fireworks-public/test/3.5m.flac"
WEBSOCKET_URL = "wss://audio-streaming.us-virginia-1.direct.fireworks.ai/v1/audio/transcriptions/streaming"
TARGET_SAMPLE_RATE = 16000
CHUNK_SIZE_MS = 50
LANGUAGE = "en"


NUM_CONCURRENT_STREAMS = 50
EXPAND_DELAY_S = 1 # Delay between starting streams
LAST_N_CHARS = 50

class AudioStreaming:
    """Handles concurrent real-time audio transcription via WebSocket streaming."""

    def __init__(self):
        # Global stats
        self.stats = {'completed': 0, 'failed': 0}
        self.stats_lock = threading.Lock()

        # Non-blocking data collection
        self.latency_queue = queue.Queue()
        self.text_queue = queue.Queue()
        self.display_latencies: Dict[int, float] = {}
        self.display_texts: Dict[int, str] = {}

        # Shutdown control
        self.shutdown_event = threading.Event()
        self.active_websockets = []
        self.websocket_lock = threading.Lock()


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


    def print_streams(self, period: float = 1):
        while not self.shutdown_event.is_set():
            time.sleep(period)
            
            try:
                while True:
                    stream_id, latency = self.latency_queue.get_nowait()
                    self.display_latencies[stream_id] = latency
            except queue.Empty:
                pass
                
            try:
                while True:
                    stream_id, text = self.text_queue.get_nowait()
                    self.display_texts[stream_id] = text
            except queue.Empty:
                pass
            
            # Only print if there are active streams
            if self.display_latencies:
                print(f"\n{'='*80}")
                for stream_id in range(NUM_CONCURRENT_STREAMS):
                    if stream_id in self.display_latencies:
                        latency = self.display_latencies[stream_id]
                        if stream_id in self.display_texts:
                            text = self.display_texts[stream_id]
                            print(f"[Stream {stream_id:>3d}] [Latency {latency:.3f}] {text}")
                        else:
                            print(f"[Stream {stream_id:>3d}] [Latency {latency:.3f}]")
                    else:
                        # Stream hasn't started yet
                        print(f"[Stream {stream_id:>3d}] [Latency   ---] [Not started]")
                print(f"{'='*80}")


    def shutdown(self, signum, frame):
        """Handle shutdown on signal."""
        print(f"\nReceived signal {signum}. Starting shutdown...")
        self.shutdown_event.set()
        
        # Close all active WebSocket connections
        with self.websocket_lock:
            print(f"Closing {len(self.active_websockets)} active WebSocket connections...")
            for ws in self.active_websockets:
                try:
                    ws.close()
                except Exception as e:
                    print(f"Error closing WebSocket: {e}")
        
        # Wait for connections to close
        time.sleep(2)
        print("Shutdown completed.")
        sys.exit(0)

    def run_websocket_client(self, stream_id: int):
        """
        Send audio chunks over WebSocket for streaming transcription.
        """
        recent_latencies = []
        ws = None

        def on_open(websocket_conn):
            nonlocal ws
            ws = websocket_conn
            
            # Register this WebSocket for shutdown
            with self.websocket_lock:
                self.active_websockets.append(ws)

            def stream_audio(websocket_conn):
                """Stream audio chunks to the WebSocket."""

                for i, chunk in enumerate(self.audio_chunks):
                    if self.shutdown_event.is_set():
                        break
                    ws.send(chunk, opcode=websocket.ABNF.OPCODE_BINARY)
                    
                    # Send latency trace every 10 chunks
                    if (i % 10) == 0:
                        trace = json.dumps({"event_id": "test", "object": "stt.input.trace", "trace_id": str(time.time())})
                        websocket_conn.send(trace, opcode=websocket.ABNF.OPCODE_TEXT)
                                    
                    time.sleep(CHUNK_SIZE_MS / 1000)  # Maintain real-time pace

                if not self.shutdown_event.is_set():
                    final_trace = json.dumps({
                        "event_id": "streaming_complete",
                        "object": "stt.input.trace",
                        "trace_id": "final"
                    })
                    ws.send(final_trace, opcode=websocket.ABNF.OPCODE_TEXT)
                    
                    print("Audio streaming complete - final trace sent")
                    time.sleep(10)

                websocket_conn.close()

            threading.Thread(target=stream_audio, args=(websocket_conn,)).start()

        def on_error(websocket_conn, error):
            with self.stats_lock:
                self.stats['failed'] += 1
            
            # Remove from active list
            with self.websocket_lock:
                if websocket_conn in self.active_websockets:
                    self.active_websockets.remove(websocket_conn)

        def on_message(websocket_conn, message):
            recv_time = time.time()
            try:
                response = json.loads(message)
            except:
                return  # Skip malformed messages
            
            # Handle application error
            if "error" in response:
                return
            
            # Handle control trace
            if response.get("object") == "stt.output.trace":
                if response.get("trace_id") == "final":
                    pass
                else:
                    # Process latency trace
                    start = float(response.get("trace_id"))
                    recent_latencies.append(recv_time - start)
                    if len(recent_latencies) > 10:
                        recent_latencies.pop(0)
                    
                    if recent_latencies:
                        avg_latency = sum(recent_latencies) / len(recent_latencies)
                        self.latency_queue.put_nowait((stream_id, avg_latency))
                return
            
            text = response.get("text", "")
            if text:
                self.text_queue.put_nowait((stream_id, text[-LAST_N_CHARS:]))

        def on_close(websocket_conn, close_status_code, close_msg):
            print(f"Stream {stream_id}: closed")
            with self.stats_lock:
                self.stats['completed'] += 1
            
            # Remove from active list
            with self.websocket_lock:
                if websocket_conn in self.active_websockets:
                    self.active_websockets.remove(websocket_conn)
        
        params = urllib.parse.urlencode({"language": LANGUAGE})
        full_url = f"{WEBSOCKET_URL}?{params}"

        api_key = os.environ.get("FIREWORKS_API_KEY")
        if not api_key:
            raise ValueError("FIREWORKS_API_KEY environment variable not set")
        
        websocket.setdefaulttimeout(60)
        ws = websocket.WebSocketApp(
            full_url,
            header={"Authorization": api_key},
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )
        
        try:
            ws.run_forever()
        except Exception as e:
            print(f"Stream {stream_id}: Connection failed: {e}")
            with self.stats_lock:
                self.stats['failed'] += 1
            with self.websocket_lock:
                if ws in self.active_websockets:
                    self.active_websockets.remove(ws)

    def run(self, num_concurrent_streams: int = NUM_CONCURRENT_STREAMS, expand_delay_s: float = EXPAND_DELAY_S):
        """Main execution flow."""
        audio_bytes = self.download_audio(AUDIO_URL)
        self.process_audio(audio_bytes)

        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
        
        threading.Thread(target=self.print_streams, daemon=True).start()
        
        print(f"Starting {num_concurrent_streams} streams...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent_streams) as executor:
            futures = []
            
            # Start all streams gradually with delays
            for i in range(num_concurrent_streams):
                if i > 0:  
                    time.sleep(expand_delay_s)
                future = executor.submit(self.run_websocket_client, i)
                futures.append(future)
            
            try:
                for future in concurrent.futures.as_completed(futures):
                    future.result()
            except KeyboardInterrupt:
                print("\nKeyboard interrupt received. Starting shutdown...")
                self.shutdown_event.set()
            
            print(f"\nFinal: {self.stats['completed']} completed, {self.stats['failed']} failed")



def main():
    """Entry point for the concurrent transcription demo."""
    print("🎤 Fireworks Audio Concurrent Transcription Demo")
    print("=" * 40)

    client = AudioStreaming()
    client.run(num_concurrent_streams=NUM_CONCURRENT_STREAMS, expand_delay_s=EXPAND_DELAY_S)

if __name__ == "__main__":
    main()