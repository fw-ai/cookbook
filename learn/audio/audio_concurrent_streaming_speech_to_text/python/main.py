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
from typing import Iterator, Tuple, Dict
from dotenv import load_dotenv

load_dotenv()


# Configuration
AUDIO_URL = "https://storage.googleapis.com/fireworks-public/test/3.5m.flac"
WEBSOCKET_URL = "ws://audio-streaming.us-virginia-1.direct.fireworks.ai/v1/audio/transcriptions/streaming"
TARGET_SAMPLE_RATE = 16000
CHUNK_SIZE_MS = 50
LANGUAGE = "en"


NUM_CONCURRENT_STREAMS = 50
EXPAND_DELAY_S = 1 # Delay between starting streams
LAST_N_CHARS = 50


# ============================================================================
# UTILITIES - Load audio, monitor streams, shutdown
# ============================================================================

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

# Global stats
stats = {'completed': 0, 'failed': 0}
stats_lock = threading.Lock()

# Non-blocking data collection
latency_queue = queue.Queue()
text_queue = queue.Queue()
display_latencies: Dict[int, float] = {}
display_texts: Dict[int, str] = {}

# Shutdown control
shutdown_event = threading.Event()
active_websockets = []
websocket_lock = threading.Lock()

def print_streams(period: float = 1):
    while not shutdown_event.is_set():
        time.sleep(period)
        
        try:
            while True:
                stream_id, latency = latency_queue.get_nowait()
                display_latencies[stream_id] = latency
        except queue.Empty:
            pass
            
        try:
            while True:
                stream_id, text = text_queue.get_nowait()
                display_texts[stream_id] = text
        except queue.Empty:
            pass
        
        # Only print if there are active streams
        if display_latencies:
            print(f"\n{'='*80}")
            for stream_id in range(NUM_CONCURRENT_STREAMS):
                if stream_id in display_latencies:
                    latency = display_latencies[stream_id]
                    if stream_id in display_texts:
                        text = display_texts[stream_id]
                        print(f"[Stream {stream_id:>3d}] [Latency {latency:.3f}] {text}")
                    else:
                        print(f"[Stream {stream_id:>3d}] [Latency {latency:.3f}]")
                else:
                    # Stream hasn't started yet
                    print(f"[Stream {stream_id:>3d}] [Latency   ---] [Not started]")
            print(f"{'='*80}")


def shutdown(signum, frame):
    """Handle shutdown on signal."""
    print(f"\nReceived signal {signum}. Starting shutdown...")
    shutdown_event.set()
    
    # Close all active WebSocket connections
    with websocket_lock:
        print(f"Closing {len(active_websockets)} active WebSocket connections...")
        for ws in active_websockets:
            try:
                ws.close()
            except Exception as e:
                print(f"Error closing WebSocket: {e}")
    
    # Wait for connections to close
    time.sleep(2)
    print("Shutdown completed.")
    sys.exit(0)

# ============================================================================
# Audio Streaming
# ============================================================================            

def run_websocket_client(stream_id: int, audio_stream: Iterator[Tuple[bytes, float]]):
    """
    Send audio chunks over WebSocket for streaming transcription.
    """
    recent_latencies = []
    ws = None

    def on_open(websocket_conn):
        nonlocal ws
        ws = websocket_conn
        
        # Register this WebSocket for shutdown
        with websocket_lock:
            active_websockets.append(ws)
        
        def stream_audio(websocket_conn):
            # Stream each chunk, then sleep for chunk duration
            for chunk_idx, (audio_chunk, duration) in enumerate(audio_stream):
                if shutdown_event.is_set():
                    break
                    
                websocket_conn.send(audio_chunk, opcode=websocket.ABNF.OPCODE_BINARY)
                
                # Send latency trace every 10 chunks
                if (chunk_idx % 10) == 0:
                    trace = json.dumps({"event_id": "test", "object": "stt.input.trace", "trace_id": str(time.time())})
                    websocket_conn.send(trace, opcode=websocket.ABNF.OPCODE_TEXT)
                
                time.sleep(duration)

            # Send final trace if not shutting down
            if not shutdown_event.is_set():
                trace = json.dumps({"event_id": "test", "object": "stt.input.trace", "trace_id": "final"})
                websocket_conn.send(trace, opcode=websocket.ABNF.OPCODE_TEXT)
                
                # Give the server some time to finalize any last transcription segments
                time.sleep(10)
            
            # Close WebSocket
            websocket_conn.close()

        threading.Thread(target=stream_audio, args=(websocket_conn,)).start()

    def on_error(websocket_conn, error):
        with stats_lock:
            stats['failed'] += 1
        
        # Remove from active list
        with websocket_lock:
            if websocket_conn in active_websockets:
                active_websockets.remove(websocket_conn)

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
                    latency_queue.put_nowait((stream_id, avg_latency))
            return
        
        text = response.get("text", "")
        if text:
            text_queue.put_nowait((stream_id, text[-LAST_N_CHARS:]))

    def on_close(websocket_conn, close_status_code, close_msg):
        with stats_lock:
            stats['completed'] += 1
        
        # Remove from active list
        with websocket_lock:
            if websocket_conn in active_websockets:
                active_websockets.remove(websocket_conn)
    

    
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
        with stats_lock:
            stats['failed'] += 1
        with websocket_lock:
            if ws in active_websockets:
                active_websockets.remove(ws)

def main():
    
    audio_chunks = process_audio(AUDIO_URL)

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    
    threading.Thread(target=print_streams, daemon=True).start()
    
    print(f"Starting {NUM_CONCURRENT_STREAMS} streams...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_CONCURRENT_STREAMS) as executor:
        futures = []
        
        # Start all streams gradually with delays
        for i in range(NUM_CONCURRENT_STREAMS):
            if i > 0:  
                time.sleep(EXPAND_DELAY_S)
            future = executor.submit(run_websocket_client, i, audio_chunks)
            futures.append(future)
        
        try:
            for future in concurrent.futures.as_completed(futures):
                future.result()
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received. Starting shutdown...")
            shutdown_event.set()
        
        print(f"\nFinal: {stats['completed']} completed, {stats['failed']} failed")

if __name__ == "__main__":
    main()