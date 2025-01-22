# [Python] Streaming Audio Transcription

This is a Python project that streams audio to a WebSocket API to receive transcription in real time. It uses TorchAudio for audio processing and communicates with a WebSocket server.

---

## Prerequisites

1. **Install Python**: Download and install Python 3.8 or later from [Python Official Website](https://www.python.org/).
   - Verify installation with:
     ```bash
     python --version
     pip --version
     ```

2. **Install Dependencies**: Ensure the following libraries are installed:
   - `requests`
   - `torch`
   - `torchaudio`
   - `websocket-client`
     ```bash
     pip install -r requirements.txt
     ```

3. **API Key**: Obtain an API key for the WebSocket server and set it as an environment variable:
   - **macOS/Linux**:
     ```bash
     export FIREWORKS_API_KEY=your_api_key_here
     ```
   - **Windows (Command Prompt)**:
     ```cmd
     set FIREWORKS_API_KEY=your_api_key_here
     ```
   - **Windows (PowerShell)**:
     ```powershell
     $env:FIREWORKS_API_KEY="your_api_key_here"
     ```

---

## Installation

1. Clone or create the project directory and save the following files:
   - `main.py`: The main script for streaming transcription.
   - `requirements.txt`: The list of dependencies.

2. Navigate to the project directory and install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

Run the script:
```bash
python main.py
```

---

## Conclusion

In this project, you learned how to stream audio to the Streaming Speech-to-Text API and receive the transcription in real-time over a websocket connection.

For more information visit [docs.fireworks.ai](https://www.google.com/url?q=https%3A%2F%2Fdocs.fireworks.ai%2Fapi-reference%2Faudio-streaming-transcriptions).

Explore the community or reach out to us in [discord](https://www.google.com/url?q=https%3A%2F%2Fdiscord.gg%2Ffireworks-ai).

