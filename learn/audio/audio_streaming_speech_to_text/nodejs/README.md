# [Node.js] Streaming Audio Transcription

This is a Node.js project that streams audio to a WebSocket API to receive transcription live. It uses FFmpeg for audio processing and handles communication with a WebSocket server.

---

## Prerequisites

1. **Install Node.js**: Download and install Node.js from [Node.js Official Website](https://nodejs.org/).
   - Verify installation with:
     ```bash
     node -v
     npm -v
     ```

2. **Install FFmpeg**: FFmpeg is required to decode the audio.
   - Installation:
     - **macOS**: `brew install ffmpeg`
     - **Linux** (Debian/Ubuntu): `sudo apt-get install ffmpeg`
     - **Windows**: Download from [FFmpeg Official Website](https://ffmpeg.org/) and add it to your PATH.
   - Verify FFmpeg installation:
     ```bash
     ffmpeg -version
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
   - `index.js`: Your main script file.
   - `package.json`: Your dependency configuration file.

2. Navigate to the project directory and install the required Node.js dependencies:
   ```bash
   npm install
   ```

---

## Usage

Run the script with Node.js:
```bash
node index.js
```

---

## Conclusion

In this project, you learned how to stream audio to the Streaming Speech-to-Text API and receive the transcription in real-time over a websocket connection.

For more information visit [docs.fireworks.ai](https://www.google.com/url?q=https%3A%2F%2Fdocs.fireworks.ai%2Fapi-reference%2Faudio-streaming-transcriptions).

Explore the community or reach out to us in [discord](https://www.google.com/url?q=https%3A%2F%2Fdiscord.gg%2Ffireworks-ai).

