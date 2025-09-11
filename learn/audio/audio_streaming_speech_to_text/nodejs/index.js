import fetch from 'node-fetch';
import { spawn } from 'child_process';
import WebSocket from 'ws';

async function main() {
  // 1. Download the FLAC file
  const response = await fetch("https://storage.googleapis.com/fireworks-public/test/3.5m.flac");
  const audioBuffer = Buffer.from(await response.arrayBuffer());
  console.log(`Downloaded audio file size: ${audioBuffer.length} bytes`);

  // 2. Decode the FLAC into raw 16kHz, mono, 16-bit PCM using FFmpeg
  const ffmpeg = spawn('ffmpeg', [
    '-i', 'pipe:0',     // read from stdin
    '-ar', '16000',     // resample to 16000 Hz
    '-ac', '1',         // convert to mono
    '-f', 's16le',      // output raw 16-bit PCM
    'pipe:1'            // write to stdout
  ]);

  let pcmDataChunks = [];

  ffmpeg.stdout.on('data', (chunk) => pcmDataChunks.push(chunk));
  ffmpeg.stderr.on('data', (err) => console.log(err.toString()));

  ffmpeg.on('close', (code) => {
    if (code !== 0) {
      console.error(`FFmpeg exited with code ${code}`);
      return;
    }

    // 2a. Combine PCM chunks into one buffer
    const pcmData = Buffer.concat(pcmDataChunks);
    console.log(`Received decoded PCM data: ${pcmData.length} bytes`);

    // 3. Split the PCM data into 50ms chunks
    //    50ms of 16000 Hz mono => 16000 * 0.05 = 800 samples
    //    Each sample is 2 bytes (16-bit), so 800 * 2 = 1600 bytes
    const CHUNK_SIZE = 1600; // 50ms at 16kHz, mono, 16-bit
    const totalChunks = Math.ceil(pcmData.length / CHUNK_SIZE);
    const audioChunks = [];

    for (let i = 0; i < totalChunks; i++) {
      const start = i * CHUNK_SIZE;
      const end = Math.min(start + CHUNK_SIZE, pcmData.length);
      audioChunks.push(pcmData.slice(start, end));
    }

    console.log(`Split into ${audioChunks.length} chunks of ~50ms each.`);

    // 4. Open WebSocket to the Fireworks streaming endpoint
    const ws = new WebSocket(
      "wss://audio-streaming.us-virginia-1.direct.fireworks.ai/v1/audio/transcriptions/streaming?language=en",
      {
        headers: {
          "Authorization": process.env.FIREWORKS_API_KEY
        }
      }
    );

    ws.on('open', () => {
      console.log("WebSocket connection established.\n");

      // Stream the audio chunks at ~50ms intervals
      let i = 0;
      const interval = setInterval(() => {
        if (i < audioChunks.length) {
          ws.send(audioChunks[i]);
          i++;
        } else {
          clearInterval(interval);
          ws.send(JSON.stringify({ checkpoint_id: "final" }));
        }
      }, 50);
    });

    /*
      5. Handle incoming messages.

      The client maintains a state dictionary, starting with an empty
      dictionary `{}`. When the server sends the first transcription message,
      it contains a list of segments. Each segment has an `id` and `text`:

      Server initial message:
      {
          "segments": [
              {"id": "0", "text": "This is the first sentence"},
              {"id": "1", "text": "This is the second sentence"}
          ]
      }

      Client initial state:
      {
          "0": "This is the first sentence",
          "1": "This is the second sentence",
      }

      When the server sends the next updates to the transcription, the client
      updates the state dictionary based on the segment `id`:

      Server continuous message:
      {
          "segments": [
              {"id": "1", "text": "This is the second sentence modified"},
              {"id": "2", "text": "This is the third sentence"}
          ]
      }

      Client updated state:
      {
          "0": "This is the first sentence",
          "1": "This is the second sentence modified",   # overwritten
          "2": "This is the third sentence",             # new
      }
    */ 
    const state = {};
    ws.on('message', (data) => {
      try {
        const message = JSON.parse(data);
        if (message.checkpoint_id === "final") {
          ws.close();
          return;
        }
        if (message.segments) {
          message.segments.forEach(segment => {
            state[segment.id] = segment.text;
          });
        }
        console.clear();
        for (const [segmentId, text] of Object.entries(state)) {
          console.log(` - ${segmentId}: ${text}`);
        }
      } catch (err) {
        console.error("Error parsing message:", err);
      }
    });

    ws.on('error', (error) => console.error("WebSocket Error:", error));
    ws.on('close', () => console.log("WebSocket closed."));
  });

  ffmpeg.stdin.write(audioBuffer);
  ffmpeg.stdin.end();
}

// Run the main function
main().catch(console.error);
