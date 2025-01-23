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
    const segments = {};

    const ws = new WebSocket(
      "ws://audio-streaming.us-virginia-1.direct.fireworks.ai/v1/audio/transcriptions/streaming?language=en",
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
          // Allow some time for final processing on server side
          setTimeout(() => {
            ws.close();
          }, 2000);
        }
      }, 50); // 50ms interval
    });

    /*
      5. Handle incoming messages with real-time transcription.
      Segments state dictionary works helps to keep track of the segments and their updates.
      
      For example:
      1. Segments before an update:

        {
          "0": "sentence0",
          "1": "sentence1",
          "2": "sentence2",
        }

      2. A new message with an update is received (on_message):

        {
            "1": "sentence1",
            "2": "sentence2_updated",
        }

      3. Segments state after the update:

        {
            "0": "sentence0",
            "1": "sentence1",
            "2": "sentence2_updated",
        }
    */ 
    ws.on('message', (data) => {
      try {
        const message = JSON.parse(data);
        if (message.segments) {
          message.segments.forEach(segment => {
            segments[segment.id] = segment.text;
          });
        }
        console.clear();
        for (const [segmentId, text] of Object.entries(segments)) {
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
