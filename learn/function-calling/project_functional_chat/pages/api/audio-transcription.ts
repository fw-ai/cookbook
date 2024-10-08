// pages/api/audio-transcription.js
import { NextApiRequest, NextApiResponse } from 'next';
import FormData from 'form-data';
import fetch from 'node-fetch';

export const config = {
  api: {
    bodyParser: false,
  },
};

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method Not Allowed' });
  }

  try {
    const chunks: Buffer[] = [];

    for await (const chunk of req) {
      chunks.push(chunk);
    }

    const buffer = Buffer.concat(chunks);

    const formData = new FormData();
    formData.append('file', buffer, {
      filename: 'audio.mp3',
      contentType: 'audio/mp3',
    });

    const response = await fetch('https://api.fireworks.ai/inference/v1/audio/transcriptions', {
      method: 'POST',
      body: formData,
      headers: {
        'Authorization': `Bearer ${process.env.FIREWORKS_API_KEY}`,
      },
    });

    if (!response.ok) {
      throw new Error('Error processing audio');
    }

    const data = await response.json();
    res.status(200).json(data);
  } catch (error) {
    console.error('Error processing audio:', error);
    res.status(500).json({ error: 'Error processing audio' });
  }
}
