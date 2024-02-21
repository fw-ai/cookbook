import type { NextApiRequest, NextApiResponse } from 'next';

class Api {
  static async spec(req: NextApiRequest, res: NextApiResponse) {
    res.json({
      json_schema: {
        type: 'function',
        function: {
          name: 'generateImage',
          description: 'Generates an image from a text description. The tool output should be displayed as ![text](image_url) where the image url includes blob: prefix.',
          parameters: {
            type: 'object',
            properties: {
              prompt: {
                description: "description of the image",
                type: 'string'
              },
              negative_prompt: {
                description: "concepts that should be excluded from the generated image",
                type: 'string'
              }
            },
            required: [
              'prompt'
            ]
          }
        }
      }
    });
  }

  static async call(req: NextApiRequest, res: NextApiResponse) {
    // Access additional parameters from req.query
    const { args } = req.query;

    const jsonObj = JSON.parse(args as string);

    if (!jsonObj || typeof jsonObj !== 'object' || !('prompt' in jsonObj)) {
      throw new Error(`Cannot parse generate image arguments: ${args}`);
    }

    const prompt = jsonObj.prompt;
    const negativePrompt: string | undefined = jsonObj.negative_prompt;

    const baseApiUrl = 'https://api.fireworks.ai/inference/v1/image_generation';
    const modelId = process.env.FIREWORKS_IMAGE_GEN_MODEL;
    const apiKey = process.env.FIREWORKS_API_KEY;
    const response = await fetch(`${baseApiUrl}/${modelId}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'image/jpeg',
        'Authorization': `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        cfg_scale: 7,
        height: 1024,
        width: 1024,
        sampler: null,
        samples: 1,
        steps: 30,
        seed: 0,
        style_preset: null,
        safety_check: false,
        prompt: prompt,
        negative_prompt: negativePrompt,
      }),
    });

    if (response === null || !response.ok) {
      throw new Error((await response?.text()) ?? 'Something went wrong generating the image. Please try again');
    }

    // Get Content-Type from the response and set it in the Next.js API response
    const contentType = response.headers.get('Content-Type');
    if (contentType) {
      res.setHeader('Content-Type', contentType);
    }
    const buffer = await response.arrayBuffer();
    res.send(Buffer.from(buffer));
  }
}

export default async function handler(req: NextApiRequest, res: NextApiResponse): Promise<any> {
  const { action } = req.query;

  try {
    if (action === 'spec') {
      return Api.spec(req, res);
    } else if (action === 'call') {
      return Api.call(req, res);
    } else {
      res.status(404).json({ error: 'Action not found' });
    }
  } catch (error) {
    console.error(error);
    if (error instanceof Error) {
      res.status(500).json({ error: error.message });
    } else {
      res.status(500).json({ error: 'An internal server error occurred' });
    }
  }
}
