import type { NextApiRequest, NextApiResponse } from 'next';

class Api {
  static async spec(req: NextApiRequest, res: NextApiResponse) {
    res.json({
      json_schema: {
        type: 'function',
        function: {
          name: 'webSearch',
          description: 'get information from the web',
          parameters: {
            type: 'object',
            properties: {
              query: {
                description: "search query",
                type: 'string'
              }
            },
            required: [
              "query"
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

    if (!jsonObj || typeof jsonObj !== 'object' || !('query' in jsonObj)) {
      throw new Error(`Cannot parse web search arguments: ${args}`);
    }
    const query = jsonObj.query;

    const apiKey = process.env.BING_SEARCH_KEY || 'set the key in .env or .env.local';
    const endpoint = 'https://api.bing.microsoft.com/v7.0/search';
    const resultsCount = 5; // Number of results to return

    const response = await fetch(`${endpoint}?q=${encodeURIComponent(query)}&count=${resultsCount}`, {
      headers: {
        'Ocp-Apim-Subscription-Key': apiKey,
      },
    });

    if (!response.ok) {
      throw new Error(`Error from Bing API: ${response.statusText}`);
    }

    const fullResponse = await response.json();

    // Extract only the desired fields
    const data = {
      webPages: {
        value: fullResponse.webPages.value.map((item: any) => ({
          name: item.name,
          url: item.url,
          snippet: item.snippet
        }))
      }
    };

    res.json(data);
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
