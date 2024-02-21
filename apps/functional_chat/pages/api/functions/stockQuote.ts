import type { NextApiRequest, NextApiResponse } from 'next';

class Api {
  static async spec(req: NextApiRequest, res: NextApiResponse) {
    res.json({
      json_schema: {
        type: 'function',
        function: {
          name: 'stockQuote',
          description: 'Obtains the latest price and volume information for a given stock ticker symbol.',
          parameters: {
            type: 'object',
            properties: {
              symbol: {
                description: "the stock ticker symbol whose price should be quoted",
                type: 'string'
              }
            },
            required: [
              "symbol"
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

    if (!jsonObj || typeof jsonObj !== 'object' || !('symbol' in jsonObj)) {
      throw new Error(`Cannot parse get stock quote arguments: ${args}`);
    }
    const symbol = jsonObj.symbol;

    const apiKey = process.env.ALPHAVANTAGE_KEY;
    const url = `https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=${symbol}&apikey=${apiKey}`;

    const response = await fetch(url);

    if (!response.ok) {
      throw new Error(`Requesting stock for symbol ${symbol} failed with code ${response.status} message ${await response.text()}`);
    }

    const data = await response.json();

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
