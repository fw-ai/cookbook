import type { NextApiRequest, NextApiResponse } from 'next';

class Api {
  static async spec(req: NextApiRequest, res: NextApiResponse) {
    res.json({
      json_schema: {
        type: 'function',
        function: {
          name: 'popularDestinations',
          description: 'Gets the most popular directions from a specified city. Convert tool output to full city names.',
          parameters: {
            type: 'object',
            properties: {
              origin_iata: {
                type: 'string',
                pattern: '^[A-Z]{2,3}$',
                description: 'The point of departure. Must be an IATA city code or a country code, 2 to 3 symbols in length.'
              }
            },
            required: ['origin_iata']
          }
        }
      }
    });
  }

  static async call(req: NextApiRequest, res: NextApiResponse) {
    const { args } = req.query;
    const jsonObj = JSON.parse(args as string);
    if (!jsonObj || typeof jsonObj !== 'object' || !('origin_iata' in jsonObj)) {
      throw new Error(`Cannot parse popular destinations arguments: ${args}`);
    }

    const apiKey = process.env.RAPIDAPI_KEY || 'set the key in .env or .env.local';
    const accessToken = process.env.TRAVELPAYOUTS_KEY || 'set the key in .env or .env.local';

    const url = new URL('https://travelpayouts-travelpayouts-flight-data-v1.p.rapidapi.com/v1/city-directions');
    url.searchParams.append('origin', jsonObj.origin_iata);
    url.searchParams.append('currency', 'USD');

    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'X-Access-Token': accessToken,
        'X-RapidAPI-Key': apiKey,
        'X-RapidAPI-Host': 'travelpayouts-travelpayouts-flight-data-v1.p.rapidapi.com'
      }
    });

    if (!response.ok) {
      throw new Error(`Requesting popular destinations with args ${args} failed with code ${response.status} message ${await response.text()}`);
    }

    let data = await response.json();
    if (data.data === undefined) {
      throw new Error('Missing data in API response.');
    }
    data = data.data;

    // Take the top 10 entries.
    data = Object.keys(data).slice(0, 10).map(key => [key, data[key]]);

    // Extract the relevant fields.
    data = data.map((entry: any) => ({
      destination: entry[1].destination,
      price: entry[1].price
      // flight: `${entry[1].airline}${entry[1].flight_number}`
    }));

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
