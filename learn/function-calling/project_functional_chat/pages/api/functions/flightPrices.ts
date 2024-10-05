import type { NextApiRequest, NextApiResponse } from 'next';

class Api {
  static async spec(req: NextApiRequest, res: NextApiResponse) {
    res.json({
      json_schema: {
        type: 'function',
        function: {
          name: 'flightPrices',
          description: 'Returns flight tickets for specific destinations and dates.',
          parameters: {
            type: 'object',
            properties: {
              origin: {
                type: 'string',
                pattern: '^[A-Z]{3}$',
                description: 'An IATA code of a city or an airport of the origin.'
              },
              destination: {
                type: 'string',
                pattern: '^[A-Z]{3}$',
                description: 'An IATA code of a city or an airport of the destination. Required if "origin" is not specified.'
              },
              departure_at: {
                type: 'string',
                pattern: '^\\d{4}-\\d{2}(-\\d{2})?$',
                description: 'The departure date in "YYYY-MM" or "YYYY-MM-DD" format.'
              },
              return_at: {
                type: 'string',
                pattern: '^\\d{4}-\\d{2}(-\\d{2})?$',
                description: 'The return date in "YYYY-MM" or "YYYY-MM-DD" format. Do not specify for one-way tickets.'
              },
              one_way: {
                type: 'boolean',
                description: 'Indicates if the ticket is one-way (true) or round-trip (false).'
              },
              direct: {
                type: 'boolean',
                default: false,
                description: 'Indicates if only non-stop tickets should be returned. Default is false.'
              },
              sorting: {
                type: 'string',
                enum: ['price', 'route'],
                default: 'price',
                description: 'The sorting method of prices. Default is "price". Options are "price" and "route".'
              }
            },
            required: ['origin']
          }
        }
      }
    });
  }

  static async call(req: NextApiRequest, res: NextApiResponse) {
    // Access additional parameters from req.query
    const { args } = req.query;

    const jsonObj = JSON.parse(args as string);

    const apiKey = process.env.TRAVELPAYOUTS_KEY || 'set the key in .env or .env.local';
    const url = new URL('https://api.travelpayouts.com/aviasales/v3/prices_for_dates');
    if (jsonObj.origin !== undefined) {
      url.searchParams.append('origin', jsonObj.origin);
    }
    if (jsonObj.destination !== undefined) {
      url.searchParams.append('destination', jsonObj.destination);
    }
    if (jsonObj.departureAt !== undefined) {
      url.searchParams.append('departure_at', jsonObj.departureAt);
    }
    if (jsonObj.returnAt !== undefined) {
      url.searchParams.append('return_at', jsonObj.returnAt);
    }
    if (jsonObj.sorting !== undefined) {
      url.searchParams.append('sorting', jsonObj.sorting);
    }
    if (jsonObj.direct !== undefined) {
      url.searchParams.append('direct', jsonObj.direct.toString());
    }
    if (jsonObj.oneWay !== undefined) {
      url.searchParams.append('one_way', jsonObj.oneWay.toString());
    }
    url.searchParams.append('currency', 'USD');
    url.searchParams.append('market', 'us');
    url.searchParams.append('limit', '10');
    url.searchParams.append('token', apiKey);

    try {
      const response = await fetch(url.toString());

      if (!response.ok) {
        throw new Error(`Requesting flight prices with ${args} failed with code ${response.status} message ${await response.text()}`);
      }

      const data = await response.json();

      res.json(data);
    } catch (error) {
      console.error('Error fetching flight prices:', error);
      throw error;
    }
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
