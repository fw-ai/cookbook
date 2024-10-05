import type { NextApiRequest, NextApiResponse } from 'next';

class Api {
  static async spec(req: NextApiRequest, res: NextApiResponse) {
    res.json({
      json_schema: {
        type: 'function',
        function: {
          name: 'weatherHistory',
          description: 'Retrieves daily historical weather records for a given location and month. The temperature unit is Fahrenheit. When processing tool output, do not include links.',
          parameters: {
            type: 'object',
            properties: {
              locations: {
                description: 'Location to get the weather for (must be a full name, no abbreviations).',
                type: 'string'
              },
              month: {
                description: 'Month number. Must be between 1 and 12.',
                type: 'number'
              }
            },
            required: ['locations', 'month']
          }
        }
      }
    });
  }

  static async call(req: NextApiRequest, res: NextApiResponse) {
    // Access additional parameters from req.query
    const { args } = req.query;

    const jsonObj = JSON.parse(args as string);

    if (!jsonObj || typeof jsonObj !== 'object') {
      throw new Error(`Cannot parse weather history arguments: ${args}`);
    }

    const apiKey = process.env.VISUALCROSSING_KEY || 'set the key in .env or .env.local';

    const url = new URL('https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/weatherdata/history');

    url.searchParams.append('key', apiKey);
    url.searchParams.append('aggregateHours', '24');
    url.searchParams.append('unitGroup', 'us');
    url.searchParams.append('contentType', 'json');
    url.searchParams.append('outputDateTimeFormat', 'yyyy-MM-dd');

    if (jsonObj.month !== undefined) {
      const year = new Date().getFullYear() - 1;
      const month = Number(jsonObj.month);
      const lastDay = new Date(year, month, 0).getDate();
      url.searchParams.append('startDateTime', `${year}-${month}-01`);
      url.searchParams.append('endDateTime', `${year}-${month}-${lastDay}`);
    }
    if (jsonObj.locations !== undefined) {
      url.searchParams.append('locations', jsonObj.locations);
    }

    try {
      const response = await fetch(url);

      if (!response.ok) {
        throw new Error(`Requesting historical weather with args ${args} failed with code ${response.status} message ${await response.text()}`);
      }

      let data = await response.json();

      if (data.columns === undefined || data.locations === undefined) {
        return JSON.stringify(data, null);
      }

      const columns = data.columns;
      const transformed: Record<string, any[]> = {};
      Object.keys(data.locations).forEach(locationKey => {
        const location = data.locations[locationKey];
        const values = location.values.map((value: Record<string, any>) => {
          const transformedValue: any = {};
          Object.entries(columns).forEach(([key, column]) => {
            if (key === 'temp' || key === 'precip') {
              const col = column as any;
              transformedValue[col.name] = value[key];
            }
          });
          if (value.datetimeStr !== undefined) {
            transformedValue['datetimeStr'] = value.datetimeStr;
          }
          return transformedValue;
        });
        transformed[locationKey] = values;
      });

      res.json(transformed);
    } catch (error) {
      console.error('Error requesting historical weather:', error);
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
      res.status(404).json({ error: `Action ${action} not found` });
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
