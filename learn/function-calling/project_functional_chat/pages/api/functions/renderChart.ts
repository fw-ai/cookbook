import type { NextApiRequest, NextApiResponse } from 'next';

class Api {
  static async spec(req: NextApiRequest, res: NextApiResponse) {
    res.json({
      json_schema: {
        type: 'function',
        function: {
          name: 'renderChart',
          description: 'Generates chart from numeric data. The chart is rendered by Chart.js, a popular open-source charting library. The tool output should be displayed as ![text](image_url) where the image url includes blob: prefix.',
          parameters: {
            type: 'object',
            properties: {
              type: {
                type: 'string',
                description: 'type of the chart'
              },
              data: {
                type: 'object',
                description: 'labels for the chart data',
                properties: {
                  labels: {
                    type: 'array',
                    items: {
                      type: 'string'
                    }
                  },
                  datasets: {
                    type: 'array',
                    description: 'data points for the dataset',
                    items: {
                      type: 'object',
                      properties: {
                        label: {
                          type: 'string',
                          description: 'the label for the dataset which appears in the legend and tooltips'
                        },
                        data: {
                          type: 'array',
                          items: {
                            type: 'number'
                          }
                        }
                      },
                      required: ['data']
                    }
                  }
                },
                required: ['labels', 'datasets']
              }
            },
            required: ['type', 'data']
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
      throw new Error(`Cannot parse render chart arguments: ${args}`);
    }
    const chartJson = encodeURIComponent(JSON.stringify(jsonObj));
    const url = `https://quickchart.io/chart?c=${chartJson}`;

    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Error fetching image for args ${args}: ${response.statusText}`);
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
