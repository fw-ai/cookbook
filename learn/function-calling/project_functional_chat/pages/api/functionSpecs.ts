import type { NextApiRequest, NextApiResponse } from 'next';

function createMockRequest(queryParams: Record<string, any> = {}): NextApiRequest {
  return {
    query: queryParams
  } as NextApiRequest;
}

interface MockNextApiResponse extends NextApiResponse {
  data?: any;
}

function createMockResponse(): MockNextApiResponse {
  const res: Partial<MockNextApiResponse> = {};

  res.json = (data: any) => {
    res.data = data;
    return res as MockNextApiResponse;
  };

  return res as MockNextApiResponse;
}

export async function fetchFunctionSpecs(endpoints: string[]) {
  const specs = await Promise.all(endpoints.map(async (func) => {
    const importedModule = await import(`./functions/${func}.ts`);
    const req = createMockRequest({ action: 'spec' });
    const res = createMockResponse();
    await importedModule.default(req, res);
    return res.data.json_schema;
  }));

  return specs;
}

export default async function handler(req: NextApiRequest, res: NextApiResponse): Promise<any> {
  const endpointsStr = process.env.ACTIVE_FUNCTIONS;
  const endpoints = endpointsStr ? endpointsStr.split(',') : [];
  const specs = await fetchFunctionSpecs(endpoints);
  res.json(specs);
}
