// app/api/qa/route.ts
import { NextResponse } from 'next/server';

export async function POST(request: Request) {
    const { documentId, question } = await request.json();
    const collection = documentId; // Assuming collection corresponds to documentId

    // Define the OCR_SERVICE_URL
    const OCR_SERVICE_URL = process.env.OCR_SERVICE_URL;

    if (!OCR_SERVICE_URL) {
        return NextResponse.json({ error: 'OCR_SERVICE_URL is not defined' }, { status: 500 });
    }

    // Build the URL with query parameters
    const url = new URL(`${OCR_SERVICE_URL}/v1/collections/${collection}/qa`);
    url.searchParams.append('query', question);

    try {
        // Make the fetch request to the service
        const response = await fetch(url.toString(), {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();

        // Extract the answer from the response
        const answer = data.response;
        return NextResponse.json({ answer });
    } catch (error: any) {
        return NextResponse.json({ error: error.message }, { status: 500 });
    }
}
