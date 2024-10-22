// page.tsx
"use client"

import { useState } from "react";
import Image from 'next/image';  // Import Next.js Image component
import DocumentPicker from '../components/DocumentPicker';
import QASection from '../components/QASection';

export default function Home() {
    const [selectedDocumentId, setSelectedDocumentId] = useState<string | null>(null);

    return (
        <div className="flex flex-col min-h-screen">
            <div className="flex-grow flex flex-col items-center justify-center">
                <h1 className="text-3xl font-bold mb-4">Document Search Engine</h1>
                <QASection documentId={selectedDocumentId} />
                <div className="mt-8">  {/* Add margin-top here */}
                    <DocumentPicker onSelectDocument={setSelectedDocumentId} />
                </div>
            </div>
            <footer className="bg-gray-100 w-full py-4 mt-auto"> {/* Footer with a gray background */}
                <div className="max-w-screen-lg mx-auto text-center">
                    <p className="text-lg mb-2">Powered by:</p>
                    <div className="flex justify-center items-center gap-4">
                        <Image src="/mongodb-logo.png" alt="MongoDB Logo" width={100} height={60} />
                        <Image src="/fireworks-logo.png" alt="Fireworks.ai Logo" width={200} height={60} />
                    </div>
                </div>
            </footer>
        </div>
    );
}
