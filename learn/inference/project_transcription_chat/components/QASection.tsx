import { useState, useRef } from 'react';
import axios from 'axios';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

interface QASectionProps {
    documentId: string | null;
}

type Timeout = ReturnType<typeof setTimeout>;



function QASection({ documentId }: QASectionProps) {
    const [question, setQuestion] = useState('');
    const [answer, setAnswer] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [recording, setRecording] = useState(false);
    const audioRecorder = useRef<MediaRecorder | null>(null); // Specify the type here
    const audioChunks = useRef<Blob[]>([]);

    const handleAudioStart = async () => {
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                audioRecorder.current = new MediaRecorder(stream);
                audioChunks.current = [];

                audioRecorder.current.ondataavailable = (event) => {
                    audioChunks.current.push(event.data);
                };

                audioRecorder.current.onstop = async () => {
                    const audioBlob = new Blob(audioChunks.current, { type: 'audio/mp3' });
                    const formData = new FormData();
                    formData.append('file', audioBlob, 'question.mp3');

                    const response = await axios.post('https://dev.api.fireworks.ai/inference/v1/audio/translations', formData, {
                        headers: {
                            'Authorization': `Bearer ${process.env.NEXT_PUBLIC_FIREWORKS_API_KEY}`,
                        }
                    });
                    console.log("Show response data", response.data);

                    if (response.data && response.data.text) {
                        setQuestion(response.data.text);
                        handleQuestionSubmit(response.data.text); // Automatically submit the question
                    }
                    setRecording(prevRecording => !prevRecording);
                };

                audioRecorder.current.start();
                setRecording(true);

            } catch (err) {
                console.error('Error accessing the microphone', err);
            }
        }
    };

    const handleAudioStop = () => {
        if (audioRecorder.current && recording) {
            audioRecorder.current.stop();
        }
    };

    const handleQuestionSubmit = async (event: any) => {
        setLoading(true);
        setError(null);

        try {
            const response = await axios.post('/api/qa', { documentId, question });
            setAnswer(response.data.answer);
        } catch (err) {
            setError('Failed to fetch the answer. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
            <h3 className="text-xl mb-4">Ask a question about the document:</h3>
            <div className="flex items-center gap-2 w-full">
                <Input
                    type="text"
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    className="p-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                    placeholder={!documentId ? "Type your question here" : `What do you want to know about document ${documentId}?`}
                    disabled={!documentId}
                    style={{ flex: '3' }}  // Takes up 3 parts of the space
                />
                <Button
                    onClick={handleQuestionSubmit}
                    className="px-4 py-2 bg-blue-500 text-white rounded"
                    disabled={!documentId || loading}
                    style={{ flex: '1' }}  // Takes up 1 part of the space
                >
                    {loading ? 'Submitting...' : 'Submit'}
                </Button>
                <Button
                    onClick={recording ? handleAudioStop : handleAudioStart}
                    className="px-4 py-2 bg-red-500 text-white rounded"
                    disabled={!documentId || loading}
                    style={{ flex: '1' }}
                >
                    {recording ? 'Stop Recording' : 'Voice Search'}
                </Button>
            </div>
            <div className="mt-4">
                {error && <p className="text-red-500">{error}</p>}
                <p className="text-gray-700">{answer}</p>
            </div>
        </div>
    );
}

export default QASection;
