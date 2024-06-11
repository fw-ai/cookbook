import { Carousel, CarouselContent, CarouselItem, CarouselNext, CarouselPrevious } from "@/components/ui/carousel";
import { Button } from "@/components/ui/button";
import * as React from "react";

function ImageModal({ src, alt, onClose }: { src?: string | null; alt?: string; onClose: () => void }) {
    if (!src) return null;

    // Handler to close the modal if the backdrop is clicked
    const handleBackdropClick = (event: React.MouseEvent<HTMLDivElement>) => {
        if (event.currentTarget === event.target) {
            onClose();  // Only close if the click is on the backdrop itself, not on the child elements
        }
    };

    return (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center p-4"
            onClick={handleBackdropClick}>  {/* Added click handler here */}
            <div className="bg-white p-4 rounded-lg shadow-lg max-w-full max-h-full overflow-auto relative">
                <img src={src} alt={alt} className="max-w-full max-h-full" />
                <button onClick={onClose} className="absolute top-0 right-0 m-2 text-xl font-bold text-white">&times;</button>
            </div>
        </div>
    );
}

type ImageData = {
    src: string;
    alt: string;
} | null;


type DocumentPickerProps = {
    onSelectDocument: React.Dispatch<React.SetStateAction<string | null>>;
};

function DocumentPicker({ onSelectDocument }: DocumentPickerProps) {
    const [selectedImage, setSelectedImage] = React.useState<ImageData>(null);


    const documents = {
        "MPRA-POA.pdf": 2,
        "brown-fd.pdf": 10,
        "NY_State_DOH_1.pdf": 12,
    };

    // Function to generate image URLs based on the document name and the number of pages
    const generateImageUrls = (fileName: string, pageCount: number) => {
        return Array.from({ length: pageCount }, (v, i) => `/documents/${fileName}/${i + 1}.jpg`);
    };
    return (
        <>
            <div className="carousel-container flex justify-start items-stretch mx-auto px-4" style={{ maxWidth: '1200px' }}> {/* Adjusted container padding */}
                {Object.entries(documents).map(([docId, pageCount]) => (
                    <div key={docId} className="document-item flex-1 mx-12 min-w-0"> {/* Increased margin for buffer */}
                        <Button onClick={() => onSelectDocument(docId)} className="p-2 bg-blue-500 text-white rounded mb-2 block w-full">
                            Select {docId}
                        </Button>
                        <Carousel className="w-full">
                            <CarouselContent>
                                {generateImageUrls(docId, pageCount).map((imgSrc, index) => (
                                    <CarouselItem key={index}>
                                        <img src={imgSrc} alt={`Document ${docId} Page ${index + 1}`} className="w-full h-auto object-cover" onClick={() => setSelectedImage({ src: imgSrc, alt: `Document ${docId} Page ${index + 1}` })} />
                                    </CarouselItem>
                                ))}
                            </CarouselContent>
                            <CarouselPrevious />
                            <CarouselNext />
                        </Carousel>
                    </div>
                ))}
            </div>
            <ImageModal src={selectedImage?.src} alt={selectedImage?.alt} onClose={() => setSelectedImage(null)} />
        </>

    );
}

export default DocumentPicker;
