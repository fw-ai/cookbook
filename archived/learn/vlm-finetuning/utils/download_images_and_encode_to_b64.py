import json
import base64
import requests
import mimetypes
from pathlib import Path
from typing import Dict, Any
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_and_encode_image(url: str, timeout: int = 30, max_retries: int = 3) -> str:
    """
    Download an image from URL and convert it to base64 with MIME type prefix.
    
    Args:
        url: Image URL to download
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        
    Returns:
        Base64 encoded string with MIME type prefix
        
    Raises:
        Exception: If download fails or image format is unsupported
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            
            # Get content type from response headers
            content_type = response.headers.get('content-type', '')
            
            # If content-type not available, try to guess from URL
            if not content_type or not content_type.startswith('image/'):
                guessed_type, _ = mimetypes.guess_type(url)
                if guessed_type and guessed_type.startswith('image/'):
                    content_type = guessed_type
                else:
                    # Default to jpeg if we can't determine
                    content_type = 'image/jpeg'
            
            # Encode image to base64
            encoded_image = base64.b64encode(response.content).decode('utf-8')
            
            # Return with proper MIME type prefix
            return f"data:{content_type};base64,{encoded_image}"
            
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                # Exponential backoff: 1, 2, 4 seconds
                wait_time = 2 ** attempt
                print(f"  Retry {attempt + 1}/{max_retries} in {wait_time}s...")
                time.sleep(wait_time)
            else:
                break
    
    raise Exception(f"Failed to download/encode image from {url} after {max_retries + 1} attempts: {str(last_exception)}")

def download_single_image(url: str, processed_urls: Dict[str, str]) -> tuple[str, str]:
    """Download a single image and return (url, base64_data) tuple."""
    if url in processed_urls:
        return url, processed_urls[url]
    
    print(f"Processing image: {url[:80]}...")
    try:
        base64_url = download_and_encode_image(url)
        print("✓ Successfully encoded")
        return url, base64_url
    except Exception as e:
        print(f"✗ Error: {e}")
        return url, None

def process_message_content(content, processed_urls: Dict[str, str], max_workers: int = 5):
    """Process message content and convert image URLs to base64."""
    if isinstance(content, str):
        return content
    
    if isinstance(content, list):
        # Collect all image URLs that need processing
        urls_to_process = []
        for item in content:
            if isinstance(item, dict) and item.get('type') == 'image_url':
                image_url = item.get('image_url', {}).get('url', '')
                if not image_url.startswith('data:') and image_url not in processed_urls:
                    urls_to_process.append(image_url)
        
        # Process images in parallel
        if urls_to_process:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_url = {
                    executor.submit(download_single_image, url, processed_urls): url 
                    for url in urls_to_process
                }
                
                for future in as_completed(future_to_url):
                    url, base64_data = future.result()
                    if base64_data:
                        processed_urls[url] = base64_data
        
        # Now update the content with processed URLs
        updated_content = []
        for item in content:
            if isinstance(item, dict) and item.get('type') == 'image_url':
                image_url = item.get('image_url', {}).get('url', '')
                
                # Skip if already base64 encoded
                if image_url.startswith('data:'):
                    updated_content.append(item)
                    continue
                
                # Use processed result if available
                if image_url in processed_urls:
                    updated_item = item.copy()
                    updated_item['image_url']['url'] = processed_urls[image_url]
                    updated_content.append(updated_item)
                # Skip if processing failed
            else:
                updated_content.append(item)
        
        return updated_content
    
    return content

def convert_dataset_urls_to_base64(input_file: str, output_file: str = None, max_workers: int = 5):
    """Convert image URLs in a JSONL dataset file to base64 encoded strings."""
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    if output_file is None:
        output_file = str(input_path.with_stem(f"{input_path.stem}_base64"))
    
    processed_urls = {}
    processed_count = 0
    error_count = 0
    
    print(f"Converting image URLs to base64 in: {input_file}")
    print(f"Output will be saved to: {output_file}")
    print("-" * 50)
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
        open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            try:
                data = json.loads(line.strip())
                
                if 'messages' in data:
                    for message in data['messages']:
                        if 'content' in message:
                            original_content = message['content']
                            updated_content = process_message_content(original_content, processed_urls, max_workers)
                            message['content'] = updated_content
                
                json.dump(data, outfile, ensure_ascii=False)
                outfile.write('\n')
                processed_count += 1
                
                if line_num % 10 == 0:
                    print(f"Processed {line_num} lines...")
                    
            except Exception as e:
                print(f"✗ Error processing line {line_num}: {e}")
                error_count += 1
                outfile.write(line)
    
    print("-" * 50)
    print(f"✓ Conversion complete!")
    print(f"  - Lines processed: {processed_count}")
    print(f"  - Unique images converted: {len(processed_urls)}")
    print(f"  - Errors: {error_count}")
    print(f"  - Output saved to: {output_file}")

# Usage: python download_images_and_encode_to_b64.py --input_file input_file.jsonl [--output_file output_file.jsonl]
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert image URLs in JSONL dataset to base64 encoded strings")
    parser.add_argument("--input_file", required=True, help="Input JSONL file with image URLs")
    parser.add_argument("--output_file", help="Output JSONL file (default: adds '_base64' suffix to input filename)")
    parser.add_argument("--max_workers", type=int, default=15, help="Maximum number of concurrent downloads (default: 5)")
    
    args = parser.parse_args()
    
    try:
        convert_dataset_urls_to_base64(args.input_file, args.output_file, args.max_workers)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)