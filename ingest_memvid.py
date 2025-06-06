import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
from memvid import MemvidEncoder
from tqdm import tqdm

INPUT_DATA_DIR = Path("./input_data")
OUTPUT_MEMVID_DIR = Path("./data_memvid")
MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 512
OVERLAP = 50
MAX_CHUNKS_TO_PROCESS = 1000

def main():
    print(f"Starting memvid ingestion...")
    print(f"Using model: {MODEL_NAME}")
    print(f"Input data directory: {INPUT_DATA_DIR.resolve()}")
    print(f"Output memvid directory: {OUTPUT_MEMVID_DIR.resolve()}")

    OUTPUT_MEMVID_DIR.mkdir(parents=True, exist_ok=True)

    encoder = MemvidEncoder()
    print(f"MemvidEncoder initialized. It will use the default embedding model '{MODEL_NAME}' from its config.")
    print(f"Chunk size for add_text will be {CHUNK_SIZE}, overlap {OVERLAP}.")

    text_files = list(INPUT_DATA_DIR.glob("**/*.txt"))
    if not text_files:
        print(f"No .txt files found in {INPUT_DATA_DIR}. Exiting.")
        return

    print(f"Found {len(text_files)} text files to process.")

    for file_path in tqdm(text_files, desc="Adding text files to encoder"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            if content.strip(): # Ensure content is not just whitespace
                encoder.add_text(content, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
            else:
                print(f"Skipping empty or whitespace-only file: {file_path.name}")
            
            if len(encoder.chunks) >= MAX_CHUNKS_TO_PROCESS:
                print(f"\nReached approximately {MAX_CHUNKS_TO_PROCESS} chunks ({len(encoder.chunks)} actual). Stopping file processing.")
                break
        except Exception as e:
            print(f"Error processing file {file_path.name}: {e}")

    if not encoder.chunks:
        print("No chunks were added to the encoder. Exiting before building video.")
        return

    print(f"Added {len(encoder.chunks)} chunks in total.")
    print("Building video memory and index...")
    video_path = OUTPUT_MEMVID_DIR / "memvid_memory.mp4"
    index_path = OUTPUT_MEMVID_DIR / "memvid_index.json"
    
    encoder.build_video(str(video_path), str(index_path))
    print(f"Successfully built video memory: {video_path.resolve()}")
    print(f"Successfully built index: {index_path.resolve()}")
    print("Memvid ingestion complete.")

if __name__ == "__main__":
    main()
