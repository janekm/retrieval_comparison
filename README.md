# Ultra‑compact RAG storage + retrieval engine

This script implements, end‑to‑end, the system we designed:

*   **Chunk** documents using character-based splitting (512 characters per chunk, 50 characters overlap) for fair comparison with other systems. Then **compress** chunks into *seekable* Zstandard frames
    (64 KiB raw, dictionary‑trained, level‑19) stored in a single file
    `corpus.zs`.
*   **Build a minimal self‑index** mapping *chunk‑id → (frame‑id,
    offset‑in‑frame, length)* so we can jump straight to the chunk after
    decompressing **one** frame.
*   **Generate embeddings** with MiniLM (384‑d) and write them into a
    FAISS `IndexIVFPQ` (with parameters N_LIST=1024, M_PQ=48, N_BITS=8) for efficient similarity search. The index stores compressed vectors.
*   **Query**: embed the query, search FAISS, decompress the single
    frame(s) needed and return top‑k chunks.

Everything is wrapped in a simple CLI built with **typer** so you can
run:

```bash
python rag_compressed_store.py ingest --input docs.txt --data_dir data/
python rag_compressed_store.py search --query "What is foo?" --data_dir data/ --top_k 5
```

## Data Download and Preparation (`download_and_index.py`)

The `download_and_index.py` script automates the process of fetching source documents and preparing them for ingestion by the `rag_compressed_store.py` system. It performs the following main steps:

1.  **Downloads Data**:
    *   Fetches a configurable number of top Wikipedia articles (default: 1000 via `--num_wiki`) from [Wikipedia:Vital articles/Level/4](https://en.wikipedia.org/wiki/Wikipedia:Vital_articles/Level/4).
    *   Fetches a configurable number of top Project Gutenberg books (default: 10 via `--num_gutenberg`) from the [Gutenberg Top 100](https://www.gutenberg.org/browse/scores/top).
    *   Downloaded raw text files are saved individually into the `input_data/` directory (configurable via `--input_source_dir`).
2.  **Triggers Indexing**:
    *   Automatically invokes `python rag_compressed_store.py ingest` to process the downloaded files from the `input_data/` directory.
    *   The `rag_compressed_store.py` script then builds the compressed corpus (`corpus.zs`), FAISS index (`index.faiss`), and other necessary files in the `data/` directory (configurable via `--output_data_dir` for `download_and_index.py`, which passes it as `--data_dir` to `rag_compressed_store.py`).

### Usage

To download the default number of documents and then trigger indexing:
```bash
python download_and_index.py
```

You can customize the number of articles and books, or skip certain stages using command-line arguments:
```bash
# Download 500 Wikipedia articles and 5 Gutenberg books, then index
python download_and_index.py --num_wiki 500 --num_gutenberg 5

# Skip downloading and only run indexing (assumes data is already in input_data/)
python download_and_index.py --skip_download

# Only download data, skip the indexing step
python download_and_index.py --skip_ingest

# Specify custom directories
python download_and_index.py --input_source_dir custom_input/ --output_data_dir custom_output/
```
The script ensures the necessary directories (`input_data/` and `data/` by default) are created.


## Memvid RAG System Comparison

This project also includes scripts for ingesting data into and benchmarking the [Memvid](https://github.com/windsurf-ai/memvid) RAG system. This allows for a comparative analysis of `memvid` against the FAISS-based approach described above. Key scripts for `memvid` interaction include:

*   `ingest_memvid.py`: Ingests text data into `memvid`'s video-based storage.
*   `benchmark_memvid.py`: Runs benchmark queries against a `memvid` instance.

To ensure a fair comparison, the chunking strategy (512 characters per chunk, 50 characters overlap) and embedding model (`all-MiniLM-L6-v2`) are kept consistent across both systems.
A detailed critique and performance comparison can be found in `memvid_critique.md`.

Dependencies (install via `pip install`):
    zstandard, numpy, sentence_transformers, faiss-cpu, typer, tqdm, memvid