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

## Memvid RAG System Comparison

This project also includes scripts for ingesting data into and benchmarking the [Memvid](https://github.com/windsurf-ai/memvid) RAG system. This allows for a comparative analysis of `memvid` against the FAISS-based approach described above. Key scripts for `memvid` interaction include:

*   `ingest_memvid.py`: Ingests text data into `memvid`'s video-based storage.
*   `benchmark_memvid.py`: Runs benchmark queries against a `memvid` instance.

To ensure a fair comparison, the chunking strategy (512 characters per chunk, 50 characters overlap) and embedding model (`all-MiniLM-L6-v2`) are kept consistent across both systems.
A detailed critique and performance comparison can be found in `memvid_critique.md`.

Dependencies (install via `pip install`):
    zstandard, numpy, sentence_transformers, faiss-cpu, typer, tqdm, memvid