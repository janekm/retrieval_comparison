# rag_compressed_store.py
"""Ultra‑compact RAG storage + retrieval engine

This script implements, end‑to‑end, the system we designed:

*   **Chunk / compress** documents into *seekable* Zstandard frames
    (64 KiB raw, dictionary‑trained, level‑19) stored in a single file
    `corpus.zs`.
*   **Build a minimal self‑index** mapping *chunk‑id → (frame‑id,
    offset‑in‑frame, length)* so we can jump straight to the chunk after
    decompressing **one** frame.
*   **Generate embeddings** with MiniLM (384‑d) and write them into a
    FAISS **OPQ16_64, IVF4096, PQ16** index that keeps bulk codes on
    disk (via `OnDiskInvertedLists`) while holding only the coarse
    centroids (≈ 6 MB) in RAM.
*   **Query**: embed the query, search FAISS, decompress the single
    frame(s) needed and return top‑k chunks.

Everything is wrapped in a simple CLI built with **typer** so you can
run:

```bash
python rag_compressed_store.py ingest --input docs.txt --data_dir data/
python rag_compressed_store.py search --query "What is foo?" --data_dir data/ --top_k 5
```

Dependencies (install via `pip install`):
    zstandard, numpy, sentence_transformers, faiss-cpu, typer, tqdm

For clarity and brevity this file stays < 500 LOC; production hardening
(left as TODO comments) would add structured logging, signal handling,
async IO, etc.
"""

from __future__ import annotations

import typer
import json
import math
import mmap
import os
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import faiss  # type: ignore
import numpy as np
import time
import statistics
from typing import List, Tuple, Dict # Added for benchmark type hints
import tqdm
import zstandard as zstd
from sentence_transformers import SentenceTransformer

# ------------- constants ----------------------------------------------------
FRAME_RAW_SIZE = 64 * 1024  # 64 KiB uncompressed target
DICT_SIZE = 100_000  # bytes for zstd dictionary
ZSTD_LEVEL = 19
CHUNK_SIZE_CHARS = 512  # Character chunk size to match memvid
OVERLAP_CHARS = 50      # Character overlap to match memvid
N_LIST = 2048  # IVF coarse centroids (adjusted from 4096 due to dataset size)
M_PQ = 16# FAISS index params for original complex setup (kept for reference)
_N_LIST_ORIG = 4096
_M_PQ_ORIG = 16
_N_BITS_ORIG = 8
_OPQ_SUBSPACE_ORIG = 16

# FAISS index params for simplified IndexIVFPQ (compromise solution)
N_LIST_IVFPQ = 1024       # Number of IVF cells (Voronoi partitions)
M_PQ_IVFPQ = 48           # Number of sub-quantizers for PQ. Embed dim (384) must be divisible by M_PQ.
                          # 384 / 48 = 8 dimensions per sub-vector.
N_BITS_IVFPQ = 8          # Bits per PQ code (usually 8, meaning 256 centroids per sub-codebook).
                          # Results in M_PQ_IVFPQ bytes per compressed vector (48 bytes here).4  # rotation subspaces
N_BITS = 8  # bits per PQ code

# ---------------------------------------------------------------------------
@dataclass
class ChunkPointer:
    frame_id: int
    start: int  # offset within **decompressed** frame
    length: int

    def to_tuple(self):
        return self.frame_id, self.start, self.length


# ---------------------------------------------------------------------------
class CorpusCompressor:
    """Builds `corpus.zs`, `frames.idx`, and `chunks.idx`."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.corpus_path = self.output_dir / "corpus.zs"
        self.frame_index_path = self.output_dir / "frames.idx"
        self.chunk_index_path = self.output_dir / "chunks.idx"
        self.dict_path = self.output_dir / "dict.zstd"

        self._frame_offsets: List[Tuple[int, int]] = []  # (file_offset, comp_size)
        self._chunk_ptrs: List[ChunkPointer] = []

    # ---------------- dictionary training ----------------------------------
    def _train_dictionary(self, samples: List[bytes]):
        dict_data = zstd.train_dictionary(DICT_SIZE, samples)
        self.dict_path.write_bytes(dict_data.as_bytes())
        return dict_data

    # ---------------- ingestion -------------------------------------------
    def ingest(self, docs: List[str]):
        """Chunks docs, compresses, and writes all aux indexes."""
        # Character-based chunker with overlap (to match memvid)
        chunks: List[str] = []
        for doc_text in tqdm.tqdm(docs, desc="Chunking documents"):
            if not doc_text.strip():
                continue
            
            start = 0
            doc_len = len(doc_text)
            while start < doc_len:
                end = min(start + CHUNK_SIZE_CHARS, doc_len)
                chunk_str = doc_text[start:end]
                
                # Ensure chunk is not just whitespace and meets a minimum length if desired
                # For now, just ensuring it's not empty after stripping.
                if chunk_str.strip():
                    chunks.append(chunk_str)
                
                if end == doc_len: # Reached the end of the document
                    break
                
                # Move start position for the next chunk
                start += (CHUNK_SIZE_CHARS - OVERLAP_CHARS)
                
                # Safety break if overlap is too large or chunk size too small, though unlikely with 512/50
                if start >= end and OVERLAP_CHARS > 0 : 
                    # This case means the next chunk would start at or after the current one ended without advancing
                    # This can happen if CHUNK_SIZE_CHARS <= OVERLAP_CHARS. For 512/50 it's fine.
                    # However, if start is simply >= doc_len after advancing, that's fine, loop will terminate.
                    break
                if start >= doc_len: # Ensure next start is not past the end if we are very close to end
                    break

        # 1‑st pass gather sample bytes for dictionary
        sample_bytes = [c.encode("utf-8") for c in chunks[: 50_000]]
        dict_data = self._train_dictionary(sample_bytes)
        zstd_comp = zstd.ZstdCompressor(level=ZSTD_LEVEL, dict_data=dict_data)

        # streaming write corpus.zs
        with self.corpus_path.open("wb") as corpus_fp:
            frame_buf: List[bytes] = []
            frame_raw_len = 0
            frame_id = 0
            for idx, chunk in enumerate(tqdm.tqdm(chunks, desc="Compress chunks")):
                raw = chunk.encode("utf-8") + b"\n"  # newline separator
                frame_buf.append(raw)
                start_in_frame = frame_raw_len
                frame_raw_len += len(raw)
                # record pointer immediately (will know frame_id later)
                self._chunk_ptrs.append(ChunkPointer(frame_id, start_in_frame, len(raw)))

                if frame_raw_len >= FRAME_RAW_SIZE:
                    self._flush_frame(corpus_fp, b"".join(frame_buf), zstd_comp)
                    frame_buf.clear()
                    frame_raw_len = 0
                    frame_id += 1
            # last partial frame
            if frame_buf:
                self._flush_frame(corpus_fp, b"".join(frame_buf), zstd_comp)

        # save indexes
        self._save_indexes()
        print("[OK] Compression finished.\n  frames: {}\n  chunks: {}".format(len(self._frame_offsets), len(self._chunk_ptrs)))

    def _flush_frame(self, fp, raw_frame: bytes, comp: zstd.ZstdCompressor):
        """Compress one frame and append to corpus file."""
        file_offset = fp.tell()
        comp_bytes = comp.compress(raw_frame)
        fp.write(comp_bytes)
        self._frame_offsets.append((file_offset, len(comp_bytes)))

    def _save_indexes(self):
        np.array(self._frame_offsets, dtype=np.uint64).tofile(self.frame_index_path)
        # chunk ptr struct: 3 uint32 (frame_id, start, len) => 12 bytes each
        arr = np.zeros((len(self._chunk_ptrs), 3), dtype=np.uint32)
        for i, ptr in enumerate(self._chunk_ptrs):
            arr[i] = ptr.to_tuple()
        arr.tofile(self.chunk_index_path)


# ---------------------------------------------------------------------------
class TextStore:
    """Memory‑maps indexes and supports random chunk retrieval."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.corpus_path = data_dir / "corpus.zs"
        self.frame_index_path = data_dir / "frames.idx"
        self.chunk_index_path = data_dir / "chunks.idx"
        self.dict_path = data_dir / "dict.zstd"

        # mmap indexes
        self.frame_offsets = np.fromfile(self.frame_index_path, dtype=np.uint64).reshape(-1, 2)
        self.chunk_ptrs = np.fromfile(self.chunk_index_path, dtype=np.uint32).reshape(-1, 3)
        self.dict_data = zstd.ZstdCompressionDict(self.dict_path.read_bytes())
        self.decomp = zstd.ZstdDecompressor(dict_data=self.dict_data)
        self.corpus_fp = self.corpus_path.open("rb")
        # simple cache of last decompressed frame
        self._cache_frame_id = -1
        self._cache_frame_data: bytes | None = None

    def get_chunk(self, chunk_id: int) -> str:
        frame_id, start, length = self.chunk_ptrs[chunk_id]

        if frame_id != self._cache_frame_id:
            self._cache_frame_data = self._read_frame(frame_id)
            self._cache_frame_id = frame_id

        piece = self._cache_frame_data[start : start + length]
        decoded_piece = piece.decode("utf-8").rstrip("\n")
        return decoded_piece

    def _read_frame(self, frame_id: int) -> bytes:
        file_offset, comp_size = self.frame_offsets[frame_id]

        self.corpus_fp.seek(int(file_offset))
        comp_bytes = self.corpus_fp.read(int(comp_size))
        decomp_bytes = self.decomp.decompress(comp_bytes)
        return decomp_bytes


# ---------------------------------------------------------------------------
class VectorIndexer:
    """Generates embeddings, builds ON‑DISK IVFPQ index."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.text_store = TextStore(data_dir)
        self.emb_path = data_dir / "embeddings.npy"  # temporary fp32
        self.index_path = data_dir / "faiss.index"
        self.codes_path = data_dir / "pq.codes"  # on‑disk inverted lists
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    # ---------- embedding --------------------------------------------------
    def _embed_all(self, batch=512):
        num_chunks = len(self.text_store.chunk_ptrs)
        embs = np.memmap(self.emb_path, dtype=np.float32, mode="w+", shape=(num_chunks, 384))
        for i in tqdm.trange(0, num_chunks, batch, desc="Embedding"):
            texts = [self.text_store.get_chunk(j) for j in range(i, min(i + batch, num_chunks))]
            vecs = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
            embs[i : i + len(vecs)] = vecs
        return embs

    # ---------- training  --------------------------------------------------
    def build(self):
        embs = self._embed_all()
        num, d = embs.shape
        # train OPQ + IVF + PQ on a sample
        sample_size = min(100_000, num)
        print(f"[FAISS] Building IndexIVFPQ with N_LIST={N_LIST_IVFPQ}, M_PQ={M_PQ_IVFPQ}, N_BITS={N_BITS_IVFPQ}")
        quantizer = faiss.IndexFlatIP(d)  # Quantizer for IVF
        index = faiss.IndexIVFPQ(quantizer, d, N_LIST_IVFPQ, M_PQ_IVFPQ, N_BITS_IVFPQ)
        index.metric_type = faiss.METRIC_INNER_PRODUCT # Ensure this is set for IP search

        # Training the IndexIVFPQ
        # Training happens on a sample of the embeddings
        sample_size = min(256 * N_LIST_IVFPQ, num) # FAISS recommendation: train on at least k * ncentroids, k=30-256
        # Ensure sample_size is not larger than the number of available embeddings
        sample_size = min(sample_size, num)
        if num > sample_size:
            print(f"[FAISS] Training IndexIVFPQ on a sample of {sample_size} / {num} vectors...")
            sample = embs[np.random.choice(num, sample_size, replace=False)]
        else:
            print(f"[FAISS] Training IndexIVFPQ on all {num} vectors (sample size not applicable)...")
            sample = embs
        
        index.train(sample)
        print("[FAISS] Training complete.")

        # Add vectors to the index in batches
        print("[FAISS] Adding vectors to IndexIVFPQ...")
        bs = 50_000 # Batch size for adding to index
        for i in tqdm.trange(0, num, bs, desc="Add to IndexIVFPQ"):
            v = embs[i : i + bs]
            # IndexIVFPQ supports add_with_ids
            index.add_with_ids(v, np.arange(i, i + len(v)))
        faiss.write_index(index, str(self.index_path))
        # mmap file is closed automatically
        print("[OK] FAISS index written →", self.index_path)
        # remove temporary fp32 embeddings to save space
        try:
            os.remove(self.emb_path)
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
class RetrievalEngine:
    """Loads everything memory‑mapped; answers top‑k queries."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.text_store = TextStore(data_dir)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = faiss.read_index(str(data_dir / "faiss.index"), faiss.IO_FLAG_MMAP)
        self.index.nprobe = 32  # For IndexIVFPQ: number of cells to visit during search

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        q_vec = self.model.encode([query], normalize_embeddings=True)
        D, I = self.index.search(q_vec, top_k)
        results = []
        if I is not None and len(I) > 0:
            for score, cid in zip(D[0], I[0]):
                if cid < 0:
                    continue # Should not happen with valid indices but good practice
                chunk_text = self.text_store.get_chunk(int(cid))
                results.append((chunk_text, float(score)))
        return results


# ---------------------------------------------------------------------------
# ----------------------------- CLI -----------------------------------------

app = typer.Typer(help="Ultra-compact RAG storage + retrieval engine CLI")


@app.command("ingest")
def ingest_cmd(
    input_source: Path = typer.Option(
        ..., 
        "--input", "-i", 
        help="Input text file (one doc per line) or directory of .txt files",
        exists=True, # Typer will check if the path exists
        file_okay=True,
        dir_okay=True,
        readable=True
    ),
    data_dir: Path = typer.Option(
        ..., 
        "--data_dir", "-d", 
        help="Directory for output files (will be created if it doesn't exist)"
    )
):
    """Ingest documents: chunk, compress, build indexes."""
    print(f"[INGEST] Input: {input_source}, Output Dir: {data_dir}")
    
    docs: List[str] = []

    if input_source.is_dir():
        print(f"[INFO] Input source is a directory. Reading all .txt files...")
        found_files = False
        for filepath in sorted(input_source.glob("*.txt")):
            print(f"  Reading {filepath.name}...")
            with filepath.open("r", encoding="utf-8") as f:
                docs.append(f.read())
            found_files = True
        if not found_files:
            print(f"[WARN] No .txt files found in directory {input_source}.")
    elif input_source.is_file():
        print(f"[INFO] Input source is a file. Reading its content as a single document...")
        with input_source.open("r", encoding="utf-8") as f:
            docs.append(f.read())
    # Typer's exists=True, file_okay=True, dir_okay=True handles other cases

    if not docs:
        print("[ERR] No documents loaded. Exiting.")
        raise typer.Exit(code=1)

    print(f"[INFO] Loaded {len(docs)} documents for ingestion.")

    data_dir.mkdir(parents=True, exist_ok=True) # Ensure data_dir exists

    compressor = CorpusCompressor(data_dir)
    compressor.ingest(docs)
    
    vindexer = VectorIndexer(data_dir)
    vindexer.build()

@app.command("search")
def search_cmd(
    query: str = typer.Option(..., "--query", "-q", help="Search query"),
    data_dir: Path = typer.Option(
        ..., 
        "--data_dir", "-d", 
        help="Directory containing the store's data files",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True
    ),
    top_k: int = typer.Option(5, "--top_k", "-k", help="Number of top results to return")
):
    """Search query against the store"""
    print(f"[SEARCH] Query: '{query}', Data Dir: {data_dir}, Top K: {top_k}")
    engine = RetrievalEngine(data_dir)
    hits = engine.search(query, top_k=top_k)
    print("\n==== RESULTS ====")
    if not hits:
        print("No results found.")
    for rank, (text, score) in enumerate(hits, 1):
        # Ensure text is not None and score is float before printing
        text_snippet = text[:120] if text else "[ERROR: Empty chunk]"
        score_val = score if isinstance(score, float) else 0.0
        print(f"{rank:2}. ({score_val:.4f}) {text_snippet}...")

@app.command()
def benchmark(
    data_dir: Path = typer.Option(..., help="Directory containing the FAISS index and corpus."),
    queries_file: Path = typer.Option("benchmark_queries.txt", help="Path to a file containing benchmark queries, one per line."),
    num_queries_to_run: int = typer.Option(0, help="Number of queries to run from the file. If 0, runs all queries in the file. If > file_size, it will cycle."),
    repetitions: int = typer.Option(5, help="Number of times to repeat each query for timing."),
    top_k: int = typer.Option(5, help="Number of top results to retrieve for each query.")
):
    """Benchmarks the search speed of the retrieval engine using queries from a file."""
    print(f"[BENCHMARK] Starting benchmark with:")
    print(f"  Data Dir: {data_dir}")
    print(f"  Queries File: {queries_file}")
    print(f"  Num Queries to Run from File: {'All' if num_queries_to_run == 0 else num_queries_to_run}")
    print(f"  Repetitions per query: {repetitions}")
    print(f"  Top-K: {top_k}")

    try:
        with open(queries_file, 'r', encoding='utf-8') as f:
            benchmark_queries_from_file = [line.strip() for line in f if line.strip()]
        if not benchmark_queries_from_file:
            print(f"[ERROR] No queries found in {queries_file} or file is empty.")
            raise typer.Exit(code=1)
        print(f"[BENCHMARK] Loaded {len(benchmark_queries_from_file)} queries from {queries_file}.")
    except FileNotFoundError:
        print(f"[ERROR] Queries file not found: {queries_file}")
        raise typer.Exit(code=1)
    except Exception as e:
        print(f"[ERROR] Could not read queries file {queries_file}: {e}")
        raise typer.Exit(code=1)

    try:
        retrieval_engine = RetrievalEngine(data_dir)
        print("[BENCHMARK] RetrievalEngine initialized.")
    except Exception as e:
        print(f"[ERROR] Could not initialize RetrievalEngine: {e}")
        raise typer.Exit(code=1)

    all_query_times: List[float] = []
    query_durations: Dict[str, List[float]] = {query: [] for query in benchmark_queries_from_file}

    queries_to_actually_run = benchmark_queries_from_file
    effective_num_queries = len(benchmark_queries_from_file)
    if num_queries_to_run > 0:
        effective_num_queries = num_queries_to_run

    for i in range(effective_num_queries):
        query_text = benchmark_queries_from_file[i % len(benchmark_queries_from_file)] # Cycle if num_queries_to_run > actual_num_queries
        print(f"\n[BENCHMARK] Running query {i+1}/{effective_num_queries}: '{query_text}' (repetitions: {repetitions})")
        
        current_query_times: List[float] = []
        for rep in range(repetitions):
            start_time = time.perf_counter()
            results = retrieval_engine.search(query_text, top_k=top_k)
            end_time = time.perf_counter()
            duration = (end_time - start_time) * 1000  # milliseconds
            current_query_times.append(duration)
            all_query_times.append(duration)
            query_durations[query_text].append(duration)
            # Minimal output during reps to avoid skewing timing too much
            # print(f"    Rep {rep+1}/{repetitions}: {duration:.2f} ms - found {len(results)} results") 

        avg_time_for_current_query = statistics.mean(current_query_times)
        print(f"  Avg time for '{query_text}': {avg_time_for_current_query:.2f} ms")

    print("\n==== BENCHMARK SUMMARY ====")
    if not all_query_times:
        print("No queries were run or no timings recorded.")
        return

    total_queries_executed = len(all_query_times)
    overall_avg_time = statistics.mean(all_query_times)
    overall_median_time = statistics.median(all_query_times)
    overall_stdev_time = statistics.stdev(all_query_times) if len(all_query_times) > 1 else 0.0
    total_time_spent_searching = sum(all_query_times) / 1000 # seconds

    print(f"Total queries executed (including repetitions): {total_queries_executed}")
    print(f"Total time spent in search calls: {total_time_spent_searching:.3f} seconds")
    print(f"Overall Average Query Time: {overall_avg_time:.2f} ms")
    print(f"Overall Median Query Time: {overall_median_time:.2f} ms")
    print(f"Overall Standard Deviation: {overall_stdev_time:.2f} ms")
    print(f"Min Query Time: {min(all_query_times):.2f} ms")
    print(f"Max Query Time: {max(all_query_times):.2f} ms")
    print("[OK] Benchmark complete.")


if __name__ == "__main__":
    app()
