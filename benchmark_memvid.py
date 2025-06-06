import time
import statistics
from pathlib import Path
import typer
from typing_extensions import Annotated
from memvid import MemvidRetriever

DEFAULT_QUERIES_FILE = Path("./benchmark_queries.txt")
DEFAULT_DATA_DIR = Path("./data_memvid")
MEMVID_VIDEO_FILE = "memvid_memory.mp4"
MEMVID_INDEX_FILE = "memvid_index.json"

app = typer.Typer()


def load_queries(queries_file: Path, num_queries_to_run: int) -> list[str]:
    if not queries_file.exists():
        print(f"Queries file not found: {queries_file}")
        raise typer.Exit(code=1)
    with open(queries_file, "r", encoding="utf-8") as f:
        queries = [line.strip() for line in f if line.strip()]
    
    if num_queries_to_run > 0:
        return queries[:num_queries_to_run]
    return queries

@app.command()
def benchmark(
    data_dir: Annotated[Path, typer.Option(help="Directory containing the memvid video and index files.")] = DEFAULT_DATA_DIR,
    queries_file: Annotated[Path, typer.Option(help="Path to the file containing benchmark queries, one per line.")] = DEFAULT_QUERIES_FILE,
    num_queries_to_run: Annotated[int, typer.Option(help="Number of queries to run from the file. 0 means all.")] = 0,
    repetitions: Annotated[int, typer.Option(help="Number of times to run each query.")] = 3,
    top_k: Annotated[int, typer.Option(help="Number of top results to retrieve.")] = 5,
):
    """
    Benchmarks the Memvid retrieval engine.
    """
    print("[MEMVID_BENCHMARK] Starting benchmark with:")
    print(f"  Data Dir: {data_dir.resolve()}")
    print(f"  Queries File: {queries_file.resolve()}")
    print(f"  Num Queries to Run from File: {'All' if num_queries_to_run == 0 else num_queries_to_run}")
    print(f"  Repetitions per query: {repetitions}")
    print(f"  Top-K: {top_k}")

    video_path = data_dir / MEMVID_VIDEO_FILE
    index_path = data_dir / MEMVID_INDEX_FILE

    if not video_path.exists() or not index_path.exists():
        print(f"Error: Memvid memory files not found in {data_dir}.")
        print(f"Expected: {video_path.name} and {index_path.name}")
        print("Please run the memvid ingestion script first.")
        raise typer.Exit(code=1)

    queries = load_queries(queries_file, num_queries_to_run)
    if not queries:
        print("No queries loaded. Exiting.")
        raise typer.Exit(code=1)
    
    print(f"[MEMVID_BENCHMARK] Loaded {len(queries)} queries from {queries_file.name}.")

    try:
        retriever = MemvidRetriever(str(video_path), str(index_path))
        print("[MEMVID_BENCHMARK] MemvidRetriever initialized.")
    except Exception as e:
        print(f"Error initializing MemvidRetriever: {e}")
        raise typer.Exit(code=1)

    all_query_times = []
    total_search_time = 0

    for i, query_text in enumerate(queries):
        print(f"\n[MEMVID_BENCHMARK] Running query {i+1}/{len(queries)}: '{query_text}' (repetitions: {repetitions})")
        query_specific_times = []
        for _ in range(repetitions):
            start_time = time.perf_counter()
            results = retriever.search(query_text, top_k=top_k)
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            query_specific_times.append(duration_ms)
            all_query_times.append(duration_ms)
            total_search_time += (end_time - start_time)

    if not all_query_times:
        print("No queries were run or no timings recorded. Cannot generate summary.")
        raise typer.Exit(code=1)

    overall_avg_time = statistics.mean(all_query_times)
    overall_median_time = statistics.median(all_query_times)
    overall_stdev_time = statistics.stdev(all_query_times) if len(all_query_times) > 1 else 0
    min_time = min(all_query_times)
    max_time = max(all_query_times)

    print("\n==== MEMVID BENCHMARK SUMMARY ====")
    print(f"Total queries executed (including repetitions): {len(all_query_times)}")
    print(f"Total time spent in search calls: {total_search_time:.3f} seconds")
    print(f"Overall Average Query Time: {overall_avg_time:.2f} ms")
    print(f"Overall Median Query Time: {overall_median_time:.2f} ms")
    print(f"Overall Standard Deviation: {overall_stdev_time:.2f} ms")
    print(f"Min Query Time: {min_time:.2f} ms")
    print(f"Max Query Time: {max_time:.2f} ms")
    print("[OK] Memvid benchmark complete.")

if __name__ == "__main__":
    app()
