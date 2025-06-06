"""
Script to download top Wikipedia articles and Project Gutenberg books,
prepare them into a single text file, and then index them using
rag_compressed_store.py.
"""
import requests
from bs4 import BeautifulSoup
import wikipedia
from pathlib import Path
import subprocess
import re
import time
import sys
import argparse
import concurrent.futures
import math



# --- Constants ---
NUM_WIKIPEDIA_ARTICLES = 1000
NUM_GUTENBERG_BOOKS = 10
CURRENT_DIR = Path(__file__).parent.resolve()
DATA_DIR = CURRENT_DIR / "data"  # Output for indexes, corpus.zs etc.
INPUT_DATA_DIR = CURRENT_DIR / "input_data" # For individual downloaded text files
# DOCS_FILE = DATA_DIR / "docs.txt" # No longer used

# URLs for fetching top lists
WIKIPEDIA_TOP_ARTICLES_URL = "https://en.wikipedia.org/wiki/Wikipedia:Vital_articles/Level/4"
GUTENBERG_TOP_BOOKS_URL = "https://www.gutenberg.org/browse/scores/top"
MAX_WIKI_DOWNLOAD_WORKERS = 8 # Number of parallel workers for Wikipedia downloads

# --- Helper Functions ---
def sanitize_filename(name: str) -> str:
    """Sanitizes a string to be a valid filename."""
    name = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', name)
    return name[:100] # Limit length to avoid overly long filenames

def clean_text(text: str) -> str:
    """Replaces newlines and multiple spaces with a single space."""
    return re.sub(r'\s+', ' ', text).strip()

# --- Wikipedia Functions ---
def get_top_wikipedia_article_titles(num_articles: int) -> list[str]:
    """
    Scrapes Wikipedia for top article titles by first finding sub-category pages
    from the main Vital Articles Level 4 page, and then scraping article titles
    from those sub-category pages.
    """
    print(f"Fetching top {num_articles} Wikipedia article titles, starting from {WIKIPEDIA_TOP_ARTICLES_URL}...")
    all_article_titles = []
    processed_subcategory_urls = set()
    subcategory_page_urls = []

    try:
        # Phase 1: Get sub-category page URLs from the main Level 4 page
        print(f"  Phase 1: Fetching sub-category links from {WIKIPEDIA_TOP_ARTICLES_URL}")
        main_page_response = requests.get(WIKIPEDIA_TOP_ARTICLES_URL, timeout=30)
        main_page_response.raise_for_status()
        main_soup = BeautifulSoup(main_page_response.content, 'html.parser')

        main_content_wrapper = main_soup.find('div', id='mw-content-text')
        if not main_content_wrapper:
            print(f"    [ERR] Could not find 'div#mw-content-text' on {WIKIPEDIA_TOP_ARTICLES_URL}. Cannot find sub-category links.")
            return []
        
        main_parser_output = main_content_wrapper.find('div', class_='mw-parser-output')
        if not main_parser_output:
            print(f"    [ERR] Could not find 'div.mw-parser-output' within 'div#mw-content-text' on {WIKIPEDIA_TOP_ARTICLES_URL}.")
            return []
        
        base_level4_path = "/wiki/Wikipedia:Vital_articles/Level/4/"
        # Find links that specifically point to Level 4 sub-category pages
        # These are typically of the form /wiki/Wikipedia:Vital_articles/Level/4/SubCategoryName
        for link in main_parser_output.find_all('a', href=True, title=True):
            href = link.get('href')
            title_attr = link.get('title')
            if href.startswith(base_level4_path) and \
               title_attr.startswith("Wikipedia:Vital articles/Level/4/") and \
               len(href) > len(base_level4_path) and \
               '/' not in href[len(base_level4_path):]: # Ensures it's a direct sub-page like /People, not /People/Writers
                
                full_url = f"https://en.wikipedia.org{href}"
                if full_url not in subcategory_page_urls:
                    subcategory_page_urls.append(full_url)
                    # print(f"    [DEBUG] Found sub-category page: {full_url} (Title: {title_attr})")

        if not subcategory_page_urls:
            print(f"    [WARN] No sub-category page URLs dynamically found on {WIKIPEDIA_TOP_ARTICLES_URL}. Using predefined list.")
            predefined_categories = [
                "People", "History", "Geography", "Arts", "Philosophy_and_religion", 
                "Everyday_life", "Society_and_social_sciences", 
                "Biological_and_health_sciences", "Physical_sciences", "Technology", "Mathematics"
            ]
            for cat_name in predefined_categories:
                subcategory_page_urls.append(f"https://en.wikipedia.org/wiki/Wikipedia:Vital_articles/Level/4/{cat_name.replace(' ', '_')}")

        print(f"  Phase 1: Identified {len(subcategory_page_urls)} sub-category pages to scrape.")

        articles_per_category = 0
        if subcategory_page_urls:
            num_sub_categories = len(subcategory_page_urls)
            articles_per_category = math.ceil(num_articles / num_sub_categories)
            print(f"  Targeting approx. {articles_per_category} articles per category.")
        else:
            print("  [WARN] No sub-category pages identified, cannot fetch articles.")
            return []

        # Phase 2: Scrape article titles from each sub-category page, aiming for proportional collection
        for sub_url in subcategory_page_urls:
            if len(all_article_titles) >= num_articles:
                print("    Overall article limit reached. Stopping further category processing.")
                break # Stop processing more categories if we already have enough articles

            if sub_url in processed_subcategory_urls:
                continue
            
            titles_from_this_category = 0

            print(f"    Phase 2: Fetching articles from sub-category page: {sub_url}")
            processed_subcategory_urls.add(sub_url)
            try:
                sub_response = requests.get(sub_url, timeout=30)
                sub_response.raise_for_status()
                sub_soup = BeautifulSoup(sub_response.content, 'html.parser')

                sub_content_wrapper = sub_soup.find('div', id='mw-content-text')
                if not sub_content_wrapper:
                    print(f"      [WARN] Could not find 'div#mw-content-text' on {sub_url}. Skipping.")
                    continue
                
                sub_parser_output = sub_content_wrapper.find('div', class_='mw-parser-output')
                if not sub_parser_output:
                    print(f"      [WARN] Could not find 'div.mw-parser-output' in 'div#mw-content-text' on {sub_url}. Skipping.")
                    continue
                
                links_on_sub_page = sub_parser_output.find_all('a', href=True, title=True)
                
                for link in links_on_sub_page:
                    href = link.get('href')
                    title_attr = link.get('title')

                    if not href.startswith('/wiki/'):
                        continue
                    
                    is_meta_page = False
                    meta_prefixes = ["Category:", "Wikipedia:", "Help:", "Template:", "File:", "Portal:", "Special:", "Talk:", "User:", "wikt:"]
                    # Check title attribute first, as it's often cleaner
                    for prefix in meta_prefixes:
                        if title_attr.startswith(prefix):
                            is_meta_page = True
                            break
                    if is_meta_page: continue
                    
                    # Check href for namespace prefixes
                    # e.g. /wiki/Category:Foo, /wiki/Wikipedia:Bar
                    # The part after /wiki/ should not contain a colon before the actual title part
                    path_segment = href[len('/wiki/'):]
                    if ':' in path_segment:
                        # Allow if it's a title like "Title: Subtitle" but not "Namespace:Title"
                        # A simple check: if the part before ':' is one of the known namespaces, it's meta.
                        potential_ns = path_segment.split(':', 1)[0]
                        if any(potential_ns.lower() == ns.lower().rstrip(':') for ns in meta_prefixes):
                            is_meta_page = True
                    if is_meta_page: continue

                    if " (page does not exist)" in title_attr or "action=edit" in href:
                        continue
                    
                    # Ensure the link is within an <li> tag, as a heuristic for actual article lists
                    if not link.find_parent('li'):
                        continue # Be stricter: only consider links within list items

                    article_title_to_add = title_attr
                    if article_title_to_add and article_title_to_add not in all_article_titles:
                        if titles_from_this_category < articles_per_category and len(all_article_titles) < num_articles:
                            all_article_titles.append(article_title_to_add)
                            titles_from_this_category += 1
                            # print(f"        [+] Added: {article_title_to_add} (Total: {len(all_article_titles)}, Cat: {titles_from_this_category}/{articles_per_category})") # Verbose
                        elif len(all_article_titles) >= num_articles:
                            # print(f"          Overall article limit {num_articles} reached while processing {sub_url}. Breaking inner loop.")
                            break # Stop adding from this category if overall limit is met
                        elif titles_from_this_category >= articles_per_category:
                            # print(f"          Category limit {articles_per_category} reached for {sub_url}. Breaking inner loop.")
                            break # Stop adding from this category if its quota is met
                
                # After iterating all links in a sub_page, check if overall limit is met
                if len(all_article_titles) >= num_articles:
                    # print(f"    Overall article limit {num_articles} reached after processing {sub_url}. Breaking outer loop.")
                    break # Stop processing more categories

            except requests.RequestException as e_sub:
                print(f"      [ERR] Could not fetch sub-category page {sub_url}: {e_sub}")
                continue

    except requests.RequestException as e_main:
        print(f"  [ERR] Could not fetch main Wikipedia page {WIKIPEDIA_TOP_ARTICLES_URL}: {e_main}")
        return []

    if not all_article_titles:
        print("[WARN] No article titles found after processing all pages. Check scraping logic and URLs.")
    elif len(all_article_titles) < num_articles:
        print(f"[WARN] Found only {len(all_article_titles)} Wikipedia articles, requested {num_articles}.")
    else:
        print(f"[OK] Successfully fetched {len(all_article_titles)} Wikipedia article titles.")

    return all_article_titles[:num_articles]

def get_wikipedia_article_content(title: str) -> str | None:
    """Fetches and cleans the content of a single Wikipedia article."""
    print(f"  Fetching Wikipedia article: {title}")
    try:
        page = wikipedia.page(title, auto_suggest=False, redirect=True)
        return clean_text(page.content)
    except wikipedia.exceptions.PageError:
        print(f"    [WARN] PageError: Article '{title}' not found or error loading.")
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"    [INFO] DisambiguationError for '{title}'. Options: {e.options[:3]}...")
        if e.options:
            new_title = e.options[0]
            print(f"    Attempting to fetch disambiguated title: '{new_title}'")
            try:
                page = wikipedia.page(new_title, auto_suggest=False, redirect=True)
                return clean_text(page.content)
            except wikipedia.exceptions.PageError:
                print(f"      [WARN] PageError: Article '{new_title}' (from disambiguation) not found.")
            except wikipedia.exceptions.DisambiguationError:
                print(f"      [WARN] DisambiguationError: '{new_title}' (from disambiguation) led to another disambiguation. Skipping.")
            except Exception as e_inner:
                print(f"      [ERR] Unexpected error fetching disambiguated article '{new_title}': {e_inner}")
        else:
            print(f"    [WARN] No disambiguation options found for '{title}'. Skipping.")
    except Exception as e:
        print(f"    [ERR] Unexpected error fetching article '{title}': {e}")
    return None

# --- Project Gutenberg Functions ---
def get_top_gutenberg_book_ids(num_books: int) -> list[int]:
    """Scrapes Project Gutenberg for top book IDs."""
    print(f"Fetching top {num_books} Gutenberg book IDs from {GUTENBERG_TOP_BOOKS_URL}...")
    book_ids = []
    try:
        response = requests.get(GUTENBERG_TOP_BOOKS_URL, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        # Links are typically like <a href="/ebooks/1342">...</a>
        for link in soup.find_all('a', href=re.compile(r'^/ebooks/\d+')):
            match = re.search(r'/ebooks/(\d+)', link['href'])
            if match:
                book_id = int(match.group(1))
                if book_id not in book_ids:
                    book_ids.append(book_id)
                    if len(book_ids) >= num_books:
                        break
        if not book_ids:
            print("[WARN] No book IDs found. Check scraping logic and URL.")
        elif len(book_ids) < num_books:
             print(f"[WARN] Found only {len(book_ids)} Gutenberg books, requested {num_books}.")
    except requests.RequestException as e:
        print(f"[ERR] Could not fetch Gutenberg top books list: {e}")
        return []
    return book_ids[:num_books]

def get_gutenberg_book_text(book_id: int) -> str | None:
    """Fetches and cleans the text of a Project Gutenberg book directly."""
    print(f"  Fetching Gutenberg book ID: {book_id}")
    # Common URL patterns for plain text UTF-8 files
    # The first one is preferred for newer books, the second for older ones.
    urls_to_try = [
        f"https://www.gutenberg.org/ebooks/{book_id}.txt.utf-8",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt" # Less common, but worth a try
    ]
    
    text_content = None
    for url in urls_to_try:
        try:
            print(f"    Trying URL: {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            text_content = response.text
            # Check if we actually got text and not an HTML error page
            if "<html" in text_content[:1000].lower():
                print(f"    [WARN] URL {url} returned HTML, not plain text.")
                text_content = None # Reset if it's HTML
                continue # Try next URL
            print(f"    [OK] Successfully fetched from {url}")
            break # Success
        except requests.RequestException as e:
            print(f"    [WARN] Failed to fetch from {url}: {e}")
            continue # Try next URL

    if not text_content:
        print(f"    [ERR] Could not download text for book ID {book_id} from any known URL pattern.")
        return None

    # Basic Gutenberg header/footer removal
    # More sophisticated regex might be needed for edge cases
    start_marker_variants = [
        "*** START OF THIS PROJECT GUTENBERG EBOOK",
        "*** START OF THE PROJECT GUTENBERG EBOOK",
        "*END*THE SMALL PRINT! FOR PUBLIC DOMAIN EBOOKS*Ver.04.29.03*END*"
    ]
    end_marker_variants = [
        "*** END OF THIS PROJECT GUTENBERG EBOOK",
        "*** END OF THE PROJECT GUTENBERG EBOOK",
        "End of the Project Gutenberg EBook",
        "End of Project Gutenberg"
    ]

    cleaned_text = text_content
    found_start = False
    for marker in start_marker_variants:
        start_match = re.search(re.escape(marker) + r".*?\*\*\*", cleaned_text, re.IGNORECASE | re.DOTALL)
        if start_match:
            cleaned_text = cleaned_text[start_match.end():]
            found_start = True
            break
    
    if not found_start:
        print(f"    [WARN] Start marker not found for book {book_id}. Using text as is.")

    found_end = False
    for marker in end_marker_variants:
        # Search from the end of the string for the end marker
        # This is a bit trickier; we'll find the last occurrence if multiple exist
        end_match = None
        for match_obj in re.finditer(re.escape(marker), cleaned_text, re.IGNORECASE):
            end_match = match_obj # Keep the last one found
        
        if end_match:
            cleaned_text = cleaned_text[:end_match.start()]
            found_end = True
            break
            
    if not found_end:
        print(f"    [WARN] End marker not found for book {book_id}. Using text as is (after start marker removal if any).")

    return clean_text(cleaned_text.strip())

# --- Main Script Logic ---
def download_and_save_wiki_article(task_args):
    """Worker function to download, clean, and save a single Wikipedia article."""
    idx, title, total_articles, input_data_dir_path = task_args
    filename = input_data_dir_path / f"wiki_{sanitize_filename(title)}.txt"

    # Check if file already exists before attempting download
    if filename.exists():
        # print(f"  Skipping already downloaded Wikipedia article: {title} ({idx + 1}/{total_articles})") # Optional: uncomment for verbose skip logs
        return "exists"

    # print(f"    Fetching Wikipedia article: '{title}' ({idx + 1}/{total_articles})") # Optional: uncomment for verbose attempt logs
    content = get_wikipedia_article_content(title) # This function has its own prints for page/disambig errors
    
    if content:
        # print(f"      Content received for '{title}', length: {len(content)}. Writing to {filename} ({idx + 1}/{total_articles})") # Optional: uncomment for verbose success logs
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  [OK] Downloaded '{title}' ({idx + 1}/{total_articles})")
            return "new"
        except IOError as e:
            print(f"  [ERR] Could not write file for article '{title}': {e} ({idx + 1}/{total_articles})")
            return "failed"
    else:
        # print(f"      No content received for '{title}'. Skipping file write. ({idx + 1}/{total_articles})") # Optional: uncomment for verbose no-content logs
        return "failed"

def main():
    """Main function to download data and trigger indexing."""
    parser = argparse.ArgumentParser(description="Download Wikipedia and Gutenberg data and/or ingest it.")
    parser.add_argument("--skip-download", action="store_true", help="Skip the data download phase.")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip the data ingestion phase.")
    args = parser.parse_args()

    processed_doc_count = 0
    newly_downloaded_count = 0

    if not args.skip_download:
        print("Starting data download process...")
        DATA_DIR.mkdir(parents=True, exist_ok=True) # For rag_compressed_store output
        INPUT_DATA_DIR.mkdir(parents=True, exist_ok=True) # For downloaded files

        # Count existing relevant files first
        existing_files = list(INPUT_DATA_DIR.glob("wiki_*.txt")) + list(INPUT_DATA_DIR.glob("gutenberg_*.txt"))
        processed_doc_count = len(existing_files)
        print(f"Found {processed_doc_count} existing documents in {INPUT_DATA_DIR}.")

        # Get Wikipedia articles
        wiki_titles = get_top_wikipedia_article_titles(NUM_WIKIPEDIA_ARTICLES)
        if wiki_titles:
            print(f"Fetching content for {len(wiki_titles)} Wikipedia articles using up to {MAX_WIKI_DOWNLOAD_WORKERS} workers...")
            tasks_to_submit = []
            for i, title in enumerate(wiki_titles):
                # Prepare arguments for the worker function
                tasks_to_submit.append((i, title, len(wiki_titles), INPUT_DATA_DIR))

            newly_downloaded_wiki_this_run = 0
            processed_wiki_tasks_count = 0 # Tracks how many tasks (attempts) have completed
            total_wiki_tasks = len(tasks_to_submit)

            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WIKI_DOWNLOAD_WORKERS) as executor:
                future_to_task_args = {executor.submit(download_and_save_wiki_article, task_args): task_args for task_args in tasks_to_submit}
                
                for future in concurrent.futures.as_completed(future_to_task_args):
                    task_args_processed = future_to_task_args[future]
                    title_processed = task_args_processed[1] # Get the title from task_args
                    processed_wiki_tasks_count += 1
                    try:
                        status = future.result()
                        if status == "new":
                            newly_downloaded_wiki_this_run += 1
                            # processed_doc_count will be updated after this loop based on newly_downloaded_count
                        elif status == "exists":
                            pass # File already existed, already counted in processed_doc_count initial calculation
                        elif status == "failed":
                            # Error messages are printed within download_and_save_wiki_article or get_wikipedia_article_content
                            print(f"  [INFO] Download/Save failed for article: {title_processed} (status from worker: {status})")
                    except Exception as exc:
                        print(f"  [EXC] Article '{title_processed}' generated an exception during future.result(): {exc}")
                    
                    # Log progress periodically
                    if processed_wiki_tasks_count % 20 == 0 or processed_wiki_tasks_count == total_wiki_tasks:
                        print(f"    Completed {processed_wiki_tasks_count}/{total_wiki_tasks} Wikipedia download tasks...")
            
            newly_downloaded_count += newly_downloaded_wiki_this_run # Add wiki downloads to total new downloads for this run
            # processed_doc_count is updated based on newly_downloaded_count later in the script if new files were added.
            # The initial value of processed_doc_count includes all existing files.
            # If newly_downloaded_wiki_this_run > 0, then processed_doc_count will effectively increase.
            print(f"    Finished Wikipedia downloads. Newly downloaded in this run: {newly_downloaded_wiki_this_run}.")
        
        # Get Gutenberg books
        gutenberg_ids = get_top_gutenberg_book_ids(NUM_GUTENBERG_BOOKS)
        if gutenberg_ids:
            print(f"Fetching content for {len(gutenberg_ids)} Gutenberg books...")
            for i, book_id in enumerate(gutenberg_ids):
                filename = INPUT_DATA_DIR / f"gutenberg_{book_id}.txt"
                if filename.exists():
                    print(f"  Skipping already downloaded Gutenberg book ID: {book_id}")
                    continue

                text = get_gutenberg_book_text(book_id)
                if text:
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(text)
                    newly_downloaded_count += 1
                    processed_doc_count += 1
                time.sleep(0.5) # Be polite to Gutenberg's servers
                if (i + 1) % 2 == 0:
                    print(f"    Processed {i+1}/{len(gutenberg_ids)} books.")

        print(f"[INFO] Newly downloaded documents in this run: {newly_downloaded_count}")
        if processed_doc_count == 0:
            print(f"[ERR] No documents found in {INPUT_DATA_DIR} after download process. Exiting before ingestion.")
            return
        print(f"[OK] Total {processed_doc_count} documents available in {INPUT_DATA_DIR} for ingestion.")
    else:
        print("[INFO] Download phase skipped by command-line argument.")
        # If download is skipped, we need to count existing files to see if ingestion can proceed
        if not INPUT_DATA_DIR.exists(): # Ensure directory exists if download is skipped
            print(f"[WARN] Input data directory {INPUT_DATA_DIR} does not exist. Cannot proceed with ingestion if selected.")
            return
        existing_files = list(INPUT_DATA_DIR.glob("wiki_*.txt")) + list(INPUT_DATA_DIR.glob("gutenberg_*.txt"))
        processed_doc_count = len(existing_files)
        if processed_doc_count == 0:
            print(f"[INFO] No documents found in {INPUT_DATA_DIR}. Ingestion cannot proceed.")
            return
        print(f"[INFO] Found {processed_doc_count} existing documents in {INPUT_DATA_DIR} for potential ingestion.")

    # Run the ingestion script
    if not args.skip_ingest:
        if processed_doc_count == 0:
            print("[INFO] No documents available for ingestion. Skipping ingestion phase.")
            return
        print("Starting ingestion process using rag_compressed_store.py...")
    ingest_script_path = CURRENT_DIR / "rag_compressed_store.py"
    venv_python_path = CURRENT_DIR / ".venv" / "bin" / "python"

    if not venv_python_path.exists():
        print(f"[ERR] Python interpreter not found at {venv_python_path}. Make sure the virtual environment is set up correctly.")
        return
    if not ingest_script_path.exists():
        print(f"[ERR] Ingest script not found at {ingest_script_path}.")
        return

    try:
        command = [
            str(venv_python_path),
            str(ingest_script_path),
            "ingest",
            "--input", str(INPUT_DATA_DIR),
            "--data_dir", str(DATA_DIR)
        ]
        print(f"Running command: {' '.join(command)}")
        # capture_output=False means stdout/stderr go directly to terminal
        process = subprocess.run(command, check=True, capture_output=False, text=True, cwd=CURRENT_DIR)
        print("[OK] Ingestion process completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"[ERR] Ingestion process failed with exit code {e.returncode}.")
        print(f"  Stdout: {e.stdout}")
        print(f"  Stderr: {e.stderr}")
    except FileNotFoundError:
        print(f"[ERR] Could not find Python interpreter or ingest script. Searched for Python at {venv_python_path}")

if __name__ == "__main__":
    # Ensure data directory exists for rag_compressed_store.py even if download is skipped
    # but ingestion is not. This is a bit of a structural dependency.
    # A more robust solution might involve rag_compressed_store.py creating its own data_dir.
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    main()
