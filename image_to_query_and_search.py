#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
image_to_query_and_search.py

Flow:
1) Read an image via with open(image, "rb")
2) Base64-encode, wrap as data URL -> pass as `image_url` to a Vision model (e.g., gpt-4o)
3) Model returns ONE search query line
4) Run SQLite FTS5 search over your reports (.txt/.md/.pdf with text layer)
5) Print top-K matches with snippets

Usage:
  export OPENAI_API_KEY="sk-..."   # required
  pip install openai pymupdf
  python image_to_query_and_search.py \
    --image /path/to/image.png \
    --reports_dir ./reports \
    --db ./reports_fts.db \
    --top_k 5 \
    --model gpt-4o \
    --prompt_hint "finance, Taiwan equities, anomaly detection"
"""

import os
import sys
import re
import base64
import argparse
import sqlite3
from pathlib import Path
from typing import List, Tuple, Optional

# --- PDF text extraction (PyMuPDF; no OCR) ---
try:
    import fitz  # PyMuPDF
except ImportError:
    print("Please install PyMuPDF first:  pip install pymupdf", file=sys.stderr)
    raise

# --- OpenAI official SDK (Responses API) ---
try:
    from openai import OpenAI
except ImportError:
    print("Please install openai first:  pip install openai", file=sys.stderr)
    raise


# ========== Utilities ==========
def image_to_base64_utf8(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("utf-8")

def guess_mime(ext: str) -> str:
    ext = ext.lower()
    if ext in (".jpg", ".jpeg"): return "image/jpeg"
    if ext == ".png":             return "image/png"
    if ext == ".webp":            return "image/webp"
    if ext == ".bmp":             return "image/bmp"
    if ext == ".gif":             return "image/gif"
    return "application/octet-stream"

def extract_text_from_pdf(pdf_path: Path, max_pages: Optional[int] = None) -> str:
    doc = fitz.open(pdf_path)
    texts = []
    pages = range(len(doc)) if max_pages is None else range(min(len(doc), max_pages))
    for i in pages:
        page = doc.load_page(i)
        texts.append(page.get_text("text"))
    doc.close()
    return "\n".join(texts)

def read_text_file(p: Path) -> str:
    for enc in ("utf-8", "cp950", "big5", "latin-1"):
        try:
            return p.read_text(encoding=enc, errors="ignore")
        except Exception:
            continue
    return ""

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


# ========== SQLite FTS5 ==========
SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS docs
USING fts5(path UNINDEXED, content, tokenize='unicode61');
"""

def build_or_update_fts(db_path: Path, reports_dir: Path) -> int:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute(SCHEMA)
        cur = conn.cursor()
        count = 0

        for p in reports_dir.rglob("*"):
            if not p.is_file():
                continue
            ext = p.suffix.lower()
            if ext not in {".txt", ".md", ".pdf"}:
                continue

            try:
                if ext == ".pdf":
                    text = extract_text_from_pdf(p)
                else:
                    text = read_text_file(p)
                text = normalize_ws(text)
                if not text:
                    continue

                cur.execute("DELETE FROM docs WHERE path = ?", (str(p),))
                cur.execute("INSERT INTO docs(path, content) VALUES(?, ?)", (str(p), text))
                count += 1
            except Exception as e:
                print(f"[WARN] Indexing failed for {p}: {e}", file=sys.stderr)

        conn.commit()
        return count
    finally:
        conn.close()

def fts_search(db_path: Path, query: str, top_k: int = 5) -> List[Tuple[str, str]]:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA query_only = TRUE;")
        cur = conn.cursor()
        sql = """
        SELECT path,
               snippet(docs, 1, '[', ']', ' … ', 16) AS snip
        FROM docs
        WHERE docs MATCH ?
        ORDER BY rank
        LIMIT ?
        """
        cur.execute(sql, (query, top_k))
        rows = cur.fetchall()

        results = []
        for path, snip in rows:
            if not snip or not snip.strip():
                cur2 = conn.cursor()
                cur2.execute("SELECT substr(content, 1, 240) FROM docs WHERE path = ?", (path,))
                head = cur2.fetchone()[0] or ""
                results.append((path, normalize_ws(head) + " …"))
            else:
                results.append((path, normalize_ws(snip)))
        return results
    finally:
        conn.close()


# ========== OpenAI Vision: image_url (data URL) ==========
def image_to_query_with_gpt(client: OpenAI, model: str, image_path: Path, prompt_hint: str = "") -> str:
    b64 = image_to_base64_utf8(image_path)
    mime = guess_mime(image_path.suffix)
    data_url = f"data:{mime};base64,{b64}"

    instruction = (
        "You are a search-query composer. Look at the image and extract key topics, "
        "proper nouns, identifiers (e.g., tickers, part numbers), and produce a concise "
        "boolean/keyword query suitable for an SQLite FTS5 MATCH.\n"
        "Prefer nouns and key phrases; avoid filler words. Return ONLY the query on one line."
    )
    if prompt_hint:
        instruction += f"\nContext hint: {prompt_hint}"

    resp = client.responses.create(
        model=model,  # e.g., "gpt-4o" (must support vision/input_image)
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text",  "text": instruction},
                {"type": "input_image", "image_url": {"url": data_url}},
            ],
        }],
    )
    return (resp.output_text or "").strip()


# ========== Main ==========
def main():
    parser = argparse.ArgumentParser(description="Image -> GPT-4o Vision -> FTS5 search")
    parser.add_argument("--image",       type=str, required=True, help="Path to image (opened with 'rb')")
    parser.add_argument("--reports_dir", type=str, required=True, help="Folder of reports (.txt/.md/.pdf with text layer)")
    parser.add_argument("--db",          type=str, default="./reports_fts.db", help="SQLite DB path (FTS5)")
    parser.add_argument("--top_k",       type=int, default=5, help="Number of results to return")
    parser.add_argument("--model",       type=str, default="gpt-4o", help="Vision-capable model (e.g., gpt-4o)")
    parser.add_argument("--prompt_hint", type=str, default="", help="Optional domain hint (e.g., finance, robotics)")
    args = parser.parse_args()

    image_path  = Path(args.image).expanduser().resolve()
    reports_dir = Path(args.reports_dir).expanduser().resolve()
    db_path     = Path(args.db).expanduser().resolve()

    assert image_path.exists(),  f"Image not found: {image_path}"
    assert reports_dir.exists(), f"Reports dir not found: {reports_dir}"

    # 1) Build/refresh FTS index
    n = build_or_update_fts(db_path, reports_dir)
    print(f"[INFO] Indexed/updated {n} files -> {db_path}")

    # 2) Call OpenAI Vision to produce a ONE-LINE query
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Missing OPENAI_API_KEY. Please: export OPENAI_API_KEY='your_key'", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    query = image_to_query_with_gpt(client, args.model, image_path, args.prompt_hint)
    if not query:
        print("[ERROR] Model returned empty query.", file=sys.stderr)
        sys.exit(2)
    print(f"[INFO] Model query: {query}")

    # 3) FTS search
    hits = fts_search(db_path, query, top_k=args.top_k)

    # 4) Output
    if not hits:
        print("[RESULT] No related reports found.")
    else:
        print("\n[RESULT] Top matches:")
        for i, (path, snip) in enumerate(hits, 1):
            print(f"{i}. {path}")
            print(f"   snippet: {snip}\n")


if __name__ == "__main__":
    main()
