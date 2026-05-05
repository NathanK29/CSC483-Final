"""
jeopardy_wiki_qa.py
Authors: Abderrahman Didan, Ali Kaddourra, Andrew Little, Jaden Gee, Nathan Kumar
Purpose: Build a Wikipedia page retrieval system for Jeopardy clues using Whoosh.

Input files:
  questions.txt: 4 lines per question: CATEGORY, CLUE, ANSWER, blank line
  wiki folder/file: Wikipedia pages where each page starts with [[Title]]

Example commands you can use:
  pip install whoosh
  python jeopardy_wiki_qa.py index --wiki wiki-data-folder --index indexdir
  python jeopardy_wiki_qa.py evaluate --questions questions.txt --index indexdir --out results.csv     (A.D come back later and touch this up + give clearer instructions on commands)
"""

import argparse
import csv
import os
import re
import shutil
from pathlib import Path

try:
    from whoosh import index
    from whoosh.analysis import StemmingAnalyzer
    from whoosh.fields import ID, TEXT, Schema
    from whoosh.qparser import MultifieldParser, OrGroup
    from whoosh.scoring import BM25F
except ImportError:
    raise SystemExit(
        "Whoosh is required. Install it with: pip install whoosh"
    )


TITLE_RE = re.compile(r"^\[\[(.*?)\]\]\s*$")
REDIRECT_RE = re.compile(r"^#REDIRECT\s+(.+)$", re.IGNORECASE)
TPL_RE = re.compile(r"\[tpl\].*?\[/tpl\]", re.DOTALL | re.IGNORECASE)
REF_RE = re.compile(r"\[ref\].*?\[/ref\]", re.DOTALL | re.IGNORECASE)
TAG_RE = re.compile(r"<[^>]+>")
BRACKET_LINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")
PUNCT_RE = re.compile(r"[^a-z0-9]+")


def normalize_answer(text):
    #  Normalizes the titles/answers so small grammar and article differences dont break matching
    text = text.lower().strip()
    text = text.replace("&", "and")
    text = PUNCT_RE.sub(" ", text)
    words = [w for w in text.split() if w not in {"the", "a", "an"}]
    return " ".join(words)


def clean_wiki_text(text):
    # Remove common Wikipedia markup while keeping useful words for retrieval
    text = TPL_RE.sub(" ", text)
    text = REF_RE.sub(" ", text)
    text = TAG_RE.sub(" ", text)

    # Keep the displayed side of links 
    #[[Target|shown text]] -> shown text, [[Target]] -> Target.
    text = BRACKET_LINK_RE.sub(lambda m: m.group(2) if m.group(2) else m.group(1), text)

    # Remove section mark but keep the heading words.
    text = text.replace("=", " ")
    text = text.replace("[", " ").replace("]", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def iter_wiki_files(path):
    # Yield all files from either one wiki file or a folder of wiki files
    path = Path(path)
    if path.is_file():
        yield path
        return

    for file_path in sorted(path.rglob("*")):
        if file_path.is_file():
            yield file_path


def iter_pages(path):
    # Parse Wiki pages --> Each page begins with a linelike [[BBC]]
    title = None
    body_lines = []

    for file_path in iter_wiki_files(path):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for raw_line in f:
                line = raw_line.rstrip("\n")
                match = TITLE_RE.match(line)

                if match:
                    if title is not None:
                        yield title, "\n".join(body_lines)
                    title = match.group(1).strip()
                    body_lines = []
                elif title is not None:
                    body_lines.append(line)

    if title is not None:
        yield title, "\n".join(body_lines)


def split_categories_and_body(body):
    # Splits out category tags from the body text if the dump has a CATEGORIES: line
    categories = []
    body_lines = []

    for line in body.splitlines():
        if line.startswith("CATEGORIES:"):
            categories.append(line.replace("CATEGORIES:", " "))
        else:
            body_lines.append(line)

    return " ".join(categories), "\n".join(body_lines)


def create_schema():
    analyzer = StemmingAnalyzer()
    return Schema(
        title=ID(stored=True, unique=True),
        title_text=TEXT(stored=False, analyzer=analyzer, field_boost=3.0),
        categories=TEXT(stored=False, analyzer=analyzer, field_boost=2.0),
        content=TEXT(stored=False, analyzer=analyzer),
    )


def build_index(wiki_path, index_dir, recreate=True):
    # Create a Whoosh BM25F index with one document per Wikipedia page
    index_dir = Path(index_dir)

    if recreate and index_dir.exists():
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    schema = create_schema()
    ix = index.create_in(index_dir, schema)
    writer = ix.writer(limitmb=512, procs=1)

    count = 0
    redirect_count = 0

    for title, raw_body in iter_pages(wiki_path):
        categories, body = split_categories_and_body(raw_body)
        body = clean_wiki_text(body)
        categories = clean_wiki_text(categories)

        redirect_match = REDIRECT_RE.match(body.strip())
        if redirect_match:
            redirect_count += 1
            body = clean_wiki_text(redirect_match.group(1))

        writer.update_document(
            title=title,
            title_text=title,
            categories=categories,
            content=body,
        )
        count += 1

        if count % 10000 == 0:
            print(f"Indexed {count:,} pages...")

    writer.commit()
    print(f"Done. Indexed {count:,} pages. Redirect-like pages handled: {redirect_count:,}.")


def read_questions(path):
    # Read 4-line Jeopardy question blocks
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.rstrip("\n") for line in f]

    questions = []
    i = 0
    while i + 2 < len(lines):
        category = lines[i].strip()
        clue = lines[i + 1].strip()
        answer = lines[i + 2].strip()

        if category and clue and answer:
            questions.append({
                "category": category,
                "clue": clue,
                "answer": answer,
                "answer_variants": [a.strip() for a in answer.split("|")],
            })
        i += 4

    return questions


def clean_query_text(text):
    # Keep meaningful clue/category words and remove Jeopardy punctuation noise
    text = text.replace("&", " and ")
    text = re.sub(r"[\"“”‘’]", " ", text)
    text = re.sub(r"[^A-Za-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()