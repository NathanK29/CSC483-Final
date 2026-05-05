"""
FileName: jeopardy_wiki_qa.py
Authors: Abderrahman Didan, Ali Kaddourra, Andrew Little, Jaden Gee, Nathan Kumar
Purpose: Build a Wikipedia page retrieval system for Jeopardy clues using Whoosh.
"""

import argparse
import csv
import os
import re
import shutil
from pathlib import Path
from whoosh import index
from whoosh.analysis import StemmingAnalyzer
from whoosh.fields import ID, TEXT, Schema
from whoosh.qparser import MultifieldParser, OrGroup
from whoosh.scoring import BM25F

#globals
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
    # Builds a BM25F Whoosh index, one document for each Wikipedia page
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
    # Parses the questions file, expecting 4-line blocks per question
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
    # Strips out punctuation noise and keeps the useful words from clues and categories
    text = text.replace("&", " and ")
    text = re.sub(r"[\"“”‘’]", " ", text)
    text = re.sub(r"[^A-Za-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Method: build_query
# Purpose: This method creates query text for one Jepoardy Clue.
# Parameters:
#   category: The Jeopardy category text
#   clue: Clue to serarch for
#   use_category: Boolean if we want to append category words to the clue query
# Returns: A cleaned query string 
def build_query(category, clue, use_category=True):

    # get the texts
    clue_text = clean_query_text(clue)
    category_text = clean_query_text(category)

    if use_category and category_text:
        return f"{clue_text} {category_text}"
    return clue_text


# Method: search_one
# Purpose: Searches the index for the best Wikipedia 
#          pages for a given clue
# Parameters:
#   ix: The index.
#   category: The Jeopardy category text
#   clue: The clue text to search for
#   top_k: The maximum number of ranked results to return
#   use_category: Whether to include category text in the query
# Returns: A list of page title, score tuples ordered by the relevance
def search_one(ix, category, clue, top_k=10, use_category=True):
    query_text = build_query(category, clue, use_category=use_category)

    # Search all useful fields
    parser = MultifieldParser(
        ["title_text", "categories", "content"],
        schema=ix.schema,
        group=OrGroup.factory(0.9),
    )
    query = parser.parse(query_text)

    with ix.searcher(weighting=BM25F()) as searcher:
        # Getting the top matching pages from the index
        hits = searcher.search(query, limit=top_k)
        results = []
        # store results 
        for hit in hits:
            results.append((hit["title"], float(hit.score)))
        return results


# Method: reciprocal_rank
# Purpose: Scores a ranked result list 
#           based on where the correct answer appears
# Parameters:
#   results: Ranked (page title, score) tuples returned by search_one method
#   answer_variants: Acceptable answer strings for the clue
# Returns: the reciprocal rank, 
#            or 0.0 if not found.
def reciprocal_rank(results, answer_variants):
    normalized_answers = set()
    # normalization step
    # and adding it to the set
    for ans in answer_variants:
        normalized_answers.add(normalize_answer(ans))

    # checking results in ranked order
    for rank, (title, _) in enumerate(results, start=1):
        if normalize_answer(title) in normalized_answers:
            return 1.0 / rank
    return 0.0


# Method: is_correct_at_1
# Purpose: Checks whether the top ranked 
#           result exactly matches an accepted answer
# Parameters:
#   results: Ranked (page title, score) tuples returned by the search_one method
#   answer_variants: Acceptable answer strings for the clue.
# Returns: True if the first result is correct, otherwise False.
def is_correct_at_1(results, answer_variants):
    if not results:
        return False
    
    normalized_answers = set()
    # normalization step
    for ans in answer_variants:
        normalized_answers.add(normalize_answer(ans))
    # get first ranked result
    top_result = results[0]
    # get page title
    top_title = top_result[0]
    # normalize the top page
    normalized_top_title = normalize_answer(top_title)
    return normalized_top_title in normalized_answers
