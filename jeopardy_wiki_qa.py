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
        