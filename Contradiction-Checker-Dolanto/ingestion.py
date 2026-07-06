import os
import json
import pickle
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi


print("🚀 INGESTION MODULE READY")

# =========================
# PATHS
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.dirname(BASE_DIR)

DEFAULT_JSON_PATH = os.path.join(BASE_DIR, "output.json")

DEFAULT_CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

DEFAULT_BM25_PATH = os.path.join(BASE_DIR, "bm25_store.pkl")

# =========================
# CLEAN TEXT
# =========================

def clean(text):
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


def _first_present(data: Dict[str, Any], keys: List[str], default: str = "") -> str:
    for key in keys:
        if key in data and data.get(key) not in [None, ""]:
            return str(data.get(key))
    return default


# =========================
# JSON NORMALISATION
# =========================

def extract_clauses(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, dict):
        if isinstance(raw.get("clauses"), list):
            return raw.get("clauses", [])
        if isinstance(raw.get("data"), dict) and isinstance(raw["data"].get("clauses"), list):
            return raw["data"].get("clauses", [])
        if isinstance(raw.get("documents"), list):
            clauses = []
            for doc in raw.get("documents", []):
                doc_id = _first_present(doc, ["document_id", "doc_id", "id", "name", "title"], "unknown_document")
                for clause in extract_clauses(doc):
                    clause = dict(clause)
                    clause.setdefault("document_id", doc_id)
                    clauses.append(clause)
            return clauses
    if isinstance(raw, list):
        return raw
    return []


def normalise_clause(clause: Dict[str, Any], document_id: str, source_file: str) -> Dict[str, Any]:
    clause_number = _first_present(
        clause,
        ["clause number", "clause_number", "clause_id", "number", "id", "section", "section_number"],
        "unknown"
    )
    clause_title = _first_present(
        clause,
        ["clause title", "clause_title", "title", "heading", "section_title"],
        ""
    )
    clause_text = _first_present(
        clause,
        ["clause", "clause text", "clause_text", "text", "content", "body"],
        ""
    )

    raw_subclauses = clause.get("subclauses", [])
    if raw_subclauses is None:
        raw_subclauses = []
    if not isinstance(raw_subclauses, list):
        raw_subclauses = []

    subclauses = []
    for sub in raw_subclauses:
        if not isinstance(sub, dict):
            continue
        sub_number = _first_present(
            sub,
            ["subclause number", "subclause_number", "subclause_id", "number", "id", "section"],
            ""
        )
        sub_title = _first_present(
            sub,
            ["subclause title", "subclause_title", "title", "heading"],
            ""
        )
        sub_text = _first_present(
            sub,
            ["subclause text", "subclause_text", "text", "content", "body", "clause"],
            ""
        )
        if clean(sub_text):
            subclauses.append({
                "subclause number": sub_number,
                "subclause title": sub_title,
                "subclause text": sub_text
            })

    return {
        "document_id": document_id,
        "source_file": source_file,
        "clause number": clause_number,
        "clause title": clause_title,
        "clause": clause_text,
        "subclauses": subclauses
    }


def load_json_files(json_paths: List[str]) -> List[Dict[str, Any]]:
    all_clauses = []

    for path in json_paths:
        source_file = os.path.basename(path)
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        default_document_id = None
        if isinstance(raw, dict):
            default_document_id = _first_present(raw, ["document_id", "doc_id", "id", "name", "title"], "")
        if not default_document_id:
            default_document_id = Path(source_file).stem

        clauses = extract_clauses(raw)
        for clause in clauses:
            if not isinstance(clause, dict):
                continue
            document_id = _first_present(
                clause,
                ["document_id", "doc_id", "document", "source_document"],
                default_document_id
            )
            normalised = normalise_clause(clause, document_id, source_file)
            all_clauses.append(normalised)

    return all_clauses


# =========================
# BUILD DOCUMENTS
# =========================

def build_documents_from_clauses(clauses: List[Dict[str, Any]]):
    documents = []
    bm25_corpus = []

    for clause in clauses:
        document_id = clean(clause.get("document_id", "unknown_document")) or "unknown_document"
        source_file = clean(clause.get("source_file", "unknown_file")) or "unknown_file"
        clause_id = clean(clause.get("clause number", "unknown")) or "unknown"
        clause_title = clean(clause.get("clause title", ""))
        clause_text = clean(clause.get("clause", ""))
        subclauses = clause.get("subclauses", [])

        unique_clause_id = f"{document_id}::{clause_id}"

        if not subclauses:
            content = f"""
Document: {document_id}
Source File: {source_file}
Clause: {clause_id} - {clause_title}

{clause_text}
"""
            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "document_id": document_id,
                        "source_file": source_file,
                        "clause_id": clause_id,
                        "unique_clause_id": unique_clause_id,
                        "display_id": f"{document_id}::{clause_id}",
                        "type": "clause"
                    }
                )
            )
            bm25_corpus.append(tokenize(content))
            continue

        for sub in subclauses:
            sub_id = clean(sub.get("subclause number", "")) or clause_id
            sub_title = clean(sub.get("subclause title", ""))
            sub_text = clean(sub.get("subclause text", ""))

            sub_text = re.split(
                r'\n?\s*[:.]?\s*\d+\.\d+\s+[A-Z]',
                sub_text,
                maxsplit=1
            )[0].strip()

            if not sub_text:
                continue

            unique_subclause_id = f"{document_id}::{sub_id}"

            content = f"""
Document: {document_id}
Source File: {source_file}
Clause: {clause_id} - {clause_title}
Subclause: {sub_id} - {sub_title}

{sub_text}
"""
            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "document_id": document_id,
                        "source_file": source_file,
                        "clause_id": clause_id,
                        "subclause_id": sub_id,
                        "unique_clause_id": unique_subclause_id,
                        "display_id": f"{document_id}::{sub_id}",
                        "type": "subclause"
                    }
                )
            )
            bm25_corpus.append(tokenize(content))

    return documents, bm25_corpus


# =========================
# INGESTION ENTRYPOINT
# =========================

def ingest_json_files(
    json_paths: List[str],
    chroma_path: str = DEFAULT_CHROMA_PATH,
    bm25_path: str = DEFAULT_BM25_PATH,
    reset: bool = True
):
    print("🚀 INGESTION STARTED")

    if reset:
        if os.path.exists(chroma_path):
            shutil.rmtree(chroma_path)
            print(f"🗑️ Deleted existing ChromaDB: {chroma_path}")
        if os.path.exists(bm25_path):
            os.remove(bm25_path)
            print(f"🗑️ Deleted existing BM25 store: {bm25_path}")

    clauses = load_json_files(json_paths)
    print(f"✅ Loaded {len(clauses)} clauses from {len(json_paths)} JSON file(s)")

    documents, bm25_corpus = build_documents_from_clauses(clauses)
    print(f"✅ Created {len(documents)} documents/subclauses")

    if not documents:
        raise ValueError("No valid clauses/subclauses found. Check your JSON structure.")

    print("Loading embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    print("✅ Embeddings loaded")

    print("Creating vector DB...")
    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=chroma_path
    )
    print(f"✅ Vector DB created with {len(documents)} documents")

    print("Saving BM25 store...")
    bm25 = BM25Okapi(bm25_corpus)
    with open(bm25_path, "wb") as f:
        pickle.dump({"bm25": bm25, "docs": documents}, f)
    print("✅ BM25 store saved")

    summary = {
        "json_files": [os.path.basename(p) for p in json_paths],
        "clauses_loaded": len(clauses),
        "documents_created": len(documents),
        "chroma_path": chroma_path,
        "bm25_path": bm25_path
    }

    print("\n🎉 INGESTION COMPLETE")
    print(f"Chroma DB → {chroma_path}")
    print(f"BM25 Store → {bm25_path}")
    return summary


if __name__ == "__main__":
    print("🚀 INGESTION STARTED")
    files_input = input("JSON file path(s), comma separated [press Enter for output.json]: ").strip()

    if files_input:
        json_paths = [p.strip().strip('"').strip("'") for p in files_input.split(",") if p.strip()]
    else:
        json_paths = [DEFAULT_JSON_PATH]

    reset_input = input("Reset existing ChromaDB and BM25 stores? (y/n) [y]: ").strip().lower()
    reset = reset_input != "n"

    ingest_json_files(
        json_paths=json_paths,
        chroma_path=DEFAULT_CHROMA_PATH,
        bm25_path=DEFAULT_BM25_PATH,
        reset=reset
    )
