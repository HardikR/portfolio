import pickle
import numpy as np
import re
import time
import csv
import json
import os
from pathlib import Path
from collections import Counter

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from llm_reranker import ollama_rerank
from contradiction_checker import check_contradiction



BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
DEFAULT_BM25_PATH = os.path.join(BASE_DIR, "bm25_store.pkl")

bm25 = None
bm25_docs = []
db = None
embeddings = None


def load_retrieval_stores(
    bm25_path=DEFAULT_BM25_PATH,
    chroma_path=DEFAULT_CHROMA_PATH
):
    global bm25, bm25_docs, db, embeddings

    print("🚀 Loading BM25...")
    with open(bm25_path, "rb") as f:
        bm25_data = pickle.load(f)

    bm25 = bm25_data["bm25"]
    bm25_docs = bm25_data["docs"]

    print("🚀 Loading Chroma...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma(
        persist_directory=chroma_path,
        embedding_function=embeddings
    )

    print(f"✅ Loaded {len(bm25_docs)} documents/subclauses")
    return bm25_docs


LEGAL_STOPWORDS = {
    "the", "and", "or", "of", "to", "in", "a", "an", "by", "for", "with",
    "under", "this", "that", "any", "all", "as", "is", "are", "be", "on",
    "from", "it", "its", "such", "which", "will", "may"
}

LEGAL_KEYWORDS = {
    "must", "must not", "shall", "shall not", "may", "may not",
    "required", "not required", "liable", "not liable",
    "approval", "consent", "notice", "claim", "change",
    "terminate", "termination", "payment", "days", "business days",
    "responsible", "responsibility", "permission", "prohibited",
    "obligation", "condition", "scope", "delay", "cost"
}


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


def norm(x):
    x = np.array(x, dtype=float)

    if len(x) == 0:
        return np.array([])

    if x.max() == x.min():
        return np.zeros_like(x)

    return (x - x.min()) / (x.max() - x.min())


def get_clause_id(doc):
    meta = getattr(doc, "metadata", {}) or {}
    return meta.get("unique_clause_id") or meta.get("display_id") or meta.get("subclause_id") or meta.get("clause_id") or "unknown"


def get_display_id(doc):
    meta = getattr(doc, "metadata", {}) or {}
    return meta.get("display_id") or get_clause_id(doc)


def get_document_id(doc):
    meta = getattr(doc, "metadata", {}) or {}
    return meta.get("document_id", "unknown_document")


def extract_nlp_features(text):
    lower = text.lower()

    tokens = [
        t for t in tokenize(lower)
        if t not in LEGAL_STOPWORDS and len(t) > 2
    ]

    keyword_hits = set()

    for kw in LEGAL_KEYWORDS:
        if kw in lower:
            keyword_hits.add(kw)

    deadlines = re.findall(
        r"\b\d+\s+(?:business\s+)?days?\b|\b\d+\s+months?\b|\b\d+\s+years?\b",
        lower
    )

    modalities = set()

    if "must not" in lower or "shall not" in lower or "may not" in lower:
        modalities.add("prohibition")

    if "must" in lower or "shall" in lower or "required" in lower:
        modalities.add("obligation")

    if "may" in lower or "entitled" in lower or "permission" in lower:
        modalities.add("permission")

    if "notice" in lower:
        modalities.add("notice")

    if "approval" in lower or "consent" in lower:
        modalities.add("approval")

    if "claim" in lower:
        modalities.add("claim")

    if "terminate" in lower or "termination" in lower:
        modalities.add("termination")

    return {
        "tokens": set(tokens),
        "keywords": keyword_hits,
        "deadlines": set(deadlines),
        "modalities": modalities
    }


def nlp_relevance_score(anchor_text, candidate_text):
    a = extract_nlp_features(anchor_text)
    b = extract_nlp_features(candidate_text)

    token_overlap = len(a["tokens"] & b["tokens"]) / max(1, len(a["tokens"] | b["tokens"]))
    keyword_overlap = len(a["keywords"] & b["keywords"]) / max(1, len(a["keywords"] | b["keywords"]))
    modality_overlap = len(a["modalities"] & b["modalities"]) / max(1, len(a["modalities"] | b["modalities"]))

    deadline_signal = 0.0

    if a["deadlines"] and b["deadlines"]:
        if a["deadlines"] != b["deadlines"]:
            deadline_signal = 1.0
        else:
            deadline_signal = 0.5

    score = (
        0.35 * token_overlap +
        0.30 * keyword_overlap +
        0.20 * modality_overlap +
        0.15 * deadline_signal
    )

    return score


def retrieve_candidates_for_anchor(
    anchor_doc,
    anchor_index,
    bm25_k=20,
    dense_k=20,
    final_k=8,
    cross_document_only=False
):
    if bm25 is None or db is None:
        load_retrieval_stores()

    anchor_text = anchor_doc.page_content
    anchor_id = get_clause_id(anchor_doc)
    anchor_document_id = get_document_id(anchor_doc)

    tokens = tokenize(anchor_text)[:80]

    bm25_scores = bm25.get_scores(tokens)
    bm25_idx = np.argsort(bm25_scores)[::-1][:bm25_k]

    bm25_results = []

    for i in bm25_idx:
        if i == anchor_index:
            continue

        doc = bm25_docs[i]
        candidate_id = get_clause_id(doc)
        candidate_document_id = get_document_id(doc)

        if candidate_id == anchor_id:
            continue

        if cross_document_only and candidate_document_id == anchor_document_id:
            continue

        bm25_results.append({
            "text": doc.page_content,
            "doc_index": i,
            "clause_id": candidate_id,
            "display_id": get_display_id(doc),
            "document_id": candidate_document_id,
            "score": bm25_scores[i],
            "type": "bm25"
        })

    dense_raw = db.similarity_search_with_score(anchor_text, k=dense_k)

    dense_results = []

    for doc, score in dense_raw:
        candidate_id = get_clause_id(doc)
        candidate_document_id = get_document_id(doc)

        if candidate_id == anchor_id:
            continue

        if cross_document_only and candidate_document_id == anchor_document_id:
            continue

        dense_results.append({
            "text": doc.page_content,
            "doc_index": None,
            "clause_id": candidate_id,
            "display_id": get_display_id(doc),
            "document_id": candidate_document_id,
            "score": -score,
            "type": "dense"
        })

    bm25_n = norm([r["score"] for r in bm25_results])
    dense_n = norm([r["score"] for r in dense_results])

    for i, r in enumerate(bm25_results):
        r["bm25_norm"] = float(bm25_n[i]) if len(bm25_n) else 0.0
        r["dense_norm"] = 0.0

    for i, r in enumerate(dense_results):
        r["dense_norm"] = float(dense_n[i]) if len(dense_n) else 0.0
        r["bm25_norm"] = 0.0

    combined = bm25_results + dense_results

    seen = set()
    unique = []

    for r in combined:
        key = (r.get("display_id"), r["text"][:220])

        if key not in seen:
            seen.add(key)
            unique.append(r)

    for r in unique:
        r["nlp_score"] = nlp_relevance_score(anchor_text, r["text"])
        r["final_score"] = (
            0.35 * r.get("bm25_norm", 0.0) +
            0.35 * r.get("dense_norm", 0.0) +
            0.30 * r.get("nlp_score", 0.0)
        )

    ranked = sorted(unique, key=lambda x: x["final_score"], reverse=True)

    return ranked[:final_k]


def scan_contract(
    provider="ollama",
    model_name="llama3",
    max_anchors=None,
    bm25_k=20,
    dense_k=20,
    candidate_k=8,
    llm_candidate_k=5,
    save_csv=True,
    output_csv_path=None,
    cross_document_only=False,
    bm25_path=DEFAULT_BM25_PATH,
    chroma_path=DEFAULT_CHROMA_PATH
):
    load_retrieval_stores(bm25_path=bm25_path, chroma_path=chroma_path)

    print("\n🚀 CONTRACT CONTRADICTION SCAN STARTED")
    print(f"🤖 Provider: {provider}")
    print(f"🤖 Model: {model_name}")
    print(f"📄 Total documents/subclauses: {len(bm25_docs)}")
    print(f"🌐 Cross-document only: {cross_document_only}")

    results = []
    checked_pairs = set()

    docs_to_scan = bm25_docs[:max_anchors] if max_anchors else bm25_docs

    for anchor_index, anchor_doc in enumerate(docs_to_scan):
        anchor_text = anchor_doc.page_content
        anchor_id = get_clause_id(anchor_doc)
        anchor_display_id = get_display_id(anchor_doc)
        anchor_document_id = get_document_id(anchor_doc)

        print("\n" + "=" * 90)
        print(f"🔎 Anchor {anchor_index + 1}/{len(docs_to_scan)}: {anchor_display_id}")
        print(anchor_text[:500])

        candidates = retrieve_candidates_for_anchor(
            anchor_doc=anchor_doc,
            anchor_index=anchor_index,
            bm25_k=bm25_k,
            dense_k=dense_k,
            final_k=candidate_k,
            cross_document_only=cross_document_only
        )

        if not candidates:
            print("No candidates found.")
            continue

        rerank_start = time.time()

        reranked = ollama_rerank(
            anchor_clause=anchor_text,
            candidates=candidates,
            model=model_name,
            provider=provider
        )

        rerank_time = time.time() - rerank_start

        top_candidates = reranked[:llm_candidate_k]

        for rank, candidate in enumerate(top_candidates, start=1):
            candidate_id = candidate.get("clause_id", "unknown")
            candidate_display_id = candidate.get("display_id", candidate_id)
            candidate_document_id = candidate.get("document_id", "unknown_document")

            pair_key = tuple(sorted([anchor_id, candidate_id]))

            if pair_key in checked_pairs:
                continue

            checked_pairs.add(pair_key)

            start = time.time()

            result = check_contradiction(
                clause_a=anchor_text,
                clause_b=candidate["text"],
                model=model_name,
                provider=provider
            )

            checker_time = time.time() - start

            label = result.get("label", "Unknown")

            print(f"\nCandidate {rank}: {candidate_display_id}")
            print(f"Label: {label}")
            print(f"Conflict Type: {result.get('conflict_type', 'none')}")
            print(f"Confidence: {result.get('confidence', 0.0)}")
            print(f"Reason: {result.get('reason', '')}")

            row = {
                "anchor_id": anchor_display_id,
                "candidate_id": candidate_display_id,
                "anchor_document_id": anchor_document_id,
                "candidate_document_id": candidate_document_id,
                "cross_document": anchor_document_id != candidate_document_id,
                "provider": provider,
                "model": model_name,
                "candidate_rank": rank,
                "hybrid_score": round(candidate.get("final_score", 0.0), 4),
                "nlp_score": round(candidate.get("nlp_score", 0.0), 4),
                "llm_score": round(candidate.get("llm_score", 0.0), 4),
                "confidence": result.get("confidence", 0.0),
                "label": label,
                "conflict_type": result.get("conflict_type", "none"),
                "reason": result.get("reason", ""),
                "evidence": json.dumps(result.get("evidence", []), ensure_ascii=False),
                "anchor_text": anchor_text,
                "candidate_text": candidate["text"],
                "rerank_time_sec": round(rerank_time, 3),
                "checker_time_sec": round(checker_time, 3)
            }

            results.append(row)

    if save_csv:
        Path("evaluation_outputs").mkdir(exist_ok=True)

        if output_csv_path:
            out_path = Path(output_csv_path)
            out_path.parent.mkdir(exist_ok=True)
        else:
            safe_model = model_name.replace(":", "_").replace("/", "_")
            out_path = Path("evaluation_outputs") / f"auto_scan_{provider}_{safe_model}.csv"

        fieldnames = [
            "anchor_id", "candidate_id", "anchor_document_id", "candidate_document_id",
            "cross_document", "provider", "model", "candidate_rank", "hybrid_score",
            "nlp_score", "llm_score", "confidence", "label", "conflict_type",
            "reason", "evidence", "anchor_text", "candidate_text", "rerank_time_sec",
            "checker_time_sec"
        ]

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            if results:
                writer.writerows(results)

        print(f"\n💾 Saved full scan to: {out_path}")

    print("\n✅ SCAN COMPLETE")

    label_counts = Counter([r["label"] for r in results])
    print("\n📊 Label summary:")
    for label, count in label_counts.items():
        print(f"{label}: {count}")

    print("\n🚩 Potential issues found:")
    for r in results:
        if r["label"] in ["CONTRADICTION", "MATERIAL_INCONSISTENCY"]:
            print(f"- {r['anchor_id']} ↔ {r['candidate_id']} | {r['label']} | {r['conflict_type']}")
            print(f"  Reason: {r['reason']}")

    return results


if __name__ == "__main__":
    provider = input("Provider [ollama/openai]: ").strip().lower() or "ollama"

    if provider == "openai":
        model_name = input("OpenAI model [gpt-5.5 | gpt-4o-mini]: ").strip() or "gpt-5.5"
    else:
        model_name = input("Ollama model [llama3 | phi4-mini | qwen2.5:14b]: ").strip() or "llama3"

    max_input = input("Max anchors to scan? [press Enter for all, or type number]: ").strip()
    max_anchors = int(max_input) if max_input else None

    cross_input = input("Cross-document only? (y/n) [n]: ").strip().lower()
    cross_document_only = cross_input == "y"

    scan_contract(
        provider=provider,
        model_name=model_name,
        max_anchors=max_anchors,
        bm25_k=20,
        dense_k=20,
        candidate_k=8,
        llm_candidate_k=5,
        save_csv=True,
        cross_document_only=cross_document_only
    )
