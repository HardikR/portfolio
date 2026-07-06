import json
import os
import sys
import importlib.util
import tempfile
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

from ingestion import ingest_json_files
from retrieval import scan_contract

EXTRACTOR_IMPORT_ERROR = None
extractor = None


def load_extractor_module():
    """Load dvs_extract_paddleocrvl.py from the same folder as this app."""
    global extractor, EXTRACTOR_IMPORT_ERROR

    if extractor is not None:
        return extractor

    module_path = Path(__file__).resolve().parent / "dvs_extract_paddleocrvl.py"

    if not module_path.exists():
        EXTRACTOR_IMPORT_ERROR = f"File not found: {module_path}"
        return None

    try:
        spec = importlib.util.spec_from_file_location("dvs_extract_paddleocrvl", module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["dvs_extract_paddleocrvl"] = module
        spec.loader.exec_module(module)
        extractor = module
        EXTRACTOR_IMPORT_ERROR = None
        return extractor
    except Exception:
        EXTRACTOR_IMPORT_ERROR = traceback.format_exc()
        extractor = None
        return None


# ==============================================================================
# Streamlit page setup
# ==============================================================================

st.set_page_config(
    page_title="Legal Contradiction Checker",
    page_icon="⚖️",
    layout="wide",
)

st.title("⚖️ Legal Contradiction Checker")
st.caption(
    "Upload contract PDFs or extracted JSON files. PDFs are automatically converted "
    "to the required JSON structure before the contradiction scan runs."
)


# ==============================================================================
# Helpers
# ==============================================================================

CONTRADICTION_LABELS = {"CONTRADICTION", "MATERIAL_INCONSISTENCY"}


def safe_file_name(name: str) -> str:
    """Keep uploaded filenames safe when writing them into a temp folder."""
    return Path(name).name.replace("\x00", "")


def progress_callback(done: int, total: int, page_no: int, message: str):
    """Callback used by the PDF extraction pipeline."""
    if done == -1:
        st.session_state.extract_stage = message
    else:
        st.session_state.extract_done = done
        st.session_state.extract_total = total
        st.session_state.extract_message = message


def write_uploaded_files(uploaded_files, temp_dir: str) -> Tuple[List[str], List[str]]:
    """
    Save uploaded PDFs/JSONs to disk.

    Returns:
        pdf_paths: uploaded PDF paths
        json_paths: uploaded JSON paths
    """
    pdf_paths: List[str] = []
    json_paths: List[str] = []

    for uploaded_file in uploaded_files:
        name = safe_file_name(uploaded_file.name)
        path = os.path.join(temp_dir, name)
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        suffix = Path(name).suffix.lower()
        if suffix == ".pdf":
            pdf_paths.append(path)
        elif suffix == ".json":
            json_paths.append(path)

    return pdf_paths, json_paths


def extract_pdf_to_json(pdf_path: str, temp_dir: str) -> str:
    """
    Run dvs_extract_paddleocrvl.py on one PDF and write the returned extraction
    JSON to a file that ingestion.py can consume.
    """
    module = load_extractor_module()
    if module is None:
        raise ImportError(
            "Could not import dvs_extract_paddleocrvl.py from the same folder as app.py.\n\n"
            f"Technical import error:\n{EXTRACTOR_IMPORT_ERROR or 'Unknown import error'}"
        )

    pdf_path_obj = Path(pdf_path)
    out_dir = Path(temp_dir) / "extracted_json"
    out_dir.mkdir(parents=True, exist_ok=True)

    # toc_pages is intentionally None here because the UI flow should be automatic.
    # The extraction script will still create the clean {clauses, tables} JSON.
    result = module.run(
        pdf_path=pdf_path_obj,
        output_dir=out_dir,
        toc_pages=None,
        progress_callback=progress_callback,
    )

    if not result or not isinstance(result, dict):
        raise ValueError(f"Extraction returned no valid JSON for {pdf_path_obj.name}")

    json_path = out_dir / f"{pdf_path_obj.stem}_extracted.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return str(json_path)


def prepare_json_inputs(uploaded_files) -> Tuple[List[str], Dict[str, int]]:
    """
    Save uploaded files. Convert every PDF to JSON. Return JSON paths and summary.
    """
    temp_dir = tempfile.mkdtemp(prefix="legal_contradiction_uploads_")
    pdf_paths, json_paths = write_uploaded_files(uploaded_files, temp_dir)

    summary = {
        "pdf_files": len(pdf_paths),
        "json_files": len(json_paths),
        "extracted_pdfs": 0,
    }

    if pdf_paths:
        st.info(
            "Extracting PDF text and clauses first. Scanned PDFs may take a long time "
            "because OCR must read each page."
        )
        for pdf_path in pdf_paths:
            with st.status(f"Extracting {Path(pdf_path).name}", expanded=False) as status:
                try:
                    extracted_json = extract_pdf_to_json(pdf_path, temp_dir)
                    json_paths.append(extracted_json)
                    summary["extracted_pdfs"] += 1
                    status.update(label=f"Extracted {Path(pdf_path).name}", state="complete")
                except Exception as e:
                    status.update(label=f"Extraction failed for {Path(pdf_path).name}", state="error")
                    raise RuntimeError(
                        f"PDF extraction failed for {Path(pdf_path).name}: {e}"
                    )

    return json_paths, summary


def build_simple_results(df: pd.DataFrame, issues_only: bool = True) -> pd.DataFrame:
    """
    Convert the full scan output into the simplified UI/report format requested.
    """
    if df.empty:
        return df

    view = df.copy()
    if issues_only and "label" in view.columns:
        view = view[view["label"].isin(CONTRADICTION_LABELS)].copy()

    expected = {
        "anchor_id": "",
        "anchor_text": "",
        "candidate_id": "",
        "candidate_text": "",
        "reason": "",
        "nlp_score": 0.0,
        "confidence": 0.0,
        "llm_score": 0.0,
        "hybrid_score": 0.0,
        "conflict_type": "none",
        "label": "NOT_CONTRADICTORY",
    }
    for col, default in expected.items():
        if col not in view.columns:
            view[col] = default

    simple = pd.DataFrame(
        {
            "Contradiction Clause Number": view["anchor_id"],
            "Contradiction Clause Reads": view["anchor_text"],
            "Contradicting Clause Number": view["candidate_id"],
            "Contradicting Clause Reads": view["candidate_text"],
            "Reason Why They Contradict": view["reason"],
            "NLP Score": view["nlp_score"].astype(float).round(4),
            "Confidence Score": view["confidence"].astype(float).round(4),
            "LLM Score": view["llm_score"].astype(float).round(4),
            "Hybrid Score": view["hybrid_score"].astype(float).round(4),
            "Conflict Type": view["conflict_type"],
            "Label": view["label"],
        }
    )

    return simple.reset_index(drop=True)


# ==============================================================================
# Sidebar controls — model selection and candidate selection only
# ==============================================================================

with st.sidebar:
    st.header("Model Selection")
    provider = st.selectbox("Provider", ["ollama", "openai"], index=0)

    if provider == "openai":
        model_name = st.text_input("Model", value="gpt-5.5")
    else:
        model_name = st.text_input("Model", value="llama3")

    st.header("Candidate Selection")
    llm_candidate_k = st.number_input(
        "Candidates checked per clause",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
        help="How many top candidates are sent to the contradiction checker.",
    )

# Internal retrieval setting hidden from the UI.
# The user only chooses how many final candidates the LLM checks.
candidate_k = max(8, int(llm_candidate_k) * 4)


# ==============================================================================
# Upload and run
# ==============================================================================

uploaded_files = st.file_uploader(
    "Upload PDF or JSON file(s)",
    type=["pdf", "json"],
    accept_multiple_files=True,
    help=(
        "Single file = scan within that document. Multiple files = automatic "
        "cross-document contradiction checking."
    ),
)

run_scan = st.button("Run contradiction scan", type="primary")

if run_scan:
    if not uploaded_files:
        st.error("Please upload at least one PDF or JSON file.")
        st.stop()

    st.session_state.extract_stage = "Starting extraction"
    st.session_state.extract_done = 0
    st.session_state.extract_total = 0
    st.session_state.extract_message = ""

    uploaded_count = len(uploaded_files)
    cross_document_only = uploaded_count > 1

    try:
        json_paths, input_summary = prepare_json_inputs(uploaded_files)
    except Exception as e:
        st.error(str(e))
        with st.expander("Show technical details"):
            st.code(traceback.format_exc())
        st.stop()

    if not json_paths:
        st.error("No usable JSON was produced from the uploaded files.")
        st.stop()

    scan_mode = "cross-document only" if cross_document_only else "single-document scan"
    st.info(f"Building retrieval stores from {len(json_paths)} JSON file(s). Mode: {scan_mode}.")

    try:
        summary = ingest_json_files(json_paths=json_paths, reset=True)
    except Exception as e:
        st.error(f"Ingestion failed: {e}")
        with st.expander("Show technical details"):
            st.code(traceback.format_exc())
        st.stop()

    st.success(
        f"Loaded {summary['clauses_loaded']} clauses and created "
        f"{summary['documents_created']} searchable clause/subclause records."
    )

    st.info("Running retrieval, LLM reranking, and contradiction checking...")

    try:
        results = scan_contract(
            provider=provider,
            model_name=model_name,
            max_anchors=None,
            bm25_k=20,
            dense_k=20,
            candidate_k=int(candidate_k),
            llm_candidate_k=int(llm_candidate_k),
            save_csv=True,
            output_csv_path="evaluation_outputs/ui_scan_results.csv",
            cross_document_only=cross_document_only,
        )
    except Exception as e:
        st.error(f"Scan failed: {e}")
        with st.expander("Show technical details"):
            st.code(traceback.format_exc())
        st.stop()

    full_df = pd.DataFrame(results)
    st.success("Scan complete.")

    if full_df.empty:
        st.warning("No candidate pairs were checked or no results were produced.")
        st.stop()

    simple_issues = build_simple_results(full_df, issues_only=True)
    simple_all = build_simple_results(full_df, issues_only=False)

    total_pairs = len(full_df)
    issue_count = int(full_df["label"].isin(CONTRADICTION_LABELS).sum()) if "label" in full_df else 0
    cross_pairs = int(full_df["cross_document"].sum()) if "cross_document" in full_df else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Pairs checked", total_pairs)
    c2.metric("Contradictions found", issue_count)
    c3.metric("Cross-document pairs", cross_pairs)

    st.subheader("Contradiction Results")
    if simple_issues.empty:
        st.write("No contradictions or material inconsistencies found.")
    else:
        st.dataframe(simple_issues, use_container_width=True, hide_index=True)

    with st.expander("Show all checked pairs"):
        st.dataframe(simple_all, use_container_width=True, hide_index=True)

    csv_issues = simple_issues.to_csv(index=False).encode("utf-8")
    csv_all = simple_all.to_csv(index=False).encode("utf-8")

    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button(
            label="Download contradiction results CSV",
            data=csv_issues,
            file_name="contradiction_results_simple.csv",
            mime="text/csv",
        )
    with col_b:
        st.download_button(
            label="Download all checked pairs CSV",
            data=csv_all,
            file_name="all_checked_pairs_simple.csv",
            mime="text/csv",
        )
else:
    st.info(
        "Upload one PDF/JSON to scan within a document, or upload multiple files "
        "to automatically run cross-document checking."
    )
