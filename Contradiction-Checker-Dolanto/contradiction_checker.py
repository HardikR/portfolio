import requests
import json
import re
from typing import Dict, Any

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


OLLAMA_URL = "http://localhost:11434/api/generate"


VALID_LABELS = {
    "CONTRADICTION",
    "MATERIAL_INCONSISTENCY",
    "NOT_CONTRADICTORY"
}

VALID_CONFLICT_TYPES = {
    "temporal",
    "numerical",
    "authority",
    "process",
    "policy_reversal",
    "specificity",
    "none"
}


def _extract_json_object(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("No JSON object found in model output")

    return json.loads(match.group(0))


def normalize_label(label: str) -> str:
    label = str(label).strip().upper()
    label = label.replace(" ", "_").replace("-", "_").replace("/", "_")

    if label in ["CONTRADICTION", "CONTRADICTORY", "TRUE_CONTRADICTION"]:
        return "CONTRADICTION"

    if label in [
        "MATERIAL_INCONSISTENCY",
        "INCONSISTENCY",
        "INCONSISTENT",
        "MATERIAL_CONFLICT",
        "LEGAL_INCONSISTENCY"
    ]:
        return "MATERIAL_INCONSISTENCY"

    if label in [
        "NOT_CONTRADICTORY",
        "NOT_CONTRADICTION",
        "NO_CONTRADICTION",
        "CONSISTENT",
        "RELATED",
        "NONE",
        "UNCLEAR",
        "NOT_COMPARABLE",
        "UNCLEAR_NOT_COMPARABLE"
    ]:
        return "NOT_CONTRADICTORY"

    return "NOT_CONTRADICTORY"


def normalize_conflict_type(conflict_type: str) -> str:
    conflict_type = str(conflict_type).strip().lower()
    conflict_type = conflict_type.replace(" ", "_").replace("-", "_").replace("/", "_")

    mapping = {
        "time": "temporal",
        "timing": "temporal",
        "date": "temporal",
        "deadline": "temporal",
        "duration": "temporal",
        "sequence": "temporal",

        "number": "numerical",
        "amount": "numerical",
        "value": "numerical",
        "percentage": "numerical",
        "money": "numerical",
        "quantitative": "numerical",
        "frequency": "numerical",
        "threshold": "numerical",

        "source": "authority",
        "issuer": "authority",
        "decision_maker": "authority",
        "approval_authority": "authority",

        "procedure": "process",
        "workflow": "process",
        "route": "process",
        "method": "process",
        "cross_document": "process",

        "direct": "policy_reversal",
        "deontic": "policy_reversal",
        "direct_negation": "policy_reversal",
        "reversal": "policy_reversal",
        "permission_prohibition": "policy_reversal",
        "obligation_prohibition": "policy_reversal",
        "permitted_prohibited": "policy_reversal",
        "required_optional": "policy_reversal",

        "scope": "specificity",
        "scope_exception": "specificity",
        "general_specific": "specificity",
        "broad_narrow": "specificity",
        "specific_exception": "specificity",
        "conditional_override": "specificity",
        "force_majeure_exception": "specificity",
        "liability_limitation": "specificity",
        "remedy_limitation": "specificity",
        "carve_out": "specificity",
        "qualification": "specificity",
        "exception": "specificity",

        "none": "none",
        "": "none"
    }

    conflict_type = mapping.get(conflict_type, conflict_type)

    if conflict_type not in VALID_CONFLICT_TYPES:
        return "none"

    return conflict_type


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
        if value < 0:
            return 0.0
        if value > 1:
            return 1.0
        return value
    except Exception:
        return default


def call_llm(prompt: str, model: str, provider: str = "ollama") -> str:
    provider = provider.lower().strip()

    if provider == "openai":
        if OpenAI is None:
            raise ImportError("OpenAI package not installed. Run: python -m pip install openai")

        client = OpenAI()

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a legal contradiction analysis model. Return only valid JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.choices[0].message.content

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )

    response.raise_for_status()
    return response.json()["response"]


def check_contradiction(
    clause_a: str,
    clause_b: str,
    model: str = "llama3",
    provider: str = "ollama"
) -> Dict[str, Any]:

    prompt = f"""
You are a legal contradiction analysis model. Your task is to compare two legal clauses, passages, or document snippets and decide whether they are:

1. CONSISTENT
2. CONTRADICTORY
3. UNCLEAR / NOT COMPARABLE

A contradiction exists when both statements cannot be true at the same time in the same legal scope. Focus on the legal effect, not just surface wording.

Important contradiction types to detect:
- Direct contradiction: one clause says something is allowed/required, the other says it is not.
- Deontic contradiction: conflict in legal force, such as must vs may, permitted vs prohibited, required vs optional.
- Temporal contradiction: conflict in dates, deadlines, durations, start/end times, or sequence.
- Quantitative contradiction: conflict in numbers, thresholds, percentages, amounts, or frequencies.
- Scope or exception contradiction: one clause is broad, the other narrows, limits, or exempts part of it in a way that changes the meaning.
- Cross-document contradiction: the two clauses come from different documents but describe the same obligation, right, or rule differently.

How to reason:
- Compare the same subject, action, party, condition, and legal effect.
- Check whether one clause negates, reverses, or materially limits the other.
- If there is an exception case in the clauses that you compare, that is legally correct, use your reasoning to mark it as contradicting or consistent.
- Pay attention to exceptions like “unless,” “except,” “notwithstanding,” and “provided that.”
- Do not mark a contradiction if one clause only adds detail, examples, or extra requirements that can coexist.
- If the clauses refer to different parties, different events, or different scopes, mark UNCLEAR / NOT COMPARABLE unless there is still a direct conflict.
- If one clause is more specific and the other is general, determine whether the specific clause is a valid exception or a true conflict.

Output format:
Return only valid JSON with these keys:
{{
  "label": "CONSISTENT | CONTRADICTORY | UNCLEAR",
  "documentID": "document number or ID as per input | mention multiple values if there are multiple documents",
  "clause_ID": "clause ID as per input | mention both clause IDs that are contradicting",
  "contradiction_type": "direct | deontic | temporal | quantitative | scope_exception | cross_document | none",
  "confidence": 0.0,
  "reason": "one concise explanation",
  "evidence": ["short supporting phrase from clause 1", "short supporting phrase from clause 2"]
}}

Examples:

Example 1
Clause A: "The tenant must pay rent by the first day of each month."
Clause B: "The tenant is not required to pay rent by the first day of each month."
Output:
{{
  "label": "CONTRADICTORY",
  "documentID": "document 1",
  "clause_ID": "clause 3.2 and clause 4.1",
  "contradiction_type": "deontic",
  "confidence": 0.98,
  "reason": "One clause imposes a duty while the other removes it.",
  "evidence": ["must pay rent", "not required to pay rent"]
}}

Example 2
Clause A: "The agreement begins on January 1, 2026."
Clause B: "The agreement begins on March 1, 2026."
Output:
{{
  "label": "CONTRADICTORY",
  "documentID": "document 1",
  "clause_ID": "clause 3.2 and clause 4.1",
  "contradiction_type": "temporal",
  "confidence": 0.97,
  "reason": "The clauses assign different start dates to the same agreement.",
  "evidence": ["January 1, 2026", "March 1, 2026"]
}}

Example 3
Clause A: "All employees must work onsite three days per week."
Clause B: "Employees in the engineering team are exempt from onsite attendance."
Output:
{{
  "label": "CONTRADICTORY",
  "documentID": "document 1",
  "clause_ID": "clause 3.2 and clause 4.1",
  "contradiction_type": "scope_exception",
  "confidence": 0.86,
  "reason": "The second clause creates an exception to the general onsite requirement.",
  "evidence": ["All employees must work onsite", "are exempt from onsite attendance"]
}}

Example 4
Clause A: "The vendor may retain records for seven years."
Clause B: "The vendor must delete all records immediately after termination."
Output:
{{
  "label": "CONTRADICTORY",
  "documentID": "document 1",
  "clause_ID": "clause 3.2 and clause 4.1",
  "contradiction_type": "scope_exception",
  "confidence": 0.84,
  "reason": "The retention permission conflicts with the deletion requirement for the same records after termination.",
  "evidence": ["retain records for seven years", "must delete all records immediately after termination"]
}}

Example 5
Clause A: "Payment is due within 10 business days."
Clause B: "Payment is due within 30 business days."
Output:
{{
  "label": "CONTRADICTORY",
  "documentID": "document 1",
  "clause_ID": "clause 3.2 and clause 4.1",
  "contradiction_type": "quantitative",
  "confidence": 0.96,
  "reason": "The clauses disagree on the payment deadline.",
  "evidence": ["within 10 business days", "within 30 business days"]
}}

Example 6
Clause A: "The supplier may terminate the agreement with 30 days' notice."
Clause B: "The supplier may terminate the agreement with 30 days' notice for convenience."
Output:
{{
  "label": "CONSISTENT",
  "documentID": "document 1",
  "clause_ID": "clause 3.2 and clause 4.1",
  "contradiction_type": "none",
  "confidence": 0.91,
  "reason": "The second clause adds detail but does not conflict with the first.",
  "evidence": ["may terminate", "with 30 days' notice"]
}}

Example 7
Clause A: "The client must approve all expenditures in advance."
Clause B: "The client does not need to approve travel-related expenditures in advance."
Output:
{{
  "label": "CONTRADICTORY",
  "documentID": "document 1",
  "clause_ID": "clause 3.2 and clause 4.1",
  "contradiction_type": "scope_exception",
  "confidence": 0.88,
  "reason": "The second clause creates an exception to the general approval requirement.",
  "evidence": ["approve all expenditures in advance", "does not need to approve travel-related expenditures"]
}}

Now analyze the input clauses carefully and classify them using the rules above. The numbers and text in the examples are only placeholders, do not look exactly for them in the dataset.

Clause A:
{clause_a}

Clause B:
{clause_b}
"""

    try:
        result = call_llm(prompt, model=model, provider=provider)
        parsed = _extract_json_object(result)

        label = normalize_label(parsed.get("label", "NOT_CONTRADICTORY"))

        raw_conflict_type = parsed.get(
            "conflict_type",
            parsed.get("contradiction_type", "none")
        )
        conflict_type = normalize_conflict_type(raw_conflict_type)

        if label == "NOT_CONTRADICTORY":
            conflict_type = "none"

        return {
            "label": label,
            "reason": parsed.get("reason", "No reason returned"),
            "conflict_type": conflict_type,
            "confidence": safe_float(parsed.get("confidence", 0.0)),
            "evidence": parsed.get("evidence", []),
            "documentID": parsed.get("documentID", parsed.get("document_ID", "")),
            "clause_ID": parsed.get("clause_ID", "")
        }

    except Exception as e:
        print("⚠️ LLM failed:", e)

        return {
            "label": "NOT_CONTRADICTORY",
            "reason": "LLM failed or invalid JSON returned",
            "conflict_type": "none",
            "confidence": 0.0,
            "evidence": [],
            "documentID": "",
            "clause_ID": ""
        }
