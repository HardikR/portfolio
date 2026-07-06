import requests
import json
import re
from typing import List, Dict, Any

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

OLLAMA_URL = "http://localhost:11434/api/generate"


def _extract_json_array(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except Exception:
        pass

    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        raise ValueError("No JSON array found in model output")

    return json.loads(match.group(0))


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
                    "content": "You are a senior legal advisor. Return only valid JSON."
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


def ollama_rerank(
    anchor_clause: str,
    candidates: List[Dict[str, Any]],
    model: str = "llama3",
    provider: str = "ollama"
):
    print(f"🧠 Reranking candidates with provider={provider}, model={model}")

    if not candidates:
        return []

    prompt = f"""
You are a senior legal advisor.

Rank the candidate clauses by how likely they are to conflict, contradict, or be inconsistent with the anchor clause.

ONE-SHOT EXAMPLE:

Anchor clause:
The Service Provider may make a claim within 10 Business Days after becoming aware of the event.

Candidate clauses:
[0] The Service Provider may make a claim within 5 Business Days after becoming aware of the same type of event.
[1] The Service Provider must keep records of all claims.
[2] The Service Provider must comply with safety law.

Correct JSON answer:
[
  {{"index": 0, "score": 0.98}},
  {{"index": 1, "score": 0.45}},
  {{"index": 2, "score": 0.10}}
]

Ranking rules:
- Rank highest if same actor, same legal mechanism, and different deadline, condition, approval, permission, prohibition, responsibility, or scope.
- Timing mismatch is highly important.
- Same broad topic is not enough unless the legal effect overlaps.
- Return ONLY valid JSON.

Return format:
[
  {{"index": 0, "score": 0.95}},
  {{"index": 1, "score": 0.80}}
]

Anchor clause:
{anchor_clause}

Candidate clauses:
"""

    for i, c in enumerate(candidates):
        prompt += f"\n[{i}] {c['text'][:1400]}\n"

    try:
        output = call_llm(prompt, model=model, provider=provider)
        rankings = _extract_json_array(output)

    except Exception as e:
        print(f"⚠️ LLM RERANK ERROR: {e}")
        return candidates

    scored = []

    for c in candidates:
        c = dict(c)
        c["llm_score"] = 0.5
        scored.append(c)

    for r in rankings:
        idx = r.get("index")
        score = r.get("score", 0.5)

        if isinstance(idx, int) and 0 <= idx < len(scored):
            try:
                scored[idx]["llm_score"] = float(score)
            except Exception:
                scored[idx]["llm_score"] = 0.5

    return sorted(scored, key=lambda x: x["llm_score"], reverse=True)