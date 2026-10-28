import os

import requests

from .rag_chain import build_context, retrieve_records


def generate_with_gemini(prompt: str, api_key: str, model: str) -> str:
    if not api_key:
        raise ValueError(
            "Gemini is not configured. Add GEMINI_API_KEY to Streamlit secrets."
        )

    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent",
        params={"key": api_key},
        json={"contents": [{"parts": [{"text": prompt}]}]},
        timeout=120,
    )
    if not response.ok:
        detail = response.text.strip().replace("\n", " ")[:500]
        raise RuntimeError(f"Gemini returned HTTP {response.status_code}: {detail}")

    candidates = response.json().get("candidates", [])
    try:
        answer = candidates[0]["content"]["parts"][0]["text"].strip()
    except (IndexError, KeyError, TypeError) as error:
        raise RuntimeError("Gemini returned an empty response") from error
    if not answer:
        raise RuntimeError("Gemini returned an empty response")
    return answer


def answer_query_with_gemini(
    query: str,
    api_key: str,
    model: str | None = None,
) -> dict:
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")

    selected_records = retrieve_records(query)
    prompt = (
        "Answer the question using only the context below. "
        "If the context does not contain the answer, say you do not know.\n\n"
        f"Context:\n{build_context(selected_records)}\n\n"
        f"Question: {query}\nAnswer:"
    )
    generated = generate_with_gemini(
        prompt,
        api_key,
        model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    )
    sources = [
        {
            "source": record.get("source", ""),
            "title": record.get("title", ""),
        }
        for record in selected_records
    ]
    return {"answer": generated, "sources": sources}
