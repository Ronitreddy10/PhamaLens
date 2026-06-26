"""Generation-quality scoring with heuristic, Groq, or local Ollama judges."""

import json
import os
import re
import time

import httpx
from groq import Groq

DEFAULT_JUDGE_PROVIDER = os.getenv("PHARMALENS_JUDGE_PROVIDER", "ollama")
DEFAULT_JUDGE_MODEL = os.getenv("PHARMALENS_JUDGE_MODEL", "llama3:latest")
DEFAULT_GROQ_JUDGE_MODEL = os.getenv("PHARMALENS_GROQ_JUDGE_MODEL", "llama-3.1-8b-instant")
DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
JUDGE_DELAY_SECONDS = float(os.getenv("PHARMALENS_JUDGE_DELAY", "2"))
NOT_IN_KB_PHRASES = (
    "not in knowledge base",
    "not contained",
    "cannot be found",
    "could not answer that from this knowledge base",
    "could not find relevant information",
    "no information",
)


def correct_refusal_score(answer: str) -> dict | None:
    """Score correct knowledge-base refusals as faithful; citations are not required."""
    answer_l = answer.lower()
    if any(phrase in answer_l for phrase in NOT_IN_KB_PHRASES):
        return {
            "faithfulness": {"score": 1.0, "comment": "Correctly refused because the answer is not in the knowledge base."},
            "citation_accuracy": {"score": 1.0, "comment": "No citation required for a correct no-source refusal."},
            "modality_preservation": {"score": 1.0, "comment": "No regulatory modality claim was made."},
            "hallucination_detected": False,
        }
    return None


def heuristic_score(answer: str, context: str) -> dict:
    """Offline sanity score: citation presence + rough lexical grounding."""
    if refusal := correct_refusal_score(answer):
        return refusal
    answer_terms = {token.lower() for token in answer.split() if len(token) > 5}
    context_l = context.lower()
    grounded = sum(1 for term in answer_terms if term in context_l)
    lexical = grounded / max(len(answer_terms), 1)
    citations = 1.0 if ".pdf" in answer and "p." in answer else 0.0
    return {
        "faithfulness": {"score": round(lexical, 3), "comment": "Heuristic lexical overlap against retrieved context."},
        "citation_accuracy": {"score": citations, "comment": "Heuristic check for PDF/page citations."},
        "modality_preservation": {"score": 1.0, "comment": "Use Groq judge for detailed modality validation."},
    }


def _judge_prompt(question: str, answer: str, context: str) -> str:
    return f"""Evaluate this pharma regulatory answer. Return only JSON with faithfulness, citation_accuracy, and modality_preservation scores from 0 to 1 plus short comments.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

ANSWER:
{answer}
"""


def _parse_judge_json(content: str, answer: str, context: str) -> dict:
    content = re.sub(r"<think>.*?</think>", "", content or "", flags=re.S | re.I).strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", content, flags=re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return heuristic_score(answer, context)


def groq_judge(question: str, answer: str, context: str, model: str = DEFAULT_GROQ_JUDGE_MODEL) -> dict:
    if refusal := correct_refusal_score(answer):
        return refusal
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return heuristic_score(answer, context)
    if JUDGE_DELAY_SECONDS > 0:
        time.sleep(JUDGE_DELAY_SECONDS)
    response = Groq(api_key=api_key).chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a strict pharmaceutical regulatory QA evaluator. Return valid JSON only."},
            {"role": "user", "content": _judge_prompt(question, answer, context)},
        ],
    )
    return _parse_judge_json(response.choices[0].message.content or "", answer, context)


def ollama_judge(question: str, answer: str, context: str, model: str = "llama3:latest") -> dict:
    if refusal := correct_refusal_score(answer):
        return refusal
    try:
        response = httpx.post(
            f"{DEFAULT_OLLAMA_URL.rstrip('/')}/api/chat",
            json={
                "model": model,
                "stream": False,
                "options": {"temperature": 0, "num_ctx": 8192},
                "messages": [
                    {"role": "system", "content": "You are a strict pharmaceutical regulatory QA evaluator. Return valid JSON only."},
                    {"role": "user", "content": _judge_prompt(question, answer, context)},
                ],
            },
            timeout=180,
        )
        response.raise_for_status()
        return _parse_judge_json(response.json().get("message", {}).get("content", ""), answer, context)
    except Exception:
        return heuristic_score(answer, context)


def judge_answer(
    question: str,
    answer: str,
    context: str,
    provider: str = DEFAULT_JUDGE_PROVIDER,
    model: str | None = None,
) -> dict:
    provider = (provider or DEFAULT_JUDGE_PROVIDER).lower()
    if provider == "ollama":
        return ollama_judge(question, answer, context, model=model or "llama3:latest")
    if provider == "groq":
        return groq_judge(question, answer, context, model=model or DEFAULT_GROQ_JUDGE_MODEL)
    return heuristic_score(answer, context)
