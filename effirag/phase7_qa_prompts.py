from __future__ import annotations

from typing import Any


VALID_QA_PROMPT_MODES = {
    "current_phase7",
    "lightrag_short",
    "phase7_short",
}


def resolve_qa_prompt_mode(cfg: Any = None) -> str:
    mode = str(getattr(cfg, "qa_prompt_mode", "current_phase7") or "current_phase7").strip().lower()
    if mode not in VALID_QA_PROMPT_MODES:
        mode = "current_phase7"
    return mode


def build_lightrag_short_prompt(*, question: str, context: str, user_prompt: str = "") -> str:
    parts = [
        "You are a helpful assistant answering questions from the provided Context.",
        "Use only facts from Context.",
        "Return only the final short answer span.",
        "No markdown, no explanations, no citations, and no prefixes.",
        "For yes/no questions, output exactly yes or no in lowercase.",
        "If the answer cannot be found in Context, output exactly: insufficient information",
        f"Question: {question}",
    ]
    user_prompt = str(user_prompt or "").strip()
    if user_prompt:
        parts.append(f"Additional Instructions: {user_prompt}")
    parts.extend(
        [
            "Context:",
            str(context or ""),
            "Answer:",
        ]
    )
    return "\n".join(parts)


def build_phase7_short_prompt(*, question: str, context: str, user_prompt: str = "") -> str:
    return "\n".join(
        [
            "---Role---",
            "",
            "You are an expert AI assistant specializing in short answer extraction from a provided knowledge base.",
            "",
            "---Goal---",
            "",
            "Answer the user query using ONLY the provided Context.",
            "Return only the final short answer span.",
            "",
            "---Instructions---",
            "",
            "1. Grounding:",
            "  - Use only facts explicitly present in the Context.",
            "  - Do not use outside knowledge.",
            "  - Context lines may be formatted as [Title] sentence; the title is part of the evidence and may be used to resolve entities.",
            "  - You may combine multiple context lines when the required reasoning chain is explicitly supported by the Context.",
            "  - If the answer cannot be determined from the Context, output exactly: insufficient information",
            "",
            "2. Multi-hop composition:",
            "  - For relationship questions, follow explicit links across context lines.",
            "  - For comparison questions, compare the relevant attributes only when both sides are explicitly supported by the Context.",
            "  - For yes/no questions, output yes or no only when the Context explicitly supports that answer. If not, output insufficient information.",
            "",
            "3. Reasoning policy:",
            "  - Reason internally, but do not reveal chain-of-thought.",
            "  - Do not output intermediate analysis.",
            "",
            "4. Output format:",
            "  - Output exactly one line with only the final answer text.",
            "  - Return the shortest correct answer span.",
            "  - Do not output Markdown, bullets, headings, JSON, or code blocks.",
            "  - Do not output citations or a references section.",
            "  - Do not output prefixes such as \"Answer:\", \"Final answer:\", or \"So the answer is:\".",
            "  - For yes/no questions with explicit support, output exactly yes or no in lowercase.",
            "",
            "5. Language:",
            "  - Use the same language as the user query unless the answer is a proper noun.",
            "",
            f"6. Additional Instructions: {str(user_prompt or '').strip()}",
            "",
            "---Context---",
            "",
            str(context or ""),
            "",
            f"Question: {question}",
            "Answer:",
        ]
    )
