from __future__ import annotations

"""Prompt templates for LLM strategies (Self-Refine, CoVe).

These are intentionally centralized so notebooks and the API share the same
wording and can be updated without touching the strategy orchestration code.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyPrompts:
    self_refine_feedback_header: str = (
        "You are a strict scientific editor reviewing a draft explanation derived from embedding evidence.\n"
        "Goal: improve mechanistic depth, correctness, and compliance with minimal unnecessary drift."
    )
    self_refine_feedback_rules: str = (
        "Hard rules:\n"
        "- Source discipline: any numeric/factual claims about embeddings/contexts MUST come from the ORIGINAL USER PROMPT.\n"
        "- You MAY use general domain knowledge to connect gene roles/pathways *at a high level* (the prompt explicitly asks for known biology), BUT:\n"
        "  - Do NOT introduce new numeric values, mutation IDs, drug names, paper-specific claims, or dataset-specific facts unless they appear in the prompt.\n"
        "  - If GO terms are missing/N/A, you may still use canonical, widely-known gene roles (e.g., metabolic enzyme vs DNA-repair factor), but avoid overly specific claims; hedge as hypotheses.\n"
        "  - Any non-prompt biology must be explicitly hedged (may/could/plausibly) and tied back to the embedding pattern or GO terms when available.\n"
        "- Enforce the required output format: it must start with: Embedding Analysis and Context-Specific Mechanism:\n"
        "- Enforce constraints: do NOT mention model architecture/training; do NOT state the SL score.\n"
        "- Enforce style: single paragraph; 7-10 sentences; avoid redundancy; expand the cancer name once then use abbreviation if a standard abbreviation exists (e.g., lung adenocarcinoma (LUAD)).\n"
        "\nMechanistic-depth rubric (must still fit in 7-10 sentences):\n"
        "- Evidence: at least 1 sentence that cites the key embedding dimensions/values and a cross-context contrast.\n"
        "- Roles: state each gene's main role (from GO terms if present; otherwise a safe high-level role) in the target cancer context.\n"
        "- Interaction: explicitly describe a dependency/compensation chain linking the two roles (Gene A perturbation/defect → increased reliance on Gene B pathway).\n"
        "- Consequence: include a concrete downstream consequence (e.g., DNA damage, oxidative stress, replication stress) as a hypothesis if not directly evidenced.\n"
        "- Testability: include 1 concrete, measurable wet-lab readout/assay that would support/refute the proposed chain (e.g., γH2AX, comet assay, ROS readouts, rescue experiments).\n"
        "- Uncertainty: clearly mark what is inferred vs hypothesized and what cannot be concluded from embeddings alone.\n"
        "\nOutput:\n"
        "- Write 6-10 bullets.\n"
        "- Each bullet MUST be: ISSUE: ... | FIX: ... | WHY: ..."
    )
    self_refine_rewrite_header: str = (
        "Rewrite the MODEL RESPONSE to address the FEEDBACK.\n"
        "Rules:\n"
        "- MINIMAL-CHANGE POLICY: keep as much original wording as possible, but add/clarify missing mechanistic links if the draft is shallow.\n"
        "- Numeric/factual embedding statements must be supported by the ORIGINAL USER PROMPT.\n"
        "- You may add high-level, well-established biological connections consistent with the prompt (and GO terms if present); hedge them as hypotheses.\n"
        "- Keep it 7-10 sentences in ONE paragraph; avoid repeating the same idea with different wording.\n"
        "- Include 1 concise, testable wet-lab validation hook (one assay/readout/prediction) consistent with the proposed mechanism.\n"
        "- Ensure: cancer full name once then abbreviation if standard; no SL score; no architecture/training mentions.\n"
        "- Output must start with: Embedding Analysis and Context-Specific Mechanism:"
    )

    cove_questions_header: str = (
        "You are performing Chain-of-Verification (CoVe) for mechanistic scientific explanations.\n"
        "Given the ORIGINAL USER PROMPT and the DRAFT RESPONSE, produce verification questions that improve mechanistic depth and wet-lab relevance (testable predictions), while still catching obvious embedding-number hallucinations if present.\n"
        "Do NOT ask questions that can be satisfied by generic GO-term paraphrases alone; every question must require the responder to connect at least two evidence sources (e.g., embedding value + mechanistic inference, or mechanism + wet-lab consequence)."
    )
    cove_questions_rules: str = (
        "Rules:\n"
        "- Questions must be answerable from the ORIGINAL USER PROMPT and/or the DRAFT RESPONSE (do NOT use outside knowledge).\n"
        "- Focus on mechanistic depth and wet-lab relevance; avoid superficial compliance checks unless they affect scientific validity.\n"
        "- Prefer yes/no questions that pinpoint what is missing or underspecified, but make each question multi-part enough that a generic GO-term restatement is not sufficient.\n"
        "- Each question should force the answerer to cite an exact clause/value from the draft and explain why that clause/value supports the mechanistic claim.\n"
        "- If n_questions >= 3, ensure coverage includes:\n"
        "  - at least 1 question on evidence integration: does the draft tie a specific embedding value/dimension to a biological inference, rather than just restating a GO term?\n"
        "  - at least 1 question on explicit causal chain: does the draft name the intermediate stress/lesion state and explain why that creates PRKDC dependence?\n"
        "  - at least 1 question on wet-lab testability: does the draft name a concrete assay/readout, the expected direction of change, and what result would support the mechanism?\n"
        "- Numeric faithfulness: only ask a numeric/dimension question if the draft makes a specific numeric or fold-change claim; otherwise do not spend a question on it.\n"
        "- Output ONLY the numbered questions (no extra text)."
    )
    cove_answers_header: str = "Answer the verification questions using ONLY the ORIGINAL USER PROMPT and the DRAFT RESPONSE."
    cove_answers_rules: str = (
        "Rules:\n"
        "- For each question, provide exactly one line starting with: A: \n"
        "- Use one of: YES / NO / NOT VERIFIABLE.\n"
        "- When answering YES/NO, include: (1) a short quoted evidence snippet, (2) a one-sentence explanation of why that snippet is sufficient or insufficient, and (3) the specific missing step if the answer is NO.\n"
        "- Do NOT answer YES on the basis of a generic GO-term paraphrase alone; the evidence must show an explicit link between embedding pattern, causal intermediate, and/or wet-lab prediction.\n"
        "- If it cannot be verified from the provided texts, say: A: NOT VERIFIABLE.\n"
        "- Do NOT add external facts."
    )
    cove_revise_header: str = "Revise the DRAFT RESPONSE using the verification answers."
    cove_revise_rules: str = (
        "Rules:\n"
        "- MINIMAL-CHANGE POLICY: keep as much of the draft wording as possible, but fix any missing required elements.\n"
        "- Remove or weaken any NOT VERIFIABLE statements.\n"
        "- Numeric/factual embedding statements must be supported by the ORIGINAL USER PROMPT.\n"
        "- If the draft is shallow (e.g., lacks explicit gene roles or an interaction/dependency chain), strengthen it using GO terms from the prompt and safe high-level biology; hedge as hypothesis.\n"
        "- If the draft lacks wet-lab relevance, add ONE concise testable prediction/readout (assay) that follows from the proposed chain; keep it high-level and do not introduce new numeric facts.\n"
        "- Keep it 7-10 sentences in ONE paragraph; avoid redundancy.\n"
        "- Ensure: cancer full name once then abbreviation if standard; no SL score; no architecture/training mentions.\n"
        "- Output must start with: Embedding Analysis and Context-Specific Mechanism:"
    )


DEFAULT_PROMPTS = StrategyPrompts()


def _join(*parts: str) -> str:
    return "\n\n".join(p for p in parts if p)


def self_refine_feedback_prompt(*, original_prompt: str, model_response: str, prompts: StrategyPrompts = DEFAULT_PROMPTS) -> str:
    return _join(
        prompts.self_refine_feedback_header,
        prompts.self_refine_feedback_rules,
        "ORIGINAL USER PROMPT:\n" + str(original_prompt or ""),
        "MODEL RESPONSE:\n" + str(model_response or ""),
    )


def self_refine_rewrite_prompt(
    *,
    original_prompt: str,
    model_response: str,
    feedback: str,
    prompts: StrategyPrompts = DEFAULT_PROMPTS,
) -> str:
    return _join(
        prompts.self_refine_rewrite_header,
        "ORIGINAL USER PROMPT:\n" + str(original_prompt or ""),
        "MODEL RESPONSE (to rewrite):\n" + str(model_response or ""),
        "FEEDBACK:\n" + str(feedback or ""),
    )


def cove_questions_prompt(
    *,
    original_prompt: str,
    draft_response: str,
    n_questions: int,
    prompts: StrategyPrompts = DEFAULT_PROMPTS,
) -> str:
    return _join(
        prompts.cove_questions_header,
        prompts.cove_questions_rules,
        f"Write exactly {int(n_questions)} questions.",
        "ORIGINAL USER PROMPT:\n" + str(original_prompt or ""),
        "DRAFT RESPONSE:\n" + str(draft_response or ""),
    )


def cove_answers_prompt(
    *,
    original_prompt: str,
    draft_response: str,
    questions: str,
    prompts: StrategyPrompts = DEFAULT_PROMPTS,
) -> str:
    return _join(
        prompts.cove_answers_header,
        prompts.cove_answers_rules,
        "ORIGINAL USER PROMPT:\n" + str(original_prompt or ""),
        "DRAFT RESPONSE:\n" + str(draft_response or ""),
        "VERIFICATION QUESTIONS:\n" + str(questions or ""),
    )


def cove_revise_prompt(
    *,
    original_prompt: str,
    draft_response: str,
    verification_answers: str,
    prompts: StrategyPrompts = DEFAULT_PROMPTS,
) -> str:
    return _join(
        prompts.cove_revise_header,
        prompts.cove_revise_rules,
        "ORIGINAL USER PROMPT:\n" + str(original_prompt or ""),
        "DRAFT RESPONSE:\n" + str(draft_response or ""),
        "VERIFICATION ANSWERS:\n" + str(verification_answers or ""),
    )
