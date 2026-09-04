"""
Question quality evaluation prompts for the Proposer walk-mode pipeline.

Two evaluation modes:
  1. RAG-verify (primary): Use chain documents as context; check if the
     question is answerable and the answer matches ground truth.
  2. LLM-score (fallback / supplementary): Dimension-based scoring.

RAG verify formula (Python-computed):
    quality = 1.0 if model answers correctly from docs
            = 0.3 if model answers but incorrectly (chain exists, question misaligned)
            = 0.0 if model cannot determine (chain doesn't support the question)

LLM-score formula (Python-computed):
    no_leak  = 1 if model no_leak >= 0.5 else 0
    overall  = no_leak × (0.30×chain_faithful + 0.30×multi_hop
                         + 0.20×single_focus  + 0.20×clarity)
"""

# ---------------------------------------------------------------------------
# Mode 1: RAG-verify — answer the question using only chain documents
# ---------------------------------------------------------------------------

RAG_VERIFY_SYSTEM = """\
You are given a set of reference documents and a question. Your task is to answer the question using ONLY the information in the provided documents.

Rules:
- Answer with the entity name directly if you can determine it from the documents.
- If the answer cannot be determined from the documents alone, output exactly: Cannot determine
- Do NOT use external knowledge beyond the provided documents.
- Output only the answer (entity name or "Cannot determine"), nothing else."""

RAG_VERIFY_USER = """\
Reference documents:
{context}

Question: {question}

Answer:"""

# ---------------------------------------------------------------------------
# Mode 2: LLM dimension scoring (kept for supplementary use)
# ---------------------------------------------------------------------------

QUALITY_CHECK_SYSTEM_PROMPT = """\
You are an expert evaluator of multi-hop search questions.

You will receive a shared knowledge chain with hop descriptions, the correct answer, and one or more candidate questions generated from that chain.

Score EACH candidate question on four dimensions:

1. no_leak  [0 or 1 — binary hard gate]
   Score 0 if ANY of:
     • The answer entity name, a clear synonym, or a well-known abbreviation appears verbatim.
     • A phrase uniquely identifying only the answer entity is used.
     • The question is trivially answerable without the chain because it directly names or strongly implies the answer.
   Score 1 when a solver must reason through the chain to reach the answer.

2. chain_faithful  [0.0–1.0, weight 0.30]
   Does the question substantially depend on the provided chain?
   1.0 — following the chain naturally reaches the answer; skipping hops leaves it unanswerable.
   0.6 — chain is mostly useful but one hop is weak or skippable.
   0.3 — only the last 1–2 hops are needed; the earlier chain is irrelevant.
   0.0 — chain is irrelevant; the answer is reachable via a shortcut bypassing the chain.
   Note: intermediate entities need not be explicitly named — implicit traversal is fine.

3. multi_hop  [0.0–1.0, weight 0.30]
   Does the question require genuinely chained, non-compressible reasoning?
   1.0 — at least 2 genuinely dependent inferential steps, no shortcut collapses any step.
          A well-crafted 2-hop question scores 1.0 if both hops are non-trivial.
   0.7 — 2+ hops but one step is weak or bridged by common knowledge.
   0.3 — essentially single-hop; only one genuine lookup is needed.
   0.0 — direct lookup or common knowledge; no chaining.
   Important: longer chains do NOT automatically score higher.

4. single_focus  [0.0–1.0, weight 0.20]
   Does the question target exactly one specific entity, with enough constraints to rule out other plausible candidates?
   1.0 — one unambiguous target; constraints collectively eliminate all other candidates.
   0.5 — mildly ambiguous; one or two other candidates remain plausible.
   0.0 — yes/no, list-type, numerical range, or too vague.

5. clarity  [0.0–1.0, weight 0.20]
   Is the question grammatically correct, coherent, and free of contradictions?
   1.0 — fluent, parseable, internally consistent.
   0.5 — minor awkwardness but intelligible.
   0.0 — garbled, self-contradictory, or nonsensical.

Return ONLY a JSON array with exactly one object per candidate, in the same order.
Each object has exactly five keys (no "reason", no "overall"):

[
  {"no_leak": <0 or 1>, "chain_faithful": <float>, "multi_hop": <float>, "single_focus": <float>, "clarity": <float>},
  ...
]"""


QUALITY_CHECK_USER_PROMPT = """\
Knowledge chain with hop descriptions:
{chain_with_intros}

Correct answer (last node): {ground_truth}

Evaluate the following {n_candidates} candidate question(s) and return a JSON array of {n_candidates} score object(s):

{numbered_questions}"""


# ---------------------------------------------------------------------------
# Rule generation prompt (kept for API compatibility; unused in training)
# ---------------------------------------------------------------------------

RULE_GENERATION_SYSTEM_PROMPT = """\
You are an expert in question design. Based on a high-quality multi-hop question example, extract a concise, reusable pattern or rule that describes what makes this question effective.

The rule should be:
- Generalizable (applicable to other questions, not just this one)
- Actionable (tells a question creator what to do or avoid)
- Concise (1-2 sentences maximum)

Return ONLY the rule text. No preamble, no explanation."""

RULE_GENERATION_USER_PROMPT = """\
High-quality question: {question}
Reference answer: {ground_truth}
Solver success rate: {success_rate:.2f}
Quality score: {quality_score:.2f}

Extract a reusable design rule from this question."""


# ---------------------------------------------------------------------------
# Batch quality-check prompt (all candidates of one chain → one API call)
# ---------------------------------------------------------------------------

QUALITY_CHECK_SYSTEM_PROMPT = """\
You are an expert evaluator of multi-hop search questions.

You will receive a shared knowledge chain with hop descriptions, the correct answer, and one or more candidate questions generated from that chain.

Score EACH candidate question on four dimensions:

1. no_leak  [0 or 1 — binary hard gate]
   Score 0 if ANY of:
     • The answer entity name, a clear synonym, or a well-known abbreviation appears verbatim.
     • A phrase uniquely identifying only the answer entity is used.
     • The question is trivially answerable without the chain because it directly names or strongly implies the answer.
   Score 1 when a solver must reason through the chain to reach the answer.

2. chain_faithful  [0.0–1.0, weight 0.30]
   Does the question substantially depend on the provided chain?
   The question should make a solver need to pass through the intermediate hops rather than reaching the answer via an unrelated shortcut.
   1.0 — following the chain naturally reaches the answer; skipping hops leaves it unanswerable.
   0.6 — chain is mostly useful but one hop is weak or skippable.
   0.3 — only the last 1–2 hops are needed; the earlier chain is irrelevant.
   0.0 — chain is irrelevant; the answer is reachable via a shortcut bypassing the chain.
   Note: intermediate entities need not be explicitly named — implicit traversal is fine.

3. multi_hop  [0.0–1.0, weight 0.30]
   Does the question require genuinely chained, non-compressible reasoning?
   Judge by: can any step be skipped or inferred from common knowledge alone?
   1.0 — at least 2 genuinely dependent inferential steps, no shortcut collapses any step.
          A well-crafted 2-hop question scores 1.0 if both hops are non-trivial.
   0.7 — 2+ hops, but one step is weak or bridged by common knowledge.
   0.3 — essentially single-hop; only one genuine lookup is needed.
   0.0 — direct lookup or common knowledge; no chaining.
   Important: longer chains do NOT automatically score higher. Score on genuine non-compressibility.

4. single_focus  [0.0–1.0, weight 0.20]
   Does the question target exactly one specific entity, with enough constraints to rule out other plausible candidates?
   1.0 — one unambiguous target; constraints collectively eliminate all other candidates.
   0.5 — mildly ambiguous; one or two other candidates remain plausible.
   0.0 — yes/no, list-type, numerical range, multi-answer, or too vague to uniquely identify one entity.

5. clarity  [0.0–1.0, weight 0.20]
   Is the question grammatically correct, coherent, and free of contradictions?
   1.0 — fluent, parseable, internally consistent.
   0.5 — minor awkwardness but intelligible.
   0.0 — garbled, self-contradictory, or nonsensical.

────────────────────────────────────────────────────────────────────────────
OUTPUT FORMAT
────────────────────────────────────────────────────────────────────────────
Return ONLY a JSON array with exactly one object per candidate, in the same order.
Each object has exactly five keys (no "reason", no "overall"):

[
  {"no_leak": <0 or 1>, "chain_faithful": <float>, "multi_hop": <float>, "single_focus": <float>, "clarity": <float>},
  ...
]"""


QUALITY_CHECK_USER_PROMPT = """\
Knowledge chain with hop descriptions:
{chain_with_intros}

Correct answer (last node): {ground_truth}

Evaluate the following {n_candidates} candidate question(s) and return a JSON array of {n_candidates} score object(s):

{numbered_questions}"""


# ---------------------------------------------------------------------------
# Rule generation prompt (kept for API compatibility; unused in training)
# ---------------------------------------------------------------------------

RULE_GENERATION_SYSTEM_PROMPT = """\
You are an expert in question design. Based on a high-quality multi-hop question example, extract a concise, reusable pattern or rule that describes what makes this question effective.

The rule should be:
- Generalizable (applicable to other questions, not just this one)
- Actionable (tells a question creator what to do or avoid)
- Concise (1-2 sentences maximum)

Return ONLY the rule text. No preamble, no explanation."""

RULE_GENERATION_USER_PROMPT = """\
High-quality question: {question}
Reference answer: {ground_truth}
Solver success rate: {success_rate:.2f}
Quality score: {quality_score:.2f}

Extract a reusable design rule from this question."""
