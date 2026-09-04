"""
LLM-as-a-Judge prompt templates.

Migrated from coevokg/utils/llm_as_a_judge.py.
"""

JUDGE_SYSTEM_PROMPT = (
    "You are a professional judge who evaluates the correctness of answers based on given criteria."
)

JUDGE_EVALUATION_PROMPT = """\
Please determine whether the model's answer is consistent with the reference answer:

Question: {question}
Model Answer: {model_answer}
Reference Answer: {golden_answer}

Evaluation Criteria:
1. The model answer must accurately respond to the question and be consistent with the reference answer in meaning.
2. For numerical questions, the values must be equal or very close.
3. For textual questions, the core meaning must be correct.
4. Differences in wording or language are allowed as long as the core answer is the same.
5. If the model answer includes the correct answer and does not contain conflicting information, it is also considered correct.

Please respond only with "Correct" or "Wrong". Do not provide any additional explanation."""

JUDGE_ANSWER_FROM_MATERIALS_PROMPT = """\
Answer the given question based on the provided materials. You should first conduct very concise reasoning within 50 words, and then directly provide your answer without detailed illustrations after saying 'Answer:'. Materials: {materials}
 Question: {question}
"""

JUDGE_WITH_CHAIN_EXTRACTION_PROMPT = """\
Please evaluate whether the model's FINAL ANSWER is correct, AND separately extract the named entities the model used in its reasoning.

Question: {question}
Model Final Answer: {final_answer}
Reference Answer: {golden_answer}

Model Reasoning (use ONLY for entity extraction below — do NOT use it to judge correctness): {reasoning}

Correctness Criteria — judge ONLY the "Model Final Answer" against the "Reference Answer":
1. The final answer must directly answer the question and match the reference answer in meaning.
2. For numerical or date answers, the values must be equal (minor rounding or formatting differences are allowed, e.g. "50" = "fifty", "1992-93" = "1992–93").
3. For textual answers, the core meaning must be the same.
4. Differences in wording, language, or extra surrounding words are allowed, as long as the final answer itself conveys the reference answer.
5. Judge by the FINAL ANSWER. If the "Model Final Answer" is empty or clearly truncated/incomplete, you may instead use the model's CONCLUSION at the END of its reasoning. In all cases, judge the model's stated conclusion — NOT entities or values merely mentioned or searched during reasoning. If the conclusion differs from or conflicts with the reference answer, it is Wrong.

Entity Extraction:
- Look at the model's reasoning steps (thinking, search queries, intermediate conclusions).
- List the main named entities (people, places, organisations, events, works) in order of first appearance.
- Include only substantive named entities — skip generic words like "the result", "this answer", etc.
- Aim for 2–8 entities; more is fine if the reasoning is long.

Return ONLY a valid JSON object on a single line, no markdown fences:
{{"correct": true, "chain": ["Entity0", "Entity1", "Entity2"]}}

Where:
- "correct" is true if the FINAL ANSWER is correct, false otherwise.
- "chain" is the ordered list of named entities from the reasoning.
- Every chain item must be a valid JSON string. Escape internal double quotes with backslashes.
- If you are unsure about the chain, return an empty list instead of malformed JSON."""


# Write-back chain extraction: correctness is already known at write-back time, so this
# prompt only extracts a clean, canonical-named evidence chain PLUS the relation between
# each consecutive entity, for assembling a complete writable KG chain.
WRITEBACK_CHAIN_EXTRACTION_PROMPT = """\
Extract the evidence chain (entities + the relation between each consecutive pair) that leads to the answer.

Question: {question}
Answer: {final_answer}
Retrieved document titles (retrieval order): {titles}
Model Reasoning: {reasoning}

Rules:
- entities: use each entity's FULL canonical name EXACTLY as it appears in a Retrieved document title, INCLUDING disambiguation in parentheses (e.g. "Example City (fictional)" NOT "Example City").
- order entities along the path, ending with the ANSWER entity as the LAST item; DROP unrelated titles that search happened to return.
- relations: a SHORT typed phrase describing how each entity links to the NEXT one (e.g. "located in", "directed by", "member of", "is a state of"). len(relations) MUST equal len(chain)-1.
- never return an empty chain if any named entity is present.

Example:
Q: Which fictional country contains Example City? | Answer: Example Country | titles: ["Example City (fictional)", "Example Country"]
-> {{"chain": ["Example City (fictional)", "Example Country"], "relations": ["is located in"]}}

Return ONLY one line of valid JSON, no fences:
{{"chain": ["E0","E1","E2"], "relations": ["r01","r12"]}}"""


# Unified single-call prompt: judge correctness AND extract a canonical evidence
# chain + per-edge relations in ONE LLM call. Used by the reward stage so the same
# chain feeds the path-support process reward AND (via passback) the KG write-back,
# eliminating the previous second extraction call.
JUDGE_AND_EXTRACT_PROMPT = """\
Do TWO things in one pass: (1) judge whether the model's FINAL ANSWER is correct, and (2) extract the evidence chain (entities + the relation between each consecutive pair) that leads to the answer.

Question: {question}
Model Final Answer: {final_answer}
Reference Answer: {golden_answer}
Retrieved document titles (retrieval order): {titles}
Model Reasoning: {reasoning}

Correctness Criteria — judge ONLY the "Model Final Answer" against the "Reference Answer":
1. The final answer must directly answer the question and match the reference answer in meaning.
2. For numerical or date answers, values must be equal (minor rounding/formatting allowed, e.g. "50"="fifty", "1992-93"="1992–93").
3. For textual answers, the core meaning must be the same. Wording/language/extra words are allowed as long as the final answer conveys the reference answer.
4. If the "Model Final Answer" is empty or clearly truncated, judge the model's CONCLUSION at the END of its reasoning instead. Judge the stated conclusion — NOT entities merely mentioned or searched. If it conflicts with the reference answer, it is wrong.

Chain Extraction (fill this regardless of correctness):
- entities: use each entity's FULL canonical name EXACTLY as it appears in a Retrieved document title, INCLUDING parenthetical disambiguation (e.g. "Example City (fictional)" NOT "Example City").
- order entities along the path, ending with the ANSWER entity as the LAST item; DROP unrelated titles search happened to return.
- relations: a SHORT typed phrase describing how each entity links to the NEXT one (e.g. "located in", "directed by", "member of"). len(relations) MUST equal len(chain)-1.
- never return an empty chain if any named entity is present.

Example:
Q: Which fictional country contains Example City? | Final Answer: Example Country | Reference: Example Country | titles: ["Example City (fictional)", "Example Country"]
-> {{"correct": true, "chain": ["Example City (fictional)", "Example Country"], "relations": ["is located in"]}}

Return ONLY one line of valid JSON, no fences:
{{"correct": true, "chain": ["E0","E1","E2"], "relations": ["r01","r12"]}}
- "correct" is true iff the FINAL ANSWER is correct.
- Every chain item must be a valid JSON string; escape internal double quotes.
- If unsure about the chain, return an empty list for "chain" rather than malformed JSON."""
