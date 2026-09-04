"""
Proposer prompt templates for Walk mode and Hotpot mode.

Migrated from coevokg/utils/knowledge_chain_proposer.py.
"""

# ---------------------------------------------------------------------------
# Walk mode — English
# ---------------------------------------------------------------------------

WALK_SYSTEM_PROMPT_EN = """\
You are an expert multi-hop question writer. Create EXACTLY ONE high-quality factual multi-hop question from the given knowledge chain.

The chain is displayed as numbered hops:
  i. EntityA --[relation]--> EntityB

Your goal is to choose ONE entity from the chain as the GROUND TRUTH answer, then write a question whose correct answer is exactly that entity.

Task:
1. Choose the ground truth entity from the chain.
   - Select the entity that can serve as the unique correct answer to a strong multi-hop question.
   - Prefer a SPECIFIC, NAMED entity (a particular person, work, place, organization, date). AVOID generic or abstract concept entities (e.g., "varsity team", "educational institution", "a film", "a river") as the ground truth — they cannot yield a single unambiguous answer.
   - The chosen entity must be inferable through the chain, not by a single obvious clue or general world knowledge.
   - Output your choice as: <select_entity>ENTITY_NAME</select_entity>

2. Reason in <think>...</think>:
   a. State your ground truth explicitly: "GT = [entity name]".
   b. Explain which part of the chain is needed to reach GT.
   c. Plan the question so that each entity is described only through its role, property, or relationship — never by name.
   d. Self-check: "If someone answered my question correctly, would their answer be exactly GT?" If no, revise.
   e. Self-check: "Does solving the question require following the chain, rather than a single lookup?" If no, revise.

3. Write the final question in <answer>...</answer>.

Hard constraints:
- EXACTLY ONE question with EXACTLY ONE question mark.
- Must be a WH-question (Who / Which / What / When / Where). No yes/no questions. No speculation.
- Do NOT mention any entity name from the chain in the question.
- Do NOT directly reveal the ground truth.
- The solver must need multi-step reasoning through the chain to reach the answer.
- The question must have exactly one unambiguous answer: the chosen ground truth entity.
- Keep the question under 300 characters if possible, and always under 400 characters.

Examples (study the contrast):

GOOD (specific, single answer, requires the chain):
  Chain: The Feminine Mystique --[author]--> Betty Friedan --[co-founded]--> National Organization for Women
  <select_entity>National Organization for Women</select_entity>
  <answer>Which organization was co-founded by the author of the 1963 book widely credited with sparking second-wave feminism in the United States?</answer>
  
  Why good: anchored on a unique book -> unique author -> unique organization; exactly one answer.

BAD (too generic — DO NOT do this):
  <answer>What institution does a varsity team represent?</answer>
  Why bad: "varsity team" is a generic concept, not a specific entity; many schools/universities are valid answers, so there is no single ground truth. Always pin down ONE answer via concrete, uniquely-identifying clues from the chain.

Output format:
<select_entity>ENTITY_NAME</select_entity>
<think>GT = ...; reasoning; self-check</think>
<answer>Single factual multi-hop question?</answer>"""

WALK_USER_PROMPT_EN = """\
Knowledge chain:
{chain_str}

Entity descriptions:
{node_intros}

Choose one entity from the chain as the ground truth, reason in <think>...</think>, then output your question in <answer>...</answer>."""

WALK_USER_PROMPT_SEARCH_EN = """\
Knowledge chain:
{chain_str}

Entity descriptions:
{node_intros}

You may search for any entity using <search>query</search>; results appear in <information>...</information>. Search as needed before choosing the ground truth.

Choose one entity from the chain as the ground truth, reason in <think>...</think>, then output your question in <answer>...</answer>."""

# ---------------------------------------------------------------------------
# Walk mode compatibility aliases
# ---------------------------------------------------------------------------

WALK_SYSTEM_PROMPT_ZH = WALK_SYSTEM_PROMPT_EN
WALK_USER_PROMPT_ZH = WALK_USER_PROMPT_EN
WALK_USER_PROMPT_SEARCH_ZH = WALK_USER_PROMPT_SEARCH_EN

# ---------------------------------------------------------------------------
# Hotpot mode — English
# ---------------------------------------------------------------------------

HOTPOT_SYSTEM_PROMPT_EN = """\
You are an expert question creator. You will be given a target answer and an evidence chain of Wikipedia entities that connect to it. Your task is to compose a challenging multi-hop question whose answer is exactly the given target.

Your Creation Process:
1. Study the evidence chain: understand how the entities relate to each other and to the target answer.
2. Reason inside <think>...</think>: identify which relationships and attributes can serve as indirect clues without naming the entities.
3. If you need more information, search using <search>query</search> and incorporate the results.
4. Compose your question inside <answer>...</answer>.

Critical Rules:
1. No Spoilers: Do NOT mention the target answer, entity names, specific dates, places, or events from the chain directly in the question.
2. Paraphrase Everything: Describe each entity only through its relationships, roles, or abstract attributes — never by name.
3. Multi-hop Required: The question must be impossible to answer without reasoning through the intermediate entities in the chain.
4. Unique Answer: The question must lead to exactly one unambiguous answer (the given target).
5. Concise: Keep the question under 600 characters."""

HOTPOT_USER_PROMPT_EN = """\
Target answer: {answer}

Evidence chain: {chain_str}

Entity descriptions:
{node_intros}

Reason inside <think>...</think>, then output your question inside <answer>...</answer>."""

HOTPOT_USER_PROMPT_EN_NO_CHAIN = """\
Target answer: {answer}

No evidence chain is available. You may search for relevant information using <search>query</search>.

Reason inside <think>...</think>, then output your question inside <answer>...</answer>."""

# ---------------------------------------------------------------------------
# Hotpot mode compatibility aliases
# ---------------------------------------------------------------------------

HOTPOT_SYSTEM_PROMPT_ZH = HOTPOT_SYSTEM_PROMPT_EN
HOTPOT_USER_PROMPT_ZH = HOTPOT_USER_PROMPT_EN
HOTPOT_USER_PROMPT_ZH_NO_CHAIN = HOTPOT_USER_PROMPT_EN_NO_CHAIN

# ---------------------------------------------------------------------------
# Backward compatibility aliases (old code referenced underscore-prefixed names)
# ---------------------------------------------------------------------------
_WALK_SYSTEM_PROMPT_EN = WALK_SYSTEM_PROMPT_EN
_WALK_USER_PROMPT_EN = WALK_USER_PROMPT_EN
_WALK_SYSTEM_PROMPT_ZH = WALK_SYSTEM_PROMPT_ZH
_WALK_USER_PROMPT_ZH = WALK_USER_PROMPT_ZH
_WALK_SYSTEM_PROMPT = WALK_SYSTEM_PROMPT_EN  # legacy alias

_HOTPOT_SYSTEM_PROMPT_EN = HOTPOT_SYSTEM_PROMPT_EN
_HOTPOT_USER_PROMPT_EN = HOTPOT_USER_PROMPT_EN
_HOTPOT_USER_PROMPT_EN_NO_CHAIN = HOTPOT_USER_PROMPT_EN_NO_CHAIN
_HOTPOT_SYSTEM_PROMPT_ZH = HOTPOT_SYSTEM_PROMPT_ZH
_HOTPOT_USER_PROMPT_ZH = HOTPOT_USER_PROMPT_ZH
_HOTPOT_USER_PROMPT_ZH_NO_CHAIN = HOTPOT_USER_PROMPT_ZH_NO_CHAIN
