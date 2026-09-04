"""
Knowledge-Chain Proposer Utilities for Walk Mode.

Supports offline file mode: load pre-generated chains from a JSONL file.

Chain JSONL format (one record per line):
  {
    "id": "<uuid>",
    "relations": ["Entity1", "Entity2", ..., "EntityN"],
    "nodes": {"Entity1": "<text>", "Entity2": "<text>", ...}
  }
"""

import json
import logging
import re
import random
import uuid
from typing import Any, Dict, List, Optional, Tuple

from coevokg.prompts.proposer_prompts import (
    _WALK_SYSTEM_PROMPT_EN,
    _WALK_USER_PROMPT_EN,
    _WALK_SYSTEM_PROMPT_ZH,
    _WALK_USER_PROMPT_ZH,
    _WALK_SYSTEM_PROMPT,
    WALK_USER_PROMPT_SEARCH_EN,
    WALK_USER_PROMPT_SEARCH_ZH,
    _HOTPOT_SYSTEM_PROMPT_EN,
    _HOTPOT_USER_PROMPT_EN,
    _HOTPOT_USER_PROMPT_EN_NO_CHAIN,
    _HOTPOT_SYSTEM_PROMPT_ZH,
    _HOTPOT_USER_PROMPT_ZH,
    _HOTPOT_USER_PROMPT_ZH_NO_CHAIN,
)
from coevokg.prompts.solver_prompts import SOLVER_USER_PROMPT as _SOLVER_USER_PROMPT_EN

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Relation sentence extraction (chain display with edge context)
# ---------------------------------------------------------------------------

def _extract_relation_sentence(
    text_a: str,
    entity_b: str,
    text_b: str = "",
    max_chars: int = 80,
) -> str:
    """Extract the shortest sentence from text_a (or text_b) that mentions
    entity_b, used as the edge label between adjacent chain entities.
    Falls back to "related to" (<0.2% of real chains).
    """
    target_lower = entity_b.lower()
    b_words = [w for w in re.sub(r'[^\w\s]', ' ', target_lower).split() if len(w) > 3][:2]
    best = None
    for text in (text_a, text_b):
        if not text:
            continue
        for sent in re.split(r'(?<=[.!?])\s+', text.replace('\n', ' ')):
            sl = sent.lower()
            if target_lower in sl or (b_words and all(w in sl for w in b_words)):
                s = re.sub(r'\s+', ' ', sent).strip()
                if best is None or len(s) < len(best):
                    best = s
    if best:
        return best[:max_chars] + ('…' if len(best) > max_chars else '')
    return "related to"


def _build_chain_str_with_relations(
    relations: List[str],
    nodes: Dict[str, str],
    lang: str = "en",
    max_rel_chars: int = 80,
    relation_labels: Optional[List[str]] = None,
) -> str:
    """Build a numbered chain string with relation-context edge labels.

    Priority:
      1. relation_labels[i]  — pre-generated short label (e.g. "is capital of")
      2. sentence extraction — find sentence in node text mentioning next entity
      3. "related to"        — last-resort fallback

    Example with pre-generated labels:
        1. Centre Region (Cameroon) --[is capital of]--> Yaoundé
        2. Yaoundé --[headquartered in]--> Interpol

    Example with sentence extraction:
        1. Centre Region --[Yaoundé, capital of Cameroon, is at the heart…]--> Yaoundé
    """
    if not nodes and not relation_labels:
        return " → ".join(relations)
    lines = []
    for i in range(len(relations) - 1):
        a, b = relations[i], relations[i + 1]
        # Use pre-generated label if available and non-empty
        if relation_labels and i < len(relation_labels) and relation_labels[i]:
            rel = relation_labels[i].strip()
        else:
            rel = _extract_relation_sentence(
                nodes.get(a, ""), b, nodes.get(b, ""), max_rel_chars
            )
        lines.append(f"{i + 1}. {a} --[{rel}]--> {b}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Post-hoc entity selection parsing
# ---------------------------------------------------------------------------

def extract_selected_entity(output: str, relations: List[str]) -> Tuple[int, str]:
    """
    Parse <select_entity>ENTITY_NAME</select_entity> from model output.

    The proposer names the ground truth entity directly; matched case-insensitively
    against the relations list.

    Falls back to:
      - <select_entity>i</select_entity> (legacy: 0-indexed integer position)
      - <select_hops>N</select_hops>      (legacy: 1-indexed count → index N-1)
      - last entity in chain if nothing matches.

    Returns (entity_index, ground_truth_entity).
    """
    total = len(relations)
    relations_lower = [r.lower() for r in relations]

    m = re.search(r"<select_entity>\s*(.*?)\s*</select_entity>", output, re.DOTALL)
    if m:
        content = m.group(1).strip()

        # Primary (new): name-based exact match, case-insensitive
        content_lower = content.lower()
        for i, entity_lower in enumerate(relations_lower):
            if entity_lower == content_lower:
                return i, relations[i]

        # Backward compat: 0-indexed integer position
        if content.isdigit():
            i = int(content)
            if 1 <= i <= total - 1:
                return i, relations[i]

    # Legacy fallback: <select_hops>N</select_hops>  (1-indexed count)
    m = re.search(r"<select_hops>\s*(\d+)\s*</select_hops>", output)
    if m:
        n = int(m.group(1))
        if 2 <= n <= total:
            return n - 1, relations[n - 1]

    # Default: last entity
    return total - 1, relations[-1]


def extract_selected_hops(output: str, relations: List[str]) -> Tuple[int, str]:
    """Legacy wrapper — returns (hop_count, ground_truth) for backward compat."""
    idx, gt = extract_selected_entity(output, relations)
    return idx + 1, gt


# ---------------------------------------------------------------------------
# KnowledgeChainPool
# ---------------------------------------------------------------------------

class KnowledgeChainPool:
    """
    Samples knowledge chains for the walk-mode proposer.

    Loads chains from an offline JSONL file.
    """

    def __init__(
        self,
        chain_data_path: Optional[str] = None,
        min_hops: int = 3,
        lang: str = "en",
        mode: str = "file",
        use_search_proposer: bool = False,
    ):
        self.lang = lang
        self.min_hops = min_hops
        self.mode = mode
        self.use_search_proposer = use_search_proposer
        self.chains: List[Dict[str, Any]] = []
        # entity-sequence -> index in self.chains, for O(1) upsert in add_chain()
        self._chain_index: Dict[str, int] = {}

        if mode != "file":
            raise ValueError("KnowledgeChainPool only supports mode='file' in this release package.")
        if not chain_data_path:
            raise ValueError("KnowledgeChainPool requires chain_data_path.")
        self._load(chain_data_path, min_hops)

    # Terminal entities that almost always leak into the generated question,
    # causing answer_leaked extraction failures (empirically identified).
    _BAD_TERMINAL_PREFIXES = (
        "list of ",
        "lists of ",
        "academy award",
        "grammy award",
        "bafta award",
        "emmy award",
        "golden globe award",
        "tony award",
        "timeline of ",
        "history of ",
        "outline of ",
    )
    _BAD_TERMINAL_SUFFIXES = (
        " discography",
        " filmography",
        " bibliography",
        " (disambiguation)",
        " honours",
        " honor",
        " recording sessions",
    )
    _BAD_TERMINAL_SUBSTRINGS = (
        "television programs",
        "television shows",
        "list of statutory instruments",
        "list of isomers",
        "list of firearms",
        "list of military",
        "list of feature film",
    )
    # Nationality/language adjectives: too generic AND the word itself
    # will inevitably surface in any question about the topic.
    _BAD_TERMINAL_EXACT = frozenset({
        "american", "british", "english", "french", "german", "russian",
        "chinese", "japanese", "italian", "spanish", "australian", "canadian",
        "indian", "korean", "dutch", "portuguese", "swedish", "norwegian",
        "danish", "polish", "turkish", "greek", "hungarian", "czech",
        "latin", "irish", "scottish", "welsh", "swiss",
        # Band/artist names that are common English words — impossible to ask
        # about without naming them (empirically 100% leakage rate).
        "yes (band)",
        "yes",
        # Pure year strings (e.g. "2017", "1996") are unaskable without
        # mentioning the year explicitly.
    })

    @staticmethod
    def _is_bad_terminal(entity: str) -> bool:
        """Return True if the terminal entity almost always leaks into the question."""
        e = entity.lower().strip()

        # Exact-match blocklist (nationality adjectives, common-word entities)
        if e in KnowledgeChainPool._BAD_TERMINAL_EXACT:
            return True

        # Pure 4-digit year (e.g. "2017", "1996") — impossible not to leak
        if len(e) == 4 and e.isdigit() and 1800 <= int(e) <= 2100:
            return True

        # Very short entity names (≤3 chars) are almost always ambiguous or
        # impossible to describe indirectly (e.g. "UK", "EU", "Yes").
        if len(e) <= 3 and not e.isdigit():
            return True

        # Entities starting with a 4-digit year (e.g. "1918 New Year Honours (MM)")
        # always require mentioning the year, which leaks the answer.
        if len(e) > 4 and e[:4].isdigit() and 1800 <= int(e[:4]) <= 2100 and e[4] == ' ':
            return True

        if any(e.startswith(p) for p in KnowledgeChainPool._BAD_TERMINAL_PREFIXES):
            return True
        if any(e.endswith(s) for s in KnowledgeChainPool._BAD_TERMINAL_SUFFIXES):
            return True
        if any(sub in e for sub in KnowledgeChainPool._BAD_TERMINAL_SUBSTRINGS):
            return True
        return False

    def _load(self, path: str, min_hops: int) -> None:
        raw_count = 0
        filtered_hops = 0
        filtered_terminal = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                raw_count += 1
                relations = record.get("relations", [])
                if len(relations) < min_hops:
                    filtered_hops += 1
                    continue
                # Filter chains whose terminal entity almost certainly leaks into questions
                terminal = relations[-1] if relations else ""
                if self._is_bad_terminal(terminal):
                    filtered_terminal += 1
                    continue
                self.chains.append(record)

        if not self.chains:
            raise ValueError(
                f"KnowledgeChainPool: no chains with >= {min_hops} hops found "
                f"in '{path}' (total raw records: {raw_count})."
            )
        # Build entity-sequence index for O(1) upsert in add_chain()
        self._chain_index = {
            self._chain_key(c["relations"]): i
            for i, c in enumerate(self.chains)
        }
        logger.info(
            f"KnowledgeChainPool: loaded {len(self.chains)} chains "
            f"(min_hops={min_hops}) from '{path}' "
            f"(filtered from {raw_count} raw: "
            f"{filtered_hops} hops-short, {filtered_terminal} bad-terminal)."
        )

    def sample(
        self,
        n: int,
        seed_entities: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Sample n chains.

        Random sample from the pre-loaded offline chain pool.
        seed_entities is accepted for backward-compatible call sites and ignored.
        """
        if len(self.chains) >= n:
            return random.sample(self.chains, n)
        return [random.choice(self.chains) for _ in range(n)]

    # ------------------------------------------------------------------
    # Dynamic pool update (co-evolution write-back)
    # ------------------------------------------------------------------

    @staticmethod
    def _chain_key(relations: List[str]) -> str:
        return "|".join(r.lower().strip() for r in relations)

    def add_chain(self, chain: Dict[str, Any]) -> bool:
        """Upsert a chain into the in-memory pool (file mode only).

        - Same entity sequence already in pool → merge new node text into
          existing entry (enrich nodes, keep chain count stable).
        - New entity sequence → append after passing min_hops / bad-terminal
          filters.

        Returns True when the pool was modified (new or enriched).
        """
        if self.mode != "file":
            return False
        relations = chain.get("relations", [])
        if not relations:
            return False

        key = self._chain_key(relations)

        if key in self._chain_index:
            # Enrich existing chain's nodes with any new text
            existing = self.chains[self._chain_index[key]]
            existing_nodes = existing.setdefault("nodes", {})
            new_nodes = chain.get("nodes", {})
            enriched = False
            for entity, text in new_nodes.items():
                if not text:
                    continue
                prev = existing_nodes.get(entity, "")
                if text not in prev:
                    existing_nodes[entity] = (prev + "\n" + text).strip()[:2000]
                    enriched = True
            return enriched

        # New entity sequence — apply quality filters before appending
        if len(relations) < self.min_hops:
            return False
        if self._is_bad_terminal(relations[-1]):
            return False
        self._chain_index[key] = len(self.chains)
        self.chains.append(chain)
        return True

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def build_proposer_data(self, chain: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a chain record into a CoEvoKG proposer seed-data record.

        Supports two chain formats:

        1. Hotpot format (has explicit "answer" field):
           {"id", "answer", "relations", "nodes", ...}
           → Uses hotpot-style prompt: answer is known, no select_hops needed.

        2. Walk format (no "answer" field):
           {"id", "relations", "nodes"}
           → Uses walk-style prompt: proposer selects an entity via select_entity.

        Ground truth = chain["answer"] if present, else relations[-1].
        """
        relations: List[str] = chain.get("relations", [])
        nodes: Dict[str, str] = chain.get("nodes", {})

        if not relations:
            return None

        # Determine answer (hotpot format has explicit answer)
        answer: str = chain.get("answer") or relations[-1]

        # Build chain_str with relation labels.
        # Priority: pre-generated labels from chain["relation_labels"] field
        # (produced by clean_chains.py --label-relations) > sentence extraction fallback.
        relation_labels = chain.get("relation_labels", None)
        chain_str = _build_chain_str_with_relations(
            relations, nodes, self.lang, relation_labels=relation_labels
        )

        # Build per-entity intro (truncated to 300 chars for richer context)
        node_intro_lines = []
        for entity in relations:
            text = nodes.get(entity, "")
            snippet = text[:300].strip().replace("\n", " ")
            if self.lang == "zh":
                node_intro_lines.append(f"【{entity}】：{snippet}")
            else:
                node_intro_lines.append(f"[{entity}]: {snippet}")
        node_intros = "\n".join(node_intro_lines)

        is_hotpot = "answer" in chain

        if is_hotpot:
            # Hotpot format: answer-aware prompt, no select_hops
            has_chain = bool(relations and any(nodes.get(e, "").strip() for e in relations))
            if self.lang == "zh":
                system_prompt = _HOTPOT_SYSTEM_PROMPT_ZH
                if has_chain:
                    user_content = _HOTPOT_USER_PROMPT_ZH.format(
                        answer=answer,
                        chain_str=chain_str,
                        node_intros=node_intros,
                    )
                else:
                    user_content = _HOTPOT_USER_PROMPT_ZH_NO_CHAIN.format(answer=answer)
            else:
                system_prompt = _HOTPOT_SYSTEM_PROMPT_EN
                if has_chain:
                    user_content = _HOTPOT_USER_PROMPT_EN.format(
                        answer=answer,
                        chain_str=chain_str,
                        node_intros=node_intros,
                    )
                else:
                    user_content = _HOTPOT_USER_PROMPT_EN_NO_CHAIN.format(answer=answer)
        else:
            # Walk format: select_entity prompt
            if self.use_search_proposer:
                # Search-enabled: pass chain + entity descriptions, model may also search
                if self.lang == "zh":
                    system_prompt = _WALK_SYSTEM_PROMPT_ZH
                    user_content = WALK_USER_PROMPT_SEARCH_ZH.format(chain_str=chain_str)
                else:
                    system_prompt = _WALK_SYSTEM_PROMPT_EN
                    user_content = WALK_USER_PROMPT_SEARCH_EN.format(
                        chain_str=chain_str, node_intros=node_intros
                    )
            elif self.lang == "zh":
                system_prompt = _WALK_SYSTEM_PROMPT_ZH
                user_content = _WALK_USER_PROMPT_ZH.format(
                    chain_str=chain_str,
                    node_intros=node_intros,
                )
            else:
                system_prompt = _WALK_SYSTEM_PROMPT_EN
                user_content = _WALK_USER_PROMPT_EN.format(
                    chain_str=chain_str,
                    node_intros=node_intros,
                )

        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        return {
            "prompt": prompt,
            "data_source": "walk_chain",
            "reward_model": {
                "style": "rule",
                "ground_truth": {"target": answer},
            },
            "extra_info": {
                "type": "qa_search",
                "question": answer,  # seed "question" = answer entity (for fallback tracking)
                "chain_id": chain.get("id", ""),
                "chain_relations": relations,
                # Keep chain evidence available to offline quality checks even
                # when use_search_proposer=True and docs are not in the prompt.
                "chain_nodes": {
                    entity: str(nodes.get(entity, ""))[:1000]
                    for entity in relations
                    if nodes.get(entity, "")
                },
                "need_tools_kwargs": True,
                "tools_kwargs": {
                    "search": {
                        "create_kwargs": {
                            "data_source": "walk_chain",
                            "question": answer,
                            "ground_truth": {"target": answer},
                        }
                    }
                },
            },
        }

    def build_proposer_data_batch(self, chains: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build proposer data records for a list of chains, skipping invalid ones."""
        result = []
        for chain in chains:
            record = self.build_proposer_data(chain)
            if record is not None:
                result.append(record)
        return result


def build_seed_solver_problem(
    seed_record: Dict[str, Any],
    trajectory_index: int,
    solver_system_prompt: str,
    solver_user_prompt_template: str,
) -> Optional[Dict[str, Any]]:
    """
    Convert a raw seed data record directly into a solver problem dict
    (without going through the proposer).  Used for seed-padding in walk mode.
    """
    prompt_msgs = seed_record.get("prompt", [])
    # Extract the question text from the user message
    question = ""
    for msg in prompt_msgs:
        if msg.get("role") == "user":
            question = msg.get("content", "").strip()
            break

    extra_info = seed_record.get("extra_info", {})
    if not question:
        question = extra_info.get("question", "")

    if not question or question.lower() == "dummy":
        return None

    reward_model = seed_record.get("reward_model", {"ground_truth": {"target": "unknown"}, "style": "rule"})
    data_source = seed_record.get("data_source", "seed_fallback")
    ground_truth = reward_model.get("ground_truth")

    formatted_prompt = []
    if solver_system_prompt:
        formatted_prompt.append({"role": "system", "content": solver_system_prompt})
    formatted_prompt.append({
        "role": "user",
        "content": solver_user_prompt_template.format(question),
    })

    return {
        "data_source": data_source,
        "prompt": formatted_prompt,
        "ability": "fact-reasoning",
        "reward_model": reward_model,
        "extra_info": {
            "question": question,
            "need_tools_kwargs": True,
            "split": "train",
            "tools_kwargs": {
                "search": {
                    "create_kwargs": {
                        "data_source": data_source,
                        "question": question,
                        "ground_truth": ground_truth,
                    }
                }
            },
            "seed_fallback": True,
            "walk_seed_pad": True,
        },
        "metadata": None,
        "extracted_question": question,
        "formatted_prompt": formatted_prompt,
        "problem_type": "search",
        "trajectory_index": trajectory_index,
    }
