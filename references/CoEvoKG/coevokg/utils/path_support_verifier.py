"""
Path-Support Consistency Verifier.

Pure-Python, deterministic, LLM-free verifier that checks whether a given
thought chain (sequence of named entities) is supported by the static evidence
database (all_chains_filled.jsonl).

Algorithm (following idea.md §3-8):

Given thought chain hat_c = [u0, u1, ..., uL]:

1. Build inverted index: I(e) = set of record indices containing entity e.

2. For each adjacent pair (u, v):
   a. J(u,v) = I(u) ∩ I(v)  — records where both appear
   b. For each record d in J, compute:
      - order_score  (0.0 / 0.4 / 0.7 / 1.0)
      - text_score   (0.0 / 0.5 / 1.0)
      - co_score     (always 1.0 when in J)
      - s_d = 0.5*order + 0.4*text + 0.1*co
   c. M_direct(u,v) = max_d s_d(u,v)
   d. M_bridge(u,v) = eta * max_z min(M_direct(u,z), M_direct(z,v))
                     (z is drawn from entities co-occurring with u or v;
                      limited to max_bridge_candidates for speed)
   e. M(u,v) = max(M_direct, M_bridge)

3. Global chain score = geometric mean of all pair scores with eps smoothing.
4. passed = (min pair score >= tau_local) AND (global >= tau_global)
"""

import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from coevokg.utils.env import get_coevokg_env

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Return dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PairSupportResult:
    left: str
    right: str
    direct_score: float
    bridge_score: float
    final_score: float
    matched_record_ids: List[str] = field(default_factory=list)


@dataclass
class ChainSupportResult:
    chain: List[str]
    pair_results: List[PairSupportResult]
    global_score: float
    min_local_score: float
    passed: bool


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

class PathSupportVerifier:
    """
    Load a JSONL evidence database once, then answer verify_chain() queries.

    Thread-safety: read-only after __init__, safe for concurrent calls.
    """

    def __init__(
        self,
        chain_data_path: str,
        alpha: float = 0.5,
        beta: float = 0.4,
        gamma: float = 0.1,
        eta: float = 0.8,
        tau_local: float = 0.35,
        tau_global: float = 0.60,
        eps: float = 1e-6,
        max_bridge_candidates: int = 50,
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eta = eta
        self.tau_local = tau_local
        self.tau_global = tau_global
        self.eps = eps
        self.max_bridge_candidates = max_bridge_candidates

        # records[i] = {"id": str, "relations": list[str], "nodes": dict[str,str]}
        self.records: List[Dict] = []
        # inverted_index[entity_lower] = set of record indices
        self.inverted_index: Dict[str, Set[int]] = defaultdict(set)

        self._load_and_index(chain_data_path)
        logger.info(
            f"PathSupportVerifier: loaded {len(self.records)} records, "
            f"{len(self.inverted_index)} unique entities from {chain_data_path}"
        )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_and_index(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                relations = raw.get("relations", [])
                nodes = raw.get("nodes", {})
                record = {
                    "id": raw.get("id", ""),
                    "relations": relations,
                    # Precompute lowercased relations to avoid repeated list-comprehension
                    # in _order_score (was the 3rd-largest CPU hotspot).
                    "relations_lower": [r.lower() for r in relations],
                    # Precompute lowercased node texts to avoid repeated .lower() calls
                    # in _text_score for potentially long Wikipedia texts.
                    "nodes": nodes,
                    "nodes_lower": {k: v.lower() for k, v in nodes.items()},
                }
                idx = len(self.records)
                self.records.append(record)
                for entity in relations:
                    self.inverted_index[entity.lower()].add(idx)

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _order_score(self, u_lower: str, v_lower: str, relations: List[str]) -> float:
        """Check adjacency / ordering of u, v in the relations list."""
        # Use precomputed lowercased list from record dict when available.
        # Caller may pass record["relations_lower"] directly to avoid the cast.
        rels_lower = relations  # already lowercased when called from _record_score
        try:
            i_u = rels_lower.index(u_lower)
        except ValueError:
            return 0.0
        try:
            i_v = rels_lower.index(v_lower)
        except ValueError:
            return 0.0

        if i_u >= i_v:
            # u after v — wrong order or same
            return 0.4 if i_u != i_v else 0.0

        # u before v, correct order
        if i_v == i_u + 1:
            return 1.0   # adjacent and correct order
        else:
            return 0.7   # correct order but non-adjacent

    def _text_score(self, u: str, v: str, nodes_lower: Dict[str, str]) -> float:
        """Check if u's text mentions v or v's text mentions u.

        nodes_lower: precomputed lowercased node texts (from record["nodes_lower"]).
        """
        u_lower = u.lower()
        v_lower = v.lower()

        text_u_l = nodes_lower.get(u, "") or ""
        text_v_l = nodes_lower.get(v, "") or ""

        if v_lower in text_u_l or u_lower in text_v_l:
            return 1.0
        # Weak / alias mention: check any token of v in u or vice versa (skip short tokens)
        v_tokens = [t for t in v_lower.split() if len(t) > 3]
        u_tokens = [t for t in u_lower.split() if len(t) > 3]
        if any(t in text_u_l for t in v_tokens) or any(t in text_v_l for t in u_tokens):
            return 0.5
        return 0.0

    def _record_score(self, u: str, v: str, record_idx: int) -> float:
        """Compute s_d(u, v) for a single record."""
        record = self.records[record_idx]
        u_lower = u.lower()
        v_lower = v.lower()
        # Use precomputed lowercased data to avoid repeated str.lower() calls.
        order = self._order_score(u_lower, v_lower, record["relations_lower"])
        text = self._text_score(u, v, record["nodes_lower"])
        co = 1.0  # always 1.0 — both in J means they co-occur
        return self.alpha * order + self.beta * text + self.gamma * co

    def _direct_score_and_ids(self, u: str, v: str) -> Tuple[float, List[str]]:
        """M_direct(u,v) = max_d s_d(u,v) over shared records."""
        I_u = self.inverted_index.get(u.lower(), set())
        I_v = self.inverted_index.get(v.lower(), set())
        J = I_u & I_v
        if not J:
            return 0.0, []
        best = 0.0
        best_ids: List[str] = []
        for idx in J:
            s = self._record_score(u, v, idx)
            if s > best:
                best = s
                best_ids = [self.records[idx]["id"]]
            elif s == best and best > 0:
                best_ids.append(self.records[idx]["id"])
        return best, best_ids

    def _direct_score(self, u: str, v: str) -> float:
        """Lightweight version of _direct_score_and_ids (no id tracking)."""
        I_u = self.inverted_index.get(u.lower(), set())
        I_v = self.inverted_index.get(v.lower(), set())
        J = I_u & I_v
        if not J:
            return 0.0
        return max(self._record_score(u, v, idx) for idx in J)

    def _bridge_score(self, u: str, v: str) -> float:
        """
        M_bridge(u,v) = eta * max_z min(M_direct(u,z), M_direct(z,v))

        Bridge candidates z: all entities co-occurring with u or v, deduped,
        limited to max_bridge_candidates for performance.
        """
        I_u = self.inverted_index.get(u.lower(), set())
        I_v = self.inverted_index.get(v.lower(), set())
        # Collect candidate bridge entities from neighbouring records
        candidate_entities: Set[str] = set()
        for rec_idx in (I_u | I_v):
            for e in self.records[rec_idx]["relations"]:
                candidate_entities.add(e)
        # Remove u and v themselves
        candidate_entities.discard(u)
        candidate_entities.discard(v)
        # Limit for performance
        candidates = list(candidate_entities)[:self.max_bridge_candidates]

        best = 0.0
        for z in candidates:
            s_uz = self._direct_score(u, z)
            if s_uz == 0.0:
                continue
            s_zv = self._direct_score(z, v)
            bridge = min(s_uz, s_zv)
            if bridge > best:
                best = bridge
        return self.eta * best

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_pair_score(self, u: str, v: str) -> PairSupportResult:
        """Compute the full pair support score for adjacent entities u→v."""
        direct, ids = self._direct_score_and_ids(u, v)
        # Skip bridge computation when direct is already strong enough —
        # bridge can at most reach eta*1.0=0.8, so if direct >= 0.8 bridge
        # cannot improve the result. This skips the dominant CPU hotspot (~93%)
        # for well-supported pairs.
        if direct >= self.eta:
            bridge = 0.0
        else:
            bridge = self._bridge_score(u, v)
        final = max(direct, bridge)
        return PairSupportResult(
            left=u,
            right=v,
            direct_score=direct,
            bridge_score=bridge,
            final_score=final,
            matched_record_ids=ids,
        )

    def verify_chain(self, chain: List[str]) -> ChainSupportResult:
        """
        Verify a thought chain of named entities against the evidence database.

        chain: [u0, u1, ..., uL]  (at least 2 elements)

        Returns ChainSupportResult with global_score, min_local_score, passed.
        """
        if len(chain) < 2:
            # Trivial / degenerate chain — give a neutral score
            return ChainSupportResult(
                chain=chain,
                pair_results=[],
                global_score=0.0,
                min_local_score=0.0,
                passed=False,
            )

        pair_results: List[PairSupportResult] = []
        for i in range(1, len(chain)):
            pr = self.compute_pair_score(chain[i - 1], chain[i])
            pair_results.append(pr)

        scores = [pr.final_score for pr in pair_results]
        L = len(scores)
        # Geometric mean with smoothing
        log_sum = sum(math.log(s + self.eps) for s in scores)
        global_score = math.exp(log_sum / L)

        min_local = min(scores)
        passed = (min_local >= self.tau_local) and (global_score >= self.tau_global)

        return ChainSupportResult(
            chain=chain,
            pair_results=pair_results,
            global_score=global_score,
            min_local_score=min_local,
            passed=passed,
        )


# ---------------------------------------------------------------------------
# Global singleton (per-process, thread-safe after init)
# ---------------------------------------------------------------------------

_GLOBAL_VERIFIER: Optional[PathSupportVerifier] = None


def get_global_verifier(chain_data_path: str) -> PathSupportVerifier:
    """
    Lazily initialise and return the global PathSupportVerifier singleton.

    Thread-safe: in CPython, module-level assignment is atomic.
    First call loads the JSONL; subsequent calls return the cached instance.

    Path resolution order:
      1. $COEVOKG_CHAIN_DATA_LOCAL env var  (local /tmp copy, fast even on cold start)
      2. chain_data_path argument         (NAS path, may be slow on first access)
    """
    global _GLOBAL_VERIFIER
    if _GLOBAL_VERIFIER is None:
        import os
        local_path = get_coevokg_env("CHAIN_DATA_LOCAL", "")
        load_path = local_path if (local_path and os.path.exists(local_path)) else chain_data_path
        logger.info(f"Initialising global PathSupportVerifier from {load_path}")
        _GLOBAL_VERIFIER = PathSupportVerifier(load_path)
    return _GLOBAL_VERIFIER
