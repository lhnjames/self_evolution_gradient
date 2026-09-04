"""
KGWriterActor — Ray remote actor for CoT-augmented chain persistence.

Extracts knowledge chains from correct solver trajectories and writes them
back to the proposer's chain pool, enabling co-evolution between the solver
and the knowledge graph.

Chain format written:
  {
    "id":           "<hex uuid>",
    "relations":    ["Entity1", "Entity2", ..., "EntityN"],
    "nodes":        {"Entity1": "<text>", ...},   # augmented with solver retrieval
    "source":       "cot_augmented",
    "step":         <training_step>,
    "relation_labels": [...]   # optional, forwarded from original chain
  }
"""

import hashlib
import json
import logging
import os
from typing import Dict, List, Optional

import ray

logger = logging.getLogger(__name__)


@ray.remote
class KGWriterActor:
    """Ray remote actor that collects CoT-augmented chains and persists them.

    Thread-safe via Ray's single-threaded actor model.  Deduplication is based
    on an MD5 hash of the lowercased entity sequence so that the same chain
    arriving from different solver trajectories is written only once.
    """

    def __init__(self, save_path: str, max_size: int = 50000) -> None:
        self.save_path = save_path
        self.max_size = max_size
        # Dedup key: md5("|".join(relations_lower))
        self._seen: set = set()
        # Buffer of newly-added chains not yet pulled by the trainer
        self._new_chains: List[Dict] = []
        # Pending disk writes
        self._pending: List[Dict] = []
        self._total_written = 0
        self._total_skipped = 0

        self._load()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def add_chain(self, chain: Dict) -> bool:
        """Add or re-enrich a chain.

        Policy change (relaxed dedup):
          - A chain whose entity sequence has NEVER been seen → always accept.
          - A chain already seen → accept again only if it carries new solver
            evidence (solver_evidence key present).  This allows the same chain
            to accumulate evidence across training steps while still preventing
            bare duplicate writes within a single step.

        Returns True when the chain was written (new or enriched).
        """
        relations = chain.get("relations", [])
        if not relations:
            return False

        key = self._hash(relations)
        has_evidence = bool(
            chain.get("solver_evidence") or
            any(v for v in chain.get("nodes", {}).values())
        )

        if key in self._seen and not has_evidence:
            self._total_skipped += 1
            return False

        if len(self._seen) >= self.max_size:
            self._total_skipped += 1
            return False

        self._seen.add(key)
        self._new_chains.append(chain)
        self._pending.append(chain)
        self._total_written += 1

        if len(self._pending) >= 10:
            self._flush()

        return True

    def get_new_chains(self) -> List[Dict]:
        """Pull and clear the buffer of newly-added chains.

        Called by the trainer after a batch of add_chain() calls so it can
        synchronously push the chains into the in-memory KnowledgeChainPool.
        """
        chains = list(self._new_chains)
        self._new_chains.clear()
        return chains

    def flush(self) -> None:
        """Force-flush any pending writes to disk."""
        self._flush()

    def stats(self) -> Dict:
        return {
            "total_written": self._total_written,
            "total_skipped": self._total_skipped,
            "seen_size": len(self._seen),
            "pending_new": len(self._new_chains),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash(relations: List[str]) -> str:
        key = "|".join(r.lower().strip() for r in relations)
        return hashlib.md5(key.encode("utf-8")).hexdigest()

    def _flush(self) -> None:
        if not self._pending:
            return
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.save_path)), exist_ok=True)
            with open(self.save_path, "a", encoding="utf-8") as f:
                for chain in self._pending:
                    f.write(json.dumps(chain, ensure_ascii=False) + "\n")
            logger.debug(f"KGWriterActor: flushed {len(self._pending)} chains → {self.save_path}")
        except Exception as e:
            logger.warning(f"KGWriterActor: flush failed: {e}")
        finally:
            self._pending.clear()

    def _load(self) -> None:
        """Restore the dedup set from an existing save file on restart."""
        if not os.path.exists(self.save_path):
            return
        loaded = 0
        try:
            with open(self.save_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    relations = record.get("relations", [])
                    if relations:
                        self._seen.add(self._hash(relations))
                        loaded += 1
            logger.info(
                f"KGWriterActor: restored {loaded} seen-chain keys from {self.save_path}"
            )
        except Exception as e:
            logger.warning(f"KGWriterActor: failed to load from {self.save_path}: {e}")
