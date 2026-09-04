"""
Self-play Ray PPO trainer for CoEvoKG.

This trainer implements the self-play algorithm where a model acts as both
problem generator and problem solver in alternating phases within each global step.

Self-play flow:
1. Problem Generation Phase: Model generates problems from seeded data
2. Problem Extraction: Extract questions from generation trajectories
3. Problem Solving Phase: Model solves the extracted problems
4. Dual Training: Update model twice with rewards from both phases
"""

import json
import logging
import os
import re
import time
import uuid
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pprint import pprint
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import ray
import torch
import verl.utils.torch_functional as verl_F
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    process_validation_metrics,
)
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, apply_kl_penalty, compute_advantage, compute_response_mask
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.model import compute_position_id_with_mask
from verl.utils.tracking import Tracking

from coevokg.utils.coevokg_data_manager import SelfPlayPhase, CoEvoKGDataManager
from coevokg.utils.problem_extraction import ProblemExtractor, extract_problems_batch
from coevokg.utils.knowledge_chain_proposer import KnowledgeChainPool, build_seed_solver_problem, extract_selected_hops, extract_selected_entity
from coevokg.utils.kg_writer import KGWriterActor

logger = logging.getLogger(__name__)

# Enable debug logging for self-play
SELF_PLAY_DEBUG = os.environ.get("SELF_PLAY_DEBUG", "False").lower() == "true"
if SELF_PLAY_DEBUG:
    logger.setLevel(logging.DEBUG)


class CoEvoKGRayPPOTrainer(RayPPOTrainer):
    """
    Self-Play Ray PPO Trainer that implements the dual-phase training process.

    This trainer orchestrates the self-play loop:
    - Phase 1: Problem generation and extraction
    - Phase 2: Problem solving
    - Dual model updates with separate reward computations
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping,
        resource_pool_manager,
        ray_worker_group_cls=None,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset=None,
        val_dataset=None,
        collate_fn=None,
        train_sampler=None,
        device_name=None,
    ):
        self.sp_config = config.self_play

        super().__init__(
            config,
            tokenizer,
            role_worker_mapping,
            resource_pool_manager,
            ray_worker_group_cls,
            processor,
            reward_fn,
            val_reward_fn,
            train_dataset,
            val_dataset,
            collate_fn,
            train_sampler,
            device_name,
        )

        self.current_phase = SelfPlayPhase.PROBLEM_GENERATION

        self.sp_data_manager = CoEvoKGDataManager.remote()
        _extraction_log_path = os.path.join(
            os.path.dirname(self.config.trainer.default_local_dir),
            "extraction_log.jsonl",
        )
        self.problem_extractor = ProblemExtractor(
            lang=self.config.self_play.lang,
            use_rag_filter=self.sp_config.use_rag_filter,
            use_search_terms_filter=self.sp_config.use_search_terms_filter,
            noisy_rag_materials=self.sp_config.get("noisy_RAG_materials", 0),
            answer_pattern=self.sp_config.get("answer_pattern", "answer"),
            log_path=_extraction_log_path,
        )

        self._load_problem_generation_data()
        self._build_seed_question_index()

        # Walk mode initialisation
        self._proposer_mode = self.sp_config.proposer.get("mode", "seed")
        self.chain_pool: Optional[KnowledgeChainPool] = None
        self._walk_seed_entities: List[str] = []  # HotpotQA ground-truth entities
        if self._proposer_mode == "walk":
            walk_cfg = self.sp_config.proposer.get("walk", {})
            min_hops = walk_cfg.get("min_hops", 3)
            walk_source = walk_cfg.get("source", "file")
            # generate_n defaults to 2 × train_batch_size
            default_gen_n = self.config.data.train_batch_size * 2
            self._walk_generate_n = int(walk_cfg.get("generate_n") or default_gen_n)

            if walk_source != "file":
                raise ValueError("This release package supports self_play.proposer.walk.source='file' only.")
            chain_data_path = walk_cfg.get("chain_data_path", None)
            if not chain_data_path:
                raise ValueError(
                    "self_play.proposer.mode='walk' requires "
                    "self_play.proposer.walk.chain_data_path to be set."
                )
            use_search_proposer = bool(self.sp_config.proposer.get("use_search", False))
            self.chain_pool = KnowledgeChainPool(
                mode="file",
                chain_data_path=chain_data_path,
                min_hops=min_hops,
                lang=self.sp_config.lang,
                use_search_proposer=use_search_proposer,
            )
            logger.info(
                f"Walk mode enabled: chain_pool size={len(self.chain_pool.chains)}, "
                f"generate_n={self._walk_generate_n}, min_hops={min_hops}, "
                f"use_search_proposer={use_search_proposer}"
            )
        # Walk mode: track selected proposer→solver mapping
        self._walk_selected_map: Dict[int, int] = {}   # proposer_idx → solver_pos
        self._walk_generate_n_last: int = 0             # generate_n used in last step

        # KGWriter: co-evolution write-back of correct solver CoT chains
        self.kg_writer: Optional[ray.actor.ActorHandle] = None
        kg_cfg = self.sp_config.get("kg_writer", {})
        if kg_cfg.get("enable", False) and self.chain_pool is not None:
            kg_save_path = kg_cfg.get("save_path") or os.path.join(
                config.trainer.get("rollout_data_dir", "/tmp/coevokg_rollout"),
                "kg_cot_chains.jsonl",
            )
            kg_max_size = int(kg_cfg.get("max_size", 50000))
            self.kg_writer = KGWriterActor.remote(
                save_path=kg_save_path,
                max_size=kg_max_size,
            )
            logger.info(
                f"KGWriterActor initialized → {kg_save_path} (max_size={kg_max_size})"
            )

        # ----------------------------------------------------------------
        # Question quality checker for candidate filtering.
        # ----------------------------------------------------------------
        from coevokg.utils.question_quality_checker import QuestionQualityChecker
        from coevokg.utils.env import get_coevokg_env, setdefault_coevokg_env

        api_cfg = config.get("api", {})
        self._quality_check_max_workers = int(api_cfg.get("quality_check_max_workers", 32))

        # Export judge concurrency and timeout defaults for compute_score_batch and LLMJudge.
        # Values exported by the launch script take priority; yaml values are fallbacks.
        setdefault_coevokg_env("JUDGE_MAX_WORKERS",     str(int(api_cfg.get("judge_max_workers", 32))))
        setdefault_coevokg_env("JUDGE_TIMEOUT",         str(float(api_cfg.get("judge_timeout", 60.0))))
        setdefault_coevokg_env("JUDGE_MAX_RETRIES",     str(int(api_cfg.get("judge_max_retries", 0))))
        setdefault_coevokg_env("JUDGE_MAX_CHARS",       str(int(api_cfg.get("judge_max_chars", 2500))))
        setdefault_coevokg_env("JUDGE_CHAIN_MAX_TOKENS", str(int(api_cfg.get("judge_chain_max_tokens", 192))))

        self.quality_checker = QuestionQualityChecker(
            base_url=get_coevokg_env("BASE_URL", ""),
            api_key=get_coevokg_env("API_KEY", "dummy"),
            model=get_coevokg_env("MODEL", ""),
            extra_api_keys=[k for k in [get_coevokg_env("API_KEY_2")] if k] or None,
        )
        logger.info(
            "QuestionQualityChecker initialized: quality_workers=%d, judge_workers=%s, "
            "judge_timeout=%ss, judge_max_retries=%s, judge_max_chars=%s, judge_chain_max_tokens=%s",
            self._quality_check_max_workers,
            get_coevokg_env("JUDGE_MAX_WORKERS"),
            get_coevokg_env("JUDGE_TIMEOUT"),
            get_coevokg_env("JUDGE_MAX_RETRIES"),
            get_coevokg_env("JUDGE_MAX_CHARS"),
            get_coevokg_env("JUDGE_CHAIN_MAX_TOKENS"),
        )

    def _validate_config(self):
        val = self.sp_config.get("validate_config", True)
        if isinstance(val, str):
            val = val.lower() in ("1", "true", "yes", "y")
        if val:
            super()._validate_config()
        else:
            print("Skipping config validation as per self_play.validate_config=False")

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path, extra_fields=None):
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        if extra_fields:
            for k, v in extra_fields.items():
                if len(v) == n:
                    base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _validate(self):
        data_source_lst = []
        reward_model_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        sample_turns = []

        for test_data in tqdm(self.val_dataloader, desc="Validation Progress"):
            test_batch = DataProto.from_single_dict(test_data)

            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            input_ids = test_batch.batch["input_ids"]

            if input_ids.dtype != torch.long:
                print(f"WARNING: Converting input_ids from {input_ids.dtype} to torch.long")
                input_ids = input_ids.long()

            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "interaction_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("interaction_kwargs")
            if "agent_name" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("agent_name")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            output_ids = test_output_gen_batch.batch["responses"]

            print(f"DEBUG: Validation batch #{len(data_source_lst) + 1}")
            print(f"DEBUG: output_ids type: {type(output_ids)}")
            print(f"DEBUG: output_ids dtype: {output_ids.dtype}")
            print(f"DEBUG: output_ids shape: {output_ids.shape}")
            print(f"DEBUG: output_ids device: {output_ids.device}")
            print(f"DEBUG: test_gen_batch meta_info: {test_gen_batch.meta_info}")

            if output_ids.dtype != torch.long:
                print(f"ERROR: Found float32 output_ids!")
                print(f"DEBUG: output_ids min/max: {output_ids.min().item()} / {output_ids.max().item()}")
                print(f"DEBUG: Contains non-integer values: {not torch.all(output_ids == output_ids.long())}")
                print(f"DEBUG: First few values: {output_ids[0, :100]}")

                is_integer_valued = torch.all(output_ids == output_ids.round())
                print(f"DEBUG: All values are integer-valued: {is_integer_valued}")

                print(f"WARNING: Converting output_ids from {output_ids.dtype} to torch.long")
                if is_integer_valued:
                    output_ids = output_ids.long()
                else:
                    print("ERROR: Cannot safely convert non-integer floats to long!")
                    output_ids = output_ids.round().long()

            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            for key in test_batch.batch.keys():
                if key not in ["old_log_probs", "ref_log_prob"]:
                    test_batch.batch[key] = test_batch.batch[key].long()

            test_batch.meta_info["validate"] = True

            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            print(f"len reward_extra_infos_dict['reward']: {len(reward_extra_infos_dict['reward'])}")
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)
                    print(f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}")

            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))
            reward_model_lst.append(
                test_batch.non_tensor_batch.get("reward_model", ["unknown"] * reward_tensor.shape[0])
            )

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            extra_fields = {}
            data_sources = np.concatenate(data_source_lst, axis=0) if data_source_lst else None
            reward_models = np.concatenate(reward_model_lst, axis=0) if reward_model_lst else None

            if data_sources is not None:
                extra_fields["data_source"] = data_sources
            if reward_models is not None:
                extra_fields["reward_model"] = reward_models
            extra_fields = extra_fields if extra_fields else None

            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
                extra_fields=extra_fields,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        def _fill_metric_dict(target: dict, src2var2metric2val: dict, data_source_label: str = None):
            for data_source, var2metric2val in src2var2metric2val.items():
                key = data_source_label if data_source_label is not None else data_source
                core_var = "acc" if "acc" in var2metric2val else "reward"
                for var_name, metric2val in var2metric2val.items():
                    n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                    for metric_name, metric_val in metric2val.items():
                        if (
                            (var_name == core_var)
                            and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                            and (f"@{n_max}" in metric_name)
                        ):
                            metric_sec = "val-core"
                        else:
                            metric_sec = "val-aux"
                        target[f"{metric_sec}/{key}/{var_name}/{metric_name}"] = metric_val

        metric_dict = {}

        # Per-dataset metrics (e.g. val-core/hotpotqa/acc/mean@5)
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        _fill_metric_dict(metric_dict, data_src2var2metric2val)

        # Overall metrics across all datasets (val-core/overall/acc/mean@N)
        overall_sources = np.array(["overall"] * len(data_sources))
        overall_src2var2metric2val = process_validation_metrics(overall_sources, sample_inputs, reward_extra_infos_dict)
        _fill_metric_dict(metric_dict, overall_src2var2metric2val)

        # Simple scalar accuracy for this validation round — no @N suffix
        if sample_scores:
            metric_dict["val-core/overall/acc"] = float(np.mean(sample_scores))

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def _load_problem_generation_data(self):
        try:
            generation_data_path = self.sp_config.get("generation_data_path")
            if generation_data_path and os.path.exists(generation_data_path):
                with open(generation_data_path, "r") as f:
                    if generation_data_path.endswith(".jsonl"):
                        generation_data = [json.loads(line) for line in f]
                    else:
                        generation_data = json.load(f)

                ray.get(self.sp_data_manager.set_problem_generation_data.remote(generation_data))
                logger.info(f"Loaded {len(generation_data)} problem generation samples")
            else:
                proposer_mode = self.sp_config.get("proposer", {}).get("mode", "seed")
                if proposer_mode == "walk":
                    logger.debug(
                        "generation_data_path not set; using dummy placeholder (walk mode derives "
                        "seed entities from chain_data_path, not from generation_data_path)"
                    )
                else:
                    logger.warning("No generation data path specified or file not found")
                dummy_data = self._create_dummy_generation_data()
                generation_data = dummy_data
                ray.get(self.sp_data_manager.set_problem_generation_data.remote(dummy_data))

            # Extract ground-truth entities for walk-mode online chain generation.
            # These serve as starting points for the KILT random walk.
            entities = []
            for record in generation_data:
                gt = record.get("reward_model", {}).get("ground_truth", {})
                target = gt.get("target", "") if isinstance(gt, dict) else str(gt or "")
                if target and target.lower() not in ("unknown", "dummy", ""):
                    entities.append(target)
            self._walk_seed_entities = entities
            if entities:
                logger.info(
                    f"Walk seed entities: {len(entities)} extracted from generation data "
                    f"(examples: {entities[:3]})"
                )

        except Exception as e:
            logger.error(f"Error loading problem generation data: {e}")
            raise

    def _create_dummy_generation_data(self) -> List[Dict[str, Any]]:
        dummy_data = [
            {
                "prompt": [{"role": "user", "content": "What are the main causes of the Great Depression, and how did government policies influence its course?"}],
                "data_source": "dummy_generation",
                "reward_model": {"style": "rule", "ground_truth": {"target": "dummy_answer"}},
                "extra_info": {"type": "qa_search"},
            },
            {
                "prompt": [{"role": "user", "content": "How does CRISPR gene editing work, and what are its potential applications in medicine?"}],
                "data_source": "dummy_generation",
                "reward_model": {"style": "rule", "ground_truth": {"target": "dummy_answer"}},
                "extra_info": {"type": "qa_search"},
            },
        ]
        return dummy_data

    def fit(self):
        from verl.trainer.ppo.metric_utils import (
            compute_throughout_metrics,
            compute_timing_metrics,
        )

        from coevokg.utils.patch.metric_patch import quarl_compute_data_metrics

        tracking_logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name + "_selfplay",
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        self._load_checkpoint()

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            if val_metrics:
                pprint(f"Initial validation metrics: {val_metrics}")
                tracking_logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Self-Play Training")

        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        reward_dynamic_sampling_enabled = self.sp_config.get("reward_dynamic_sampling", {}).get("enable", False)

        if reward_dynamic_sampling_enabled:
            accumulated_proposer_batch = None
            accumulated_solver_batch = None
            last_step_proposer_batch = None
            last_step_solver_batch = None
            num_prompt_in_batch = 0
            num_gen_batches = 0
            target_prompt_batch_size = self.config.data.train_batch_size
            max_num_gen_batches = self.sp_config.get("reward_dynamic_sampling", {}).get("max_num_gen_batches", 20)

        for epoch in range(self.config.trainer.total_epochs):
            dataloader_iter = iter(self.train_dataloader)
            print("Epoch:", epoch)
            while True:
                try:
                    timing_raw = {}
                    all_metrics = {}

                    is_last_step = self.global_steps >= self.total_training_steps

                    with marked_timer("self_play_step", timing_raw):

                        step_metrics = {}

                        with marked_timer("proposer_generation", timing_raw):
                            proposer_gen_batch, solver_batch, sampling_metrics = (
                                self._generate_proposer_with_dynamic_sampling(dataloader_iter, timing_raw)
                            )

                            if proposer_gen_batch is None and solver_batch is None:
                                print(
                                    "No valid problems extracted after dynamic sampling, dataloader exhausted, breaking to next epoch"
                                )
                                break

                            step_metrics["self_play/proposer_samples"] = len(proposer_gen_batch.batch)
                            step_metrics["self_play/extracted_problems"] = len(solver_batch.batch)
                            step_metrics.update(sampling_metrics)
                            step_metrics.update(solver_batch.meta_info.get("metrics", {}))
                    mini_epoch = 0
                    while mini_epoch < self.sp_config.mini_epochs:

                        mini_epoch += 1

                        if mini_epoch == 1:
                            proposer_gen_batch_copy = deepcopy(proposer_gen_batch)
                            solver_batch_copy = deepcopy(solver_batch)
                        else:
                            proposer_gen_batch = deepcopy(proposer_gen_batch_copy)
                            solver_batch = deepcopy(solver_batch_copy)

                        with marked_timer("solver_generation", timing_raw):
                            solver_gen_batch = self._generate_solver_trajectories(solver_batch, timing_raw)
                            step_metrics["self_play/solver_samples"] = len(solver_gen_batch.batch)

                        # ── Overlap: API reward + GPU log-prob prefill ────────────────────────
                        # When async reward is enabled and we are NOT in dynamic-sampling mode
                        # (which manages its own reward calls), we can overlap the LLM-judge API
                        # wait (~350-400 s) with computing old_log_probs + ref_log_probs on GPU
                        # (~40 s).  The reward future is launched immediately after solver
                        # generation; GPU workers are kept busy until the future is resolved.
                        if self.config.reward_model.launch_reward_fn_async and not reward_dynamic_sampling_enabled:
                            future_reward = self._calculate_self_play_rewards(
                                proposer_gen_batch, solver_gen_batch, timing_raw, _return_future=True
                            )
                            # GPU work that doesn't depend on rewards
                            with marked_timer("reward_overlap_prefill", timing_raw):
                                proposer_gen_batch, _pm = self._prefill_policy_log_probs(
                                    proposer_gen_batch, "proposer", timing_raw
                                )
                                step_metrics.update({f"proposer/{k}": v for k, v in _pm.items()})
                                solver_gen_batch, _sm = self._prefill_policy_log_probs(
                                    solver_gen_batch, "solver", timing_raw
                                )
                                step_metrics.update({f"solver/{k}": v for k, v in _sm.items()})
                            # Block on residual API wait + compute proposer rewards
                            with marked_timer("reward_calculation", timing_raw):
                                proposer_rewards, solver_rewards = self._finalize_self_play_rewards(
                                    future_reward, proposer_gen_batch, solver_gen_batch, timing_raw
                                )
                        else:
                            with marked_timer("reward_calculation", timing_raw):
                                proposer_rewards, solver_rewards = self._calculate_self_play_rewards(
                                    proposer_gen_batch, solver_gen_batch, timing_raw
                                )

                        self._attach_self_play_rewards(
                            proposer_gen_batch,
                            solver_gen_batch,
                            proposer_rewards,
                            solver_rewards,
                            step_metrics,
                        )

                        # ── CoT → KG co-evolution ────────────────────────────────────────
                        if self.kg_writer is not None:
                            kg_added = self._write_correct_trajectories_to_kg(
                                solver_gen_batch, solver_rewards
                            )
                            step_metrics["kg_writer/chains_added"] = kg_added
                            step_metrics["kg_writer/pool_size"] = len(self.chain_pool.chains)

                        if reward_dynamic_sampling_enabled:
                            filtered_proposer_batch, filtered_solver_batch = self._apply_reward_dynamic_sampling(
                                proposer_gen_batch, solver_gen_batch, timing_raw
                            )

                            unique_prompts = set(filtered_proposer_batch.non_tensor_batch["uid"])
                            num_prompt_in_batch += len(unique_prompts)
                            num_gen_batches += 1

                            if last_step_proposer_batch is not None and accumulated_proposer_batch is None:
                                saved_unique_prompts = set(last_step_proposer_batch.non_tensor_batch["uid"])
                                num_prompt_in_batch += len(saved_unique_prompts)
                                print(f"Added {len(saved_unique_prompts)} prompts from saved data")

                            if accumulated_proposer_batch is None:
                                if last_step_proposer_batch is not None:
                                    print(
                                        f"Using {len(last_step_proposer_batch.batch)} saved proposer trajectories from last step"
                                    )
                                    accumulated_proposer_batch = DataProto.concat(
                                        [last_step_proposer_batch, filtered_proposer_batch]
                                    )
                                else:
                                    accumulated_proposer_batch = filtered_proposer_batch

                                if last_step_solver_batch is not None:
                                    print(
                                        f"Using {len(last_step_solver_batch.batch)} saved solver trajectories from last step"
                                    )
                                    accumulated_solver_batch = DataProto.concat(
                                        [last_step_solver_batch, filtered_solver_batch]
                                    )
                                else:
                                    accumulated_solver_batch = filtered_solver_batch
                            else:
                                accumulated_proposer_batch = DataProto.concat(
                                    [accumulated_proposer_batch, filtered_proposer_batch]
                                )
                                accumulated_solver_batch = DataProto.concat(
                                    [accumulated_solver_batch, filtered_solver_batch]
                                )

                            print(
                                f"Reward filtering: {len(unique_prompts)} prompts kept, total: {num_prompt_in_batch}/{target_prompt_batch_size}"
                            )

                            # Check if we have enough prompts
                            if num_prompt_in_batch < target_prompt_batch_size:
                                print(f"Need more prompts: {num_prompt_in_batch} < {target_prompt_batch_size}")
                                if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                    print(f"Batch {num_gen_batches}/{max_num_gen_batches}. Continue generating...")
                                    progress_bar.update(1)
                                    self.global_steps += 1
                                    continue
                                else:
                                    print(
                                        f"Max batches reached ({num_gen_batches}/{max_num_gen_batches}). Using accumulated data."
                                    )

                            # Use accumulated batches for training
                            proposer_gen_batch = accumulated_proposer_batch
                            solver_gen_batch = accumulated_solver_batch

                            # Truncate to target size if we have more than needed
                            proposer_n = self.sp_config.get("proposer", {}).get(
                                "n", self.config.actor_rollout_ref.rollout.n
                            )
                            target_traj_size = target_prompt_batch_size * proposer_n

                            if len(proposer_gen_batch.batch) > target_traj_size:
                                print(
                                    f"Truncating proposer batch from {len(proposer_gen_batch.batch)} to {target_traj_size}"
                                )
                                last_step_proposer_batch = proposer_gen_batch[target_traj_size:]
                                proposer_gen_batch = proposer_gen_batch[:target_traj_size]
                                print(
                                    f"Saved {len(last_step_proposer_batch.batch)} proposer trajectories for next round"
                                )
                            else:
                                last_step_proposer_batch = None

                            solver_n = self.config.actor_rollout_ref.rollout.n
                            target_solver_size = target_prompt_batch_size * proposer_n * solver_n

                            if len(solver_gen_batch.batch) > target_solver_size:
                                print(
                                    f"Truncating solver batch from {len(solver_gen_batch.batch)} to {target_solver_size}"
                                )
                                last_step_solver_batch = solver_gen_batch[target_solver_size:]
                                solver_gen_batch = solver_gen_batch[:target_solver_size]
                                print(f"Saved {len(last_step_solver_batch.batch)} solver trajectories for next round")
                            else:
                                last_step_solver_batch = None

                            print(
                                f"Final batch sizes: proposer={len(proposer_gen_batch.batch)}, solver={len(solver_gen_batch.batch)}"
                            )

                            step_metrics["self_play/reward_dynamic_sampling/batches_used"] = num_gen_batches
                            step_metrics["self_play/reward_dynamic_sampling/final_prompt_count"] = num_prompt_in_batch
                            step_metrics["self_play/reward_dynamic_sampling/target_prompt_count"] = (
                                target_prompt_batch_size
                            )

                            step_metrics["self_play/reward_dynamic_sampling/final_proposer_trajectories"] = len(
                                proposer_gen_batch.batch
                            )
                            step_metrics["self_play/reward_dynamic_sampling/final_solver_trajectories"] = len(
                                solver_gen_batch.batch
                            )

                            if last_step_proposer_batch is not None:
                                step_metrics["self_play/reward_dynamic_sampling/saved_proposer_trajectories"] = len(
                                    last_step_proposer_batch.batch
                                )
                            if last_step_solver_batch is not None:
                                step_metrics["self_play/reward_dynamic_sampling/saved_solver_trajectories"] = len(
                                    last_step_solver_batch.batch
                                )

                        if self.sp_config.save_freq > 0 and (
                            self.global_steps % self.sp_config.save_freq == 0
                            or self.sp_config.reward_dynamic_sampling.enable
                        ):
                            try:
                                inputs = self.tokenizer.batch_decode(
                                    proposer_gen_batch.batch["prompts"], skip_special_tokens=True
                                )
                                outputs = self.tokenizer.batch_decode(
                                    proposer_gen_batch.batch["responses"], skip_special_tokens=True
                                )

                                reward_extra_infos_dict = {}
                                if (
                                    hasattr(proposer_gen_batch, "non_tensor_batch")
                                    and proposer_gen_batch.non_tensor_batch
                                ):
                                    for key in ["data_source", "reward_model", "extra_info", "uid"]:
                                        if key in proposer_gen_batch.non_tensor_batch:
                                            reward_extra_infos_dict[key] = proposer_gen_batch.non_tensor_batch[
                                                key
                                            ].tolist()

                                self._dump_trajectories(
                                    inputs=inputs,
                                    outputs=outputs,
                                    scores=proposer_gen_batch.batch["token_level_scores"].sum(-1).cpu().tolist(),
                                    role="proposer",
                                    step=self.global_steps,
                                    reward_extra_infos_dict=(
                                        reward_extra_infos_dict if reward_extra_infos_dict else None
                                    ),
                                )
                            except Exception as e:
                                logger.warning(f"Failed to dump proposer trajectories: {e}")

                            try:
                                inputs = self.tokenizer.batch_decode(
                                    solver_gen_batch.batch["prompts"], skip_special_tokens=True
                                )
                                outputs = self.tokenizer.batch_decode(
                                    solver_gen_batch.batch["responses"], skip_special_tokens=True
                                )

                                reward_extra_infos_dict = {}
                                if hasattr(solver_gen_batch, "non_tensor_batch") and solver_gen_batch.non_tensor_batch:
                                    for key in ["data_source", "reward_model", "extra_info", "uid"]:
                                        if key in solver_gen_batch.non_tensor_batch:
                                            reward_extra_infos_dict[key] = solver_gen_batch.non_tensor_batch[
                                                key
                                            ].tolist()

                                self._dump_trajectories(
                                    inputs=inputs,
                                    outputs=outputs,
                                    scores=solver_gen_batch.batch["token_level_scores"].sum(-1).cpu().tolist(),
                                    role="solver",
                                    step=self.global_steps,
                                    reward_extra_infos_dict=(
                                        reward_extra_infos_dict if reward_extra_infos_dict else None
                                    ),
                                )
                            except Exception as e:
                                logger.warning(f"Failed to dump proposer trajectories: {e}")

                        combine_updates = getattr(self.sp_config, "combine_update", False)
                        update_order = self.sp_config.get("update_order", "solver_then_proposer")

                        with marked_timer("model_updates", timing_raw):

                            representative_batch = solver_gen_batch
                            should_update_solver = (
                                self.sp_config.solver.enable
                                and solver_gen_batch is not None
                                and (
                                    self.global_steps > self.sp_config.proposer.warm_up_steps
                                    or not self.sp_config.proposer.enable
                                )
                            )

                            should_prefill_proposer = (
                                self.sp_config.proposer.enable
                                and proposer_gen_batch is not None
                                and combine_updates
                            )
                            if should_prefill_proposer:
                                proposer_gen_batch, prefill_metrics = self._prefill_policy_log_probs(
                                    proposer_gen_batch, "proposer", timing_raw
                                )
                                step_metrics.update({f"proposer/{k}": v for k, v in prefill_metrics.items()})

                            if combine_updates and should_update_solver:
                                solver_gen_batch, prefill_metrics = self._prefill_policy_log_probs(
                                    solver_gen_batch, "solver", timing_raw
                                )
                                step_metrics.update({f"solver/{k}": v for k, v in prefill_metrics.items()})

                        if combine_updates:
                            batches_to_combine = []
                            if self.sp_config.proposer.enable and proposer_gen_batch is not None:
                                batches_to_combine.append(proposer_gen_batch)
                            if (
                                self.sp_config.solver.enable
                                and solver_gen_batch is not None
                                and (
                                    self.global_steps > self.sp_config.proposer.warm_up_steps
                                    or not self.sp_config.proposer.enable
                                )
                            ):
                                batches_to_combine.append(solver_gen_batch)

                            if len(batches_to_combine) > 1:
                                prepared_batches = []
                                if self.sp_config.proposer.enable and proposer_gen_batch is not None:
                                    proposer_gen_batch, proposer_metrics = self._prepare_trajectories_for_update(
                                        proposer_gen_batch, "proposer", timing_raw
                                    )
                                    step_metrics.update({f"proposer/{k}": v for k, v in proposer_metrics.items()})
                                    prepared_batches.append(proposer_gen_batch)

                                if (
                                    self.sp_config.solver.enable
                                    and solver_gen_batch is not None
                                    and (
                                        self.global_steps > self.sp_config.proposer.warm_up_steps
                                        or not self.sp_config.proposer.enable
                                    )
                                ):
                                    solver_gen_batch, solver_metrics = self._prepare_trajectories_for_update(
                                        solver_gen_batch, "solver", timing_raw
                                    )
                                    step_metrics.update({f"solver/{k}": v for k, v in solver_metrics.items()})
                                    prepared_batches.append(solver_gen_batch)

                                combined_batch = self._concat_prepared_update_batches(prepared_batches)

                                if self.use_critic:
                                    with marked_timer("combined_update_critic", timing_raw, color="pink"):
                                        critic_output = self.critic_wg.update_critic(combined_batch)
                                    critic_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                                    step_metrics.update({f"combined/critic_{k}": v for k, v in critic_metrics.items()})

                                combined_metrics = self._update_actor_on_prepared_trajectories(
                                    combined_batch, "combined", timing_raw
                                )
                                combined_metrics.update(self._reward_summary_metrics(combined_batch, "combined"))
                                step_metrics.update({f"combined/{k}": v for k, v in combined_metrics.items()})
                                representative_batch = combined_batch
                            else:
                                if self.sp_config.proposer.enable and proposer_gen_batch is not None:
                                    proposer_metrics = self._update_on_trajectories(
                                        proposer_gen_batch, "proposer", timing_raw
                                    )
                                    step_metrics.update({f"proposer/{k}": v for k, v in proposer_metrics.items()})
                                    representative_batch = proposer_gen_batch

                                if (
                                    self.sp_config.solver.enable
                                    and solver_gen_batch is not None
                                    and (
                                        self.global_steps > self.sp_config.proposer.warm_up_steps
                                        or not self.sp_config.proposer.enable
                                    )
                                ):
                                    solver_metrics = self._update_on_trajectories(
                                        solver_gen_batch, "solver", timing_raw
                                    )
                                    step_metrics.update({f"solver/{k}": v for k, v in solver_metrics.items()})
                                    representative_batch = solver_gen_batch

                        else:
                            solver_update_steps = int(self.sp_config.get("solver", {}).get("update_steps", 1))
                            solver_update_steps = max(solver_update_steps, 1)
                            step_metrics["self_play/solver_update_steps"] = solver_update_steps
                            proposer_update_batches = []
                            solver_round_batches = [
                                {
                                    "proposer_gen_batch": proposer_gen_batch,
                                    "solver_batch": solver_batch,
                                    "solver_gen_batch": solver_gen_batch,
                                    "proposer_rewards": proposer_rewards,
                                    "solver_rewards": solver_rewards,
                                }
                            ]

                            if self.sp_config.proposer.enable and proposer_gen_batch is not None:
                                proposer_gen_batch, prefill_metrics = self._prefill_policy_log_probs(
                                    proposer_gen_batch, "proposer", timing_raw
                                )
                                solver_round_batches[0]["proposer_gen_batch"] = proposer_gen_batch
                                proposer_update_batches.append(proposer_gen_batch)
                                step_metrics.update({f"proposer/{k}": v for k, v in prefill_metrics.items()})

                           
                            if (
                                solver_update_steps > 1
                                and self.sp_config.proposer.enable
                                and proposer_gen_batch is not None
                            ):
                                for round_idx in range(1, solver_update_steps):
                                    with marked_timer(f"proposer_generation_{round_idx + 1}", timing_raw):
                                        next_proposer_batch, next_solver_batch, sampling_metrics = (
                                            self._generate_proposer_with_dynamic_sampling(
                                                dataloader_iter, timing_raw
                                            )
                                        )
                                    if next_proposer_batch is None or next_solver_batch is None:
                                        print(
                                            "No valid problems extracted during proposer pre-generation, "
                                            "using fewer solver update rounds this step"
                                        )
                                        break

                                    next_proposer_batch, prefill_metrics = self._prefill_policy_log_probs(
                                        next_proposer_batch, f"proposer/round_{round_idx + 1}", timing_raw
                                    )
                                    solver_round_batches.append(
                                        {
                                            "proposer_gen_batch": next_proposer_batch,
                                            "solver_batch": next_solver_batch,
                                            "solver_gen_batch": None,
                                            "proposer_rewards": None,
                                            "solver_rewards": None,
                                        }
                                    )
                                    proposer_update_batches.append(next_proposer_batch)
                                    step_metrics[f"round_{round_idx + 1}/self_play/proposer_samples"] = len(
                                        next_proposer_batch.batch
                                    )
                                    step_metrics[f"round_{round_idx + 1}/self_play/extracted_problems"] = len(
                                        next_solver_batch.batch
                                    )
                                    step_metrics.update(
                                        {f"round_{round_idx + 1}/{k}": v for k, v in sampling_metrics.items()}
                                    )
                                    step_metrics.update(
                                        {
                                            f"round_{round_idx + 1}/{k}": v
                                            for k, v in next_solver_batch.meta_info.get("metrics", {}).items()
                                        }
                                    )
                                    step_metrics.update(
                                        {
                                            f"round_{round_idx + 1}/proposer/{k}": v
                                            for k, v in prefill_metrics.items()
                                        }
                                    )

                            actual_solver_update_steps = len(solver_round_batches)
                            step_metrics["self_play/actual_solver_update_steps"] = actual_solver_update_steps

                            def _run_solver_updates():
                                nonlocal representative_batch, proposer_gen_batch, solver_batch, solver_gen_batch
                                nonlocal proposer_rewards, solver_rewards
                                if not (
                                    self.sp_config.solver.enable
                                    and (
                                        self.global_steps > self.sp_config.proposer.warm_up_steps
                                        or not self.sp_config.proposer.enable
                                    )
                                ):
                                    return
                                for update_idx, round_state in enumerate(solver_round_batches):
                                    proposer_gen_batch = round_state["proposer_gen_batch"]
                                    solver_batch = round_state["solver_batch"]
                                    solver_gen_batch = round_state["solver_gen_batch"]
                                    if update_idx > 0:
                                        with marked_timer(f"solver_generation_{update_idx + 1}", timing_raw):
                                            solver_gen_batch = self._generate_solver_trajectories(
                                                solver_batch, timing_raw
                                            )
                                            step_metrics["self_play/solver_samples"] = len(solver_gen_batch.batch)
                                        with marked_timer(f"reward_calculation_{update_idx + 1}", timing_raw):
                                            proposer_rewards, solver_rewards = self._calculate_self_play_rewards(
                                                proposer_gen_batch, solver_gen_batch, timing_raw
                                            )
                                        self._attach_self_play_rewards(
                                            proposer_gen_batch,
                                            solver_gen_batch,
                                            proposer_rewards,
                                            solver_rewards,
                                            step_metrics,
                                        )
                                        if self.kg_writer is not None:
                                            self._write_correct_trajectories_to_kg(
                                                solver_gen_batch, solver_rewards
                                            )
                                        round_state["solver_gen_batch"] = solver_gen_batch
                                        round_state["proposer_rewards"] = proposer_rewards
                                        round_state["solver_rewards"] = solver_rewards

                                    solver_metrics = self._update_on_trajectories(
                                        solver_gen_batch, "solver", timing_raw
                                    )
                                    prefix = (
                                        "solver"
                                        if actual_solver_update_steps == 1
                                        else f"solver/update_{update_idx + 1}"
                                    )
                                    step_metrics.update({f"{prefix}/{k}": v for k, v in solver_metrics.items()})
                                    step_metrics["self_play/solver_update_round"] = update_idx + 1
                                representative_batch = solver_gen_batch

                            def _run_proposer_update():
                                nonlocal representative_batch, proposer_gen_batch
                                if not (self.sp_config.proposer.enable and proposer_gen_batch is not None):
                                    return
                                if len(proposer_update_batches) > 1:
                                    proposer_gen_batch = self._concat_unprepared_update_batches(
                                        proposer_update_batches,
                                        role="proposer",
                                    )
                                    step_metrics["self_play/proposer_accumulated_batches"] = len(
                                        proposer_update_batches
                                    )
                                    step_metrics["self_play/proposer_accumulated_samples"] = len(
                                        proposer_gen_batch.batch
                                    )
                                proposer_metrics = self._update_on_trajectories(
                                    proposer_gen_batch, "proposer", timing_raw
                                )
                                step_metrics.update({f"proposer/{k}": v for k, v in proposer_metrics.items()})
                                representative_batch = proposer_gen_batch

                            if update_order == "solver_then_proposer":
                                _run_solver_updates()
                                _run_proposer_update()
                            else:
                                _run_proposer_update()
                                _run_solver_updates()

                    print(f"Representative batch size: {len(representative_batch.batch)}")
                    all_metrics.update(step_metrics)

                    ray.get(self.sp_data_manager.update_global_step.remote(self.global_steps))

                    all_metrics.update(
                        {
                            "training/global_step": self.global_steps,
                            "training/epoch": epoch,
                        }
                    )

                    sp_stats = ray.get(self.sp_data_manager.get_statistics.remote())
                    for key, value in sp_stats.items():
                        if isinstance(value, (int, float)):
                            all_metrics[f"self_play/stats/{key}"] = value

                    if hasattr(solver_gen_batch, "meta_info") and solver_gen_batch.meta_info:
                        solver_metrics = solver_gen_batch.meta_info.get("metrics", {})
                        for key, value in solver_metrics.items():
                            if key.startswith("dynamic_sampling/") and isinstance(value, (int, float, bool)):
                                all_metrics[f"self_play/{key}"] = value
                            if key.startswith("reward_dynamic_sampling/") and isinstance(value, (int, float, bool)):
                                all_metrics[f"self_play/{key}"] = value

                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (
                            is_last_step
                            or self.global_steps % self.config.trainer.test_freq == 0
                            or self.sp_config.reward_dynamic_sampling.enable
                        )
                    ):
                        with marked_timer("testing", timing_raw):
                            val_metrics = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        all_metrics.update(val_metrics)

                    esi_close_to_expiration = should_save_ckpt_esi(
                        max_steps_duration=self.max_steps_duration,
                        redundant_time=self.config.trainer.esi_redundant_time,
                    )

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                        or esi_close_to_expiration
                        or self.sp_config.reward_dynamic_sampling.enable
                    ):

                        if esi_close_to_expiration:
                            print("Force saving checkpoint: ESI instance expiration approaching.")
                        with marked_timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()
                            

                    steps_duration = timing_raw["self_play_step"]
                    self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                    print(f"[sp_ray_trainer] compute data metrics")
                    all_metrics.update(
                        quarl_compute_data_metrics(
                            tokenizer=self.tokenizer, batch=representative_batch, use_critic=self.use_critic
                        )
                    )

                    n_gpus = self.resource_pool_manager.get_n_gpus()
                    adapted_timing_raw = timing_raw.copy()

                    if "self_play_step" in adapted_timing_raw and "step" not in adapted_timing_raw:
                        adapted_timing_raw["step"] = adapted_timing_raw["self_play_step"]

                    timer_mapping = {
                        "solver_generation": "gen",
                        "solver_ref": "ref",
                        "solver_values": "values",
                        "solver_adv": "adv",
                        "solver_update_critic": "update_critic",
                        "solver_update_actor": "update_actor",
                    }

                    for our_name, expected_name in timer_mapping.items():
                        if our_name in adapted_timing_raw and expected_name not in adapted_timing_raw:
                            adapted_timing_raw[expected_name] = adapted_timing_raw[our_name]

                    all_metrics.update(
                        compute_timing_metrics(batch=representative_batch, timing_raw=adapted_timing_raw)
                    )
                    all_metrics.update(
                        compute_throughout_metrics(
                            batch=representative_batch, timing_raw=adapted_timing_raw, n_gpus=n_gpus
                        )
                    )

                    # Log metrics
                    tracking_logger.log(data=all_metrics, step=self.global_steps)

                    progress_bar.update(1)
                    self.global_steps += 1

                    if self.sp_config.extraction_failure.strategy == "reuse":
                        if self.global_steps % self.sp_config.extraction_failure.pool_clear_interval == 0:

                            ray.get(
                                self.sp_data_manager.clear_solving_pool.remote(
                                    keep_ratio=self.sp_config.extraction_failure.keep_ratio
                                )
                            )

                            print(f"Cleared solving pool")


                    if is_last_step:
                        pprint(f"Final validation metrics: {last_val_metrics}")
                        progress_bar.close()
                        return

                    if reward_dynamic_sampling_enabled:
                        accumulated_proposer_batch = None
                        accumulated_solver_batch = None
                        num_prompt_in_batch = 0
                        num_gen_batches = 0

                except StopIteration:
                    break

    def _generate_proposer_with_dynamic_sampling(
        self, dataloader_iter, timing_raw: Dict
    ) -> Tuple[DataProto, Optional[DataProto], Dict[str, Any]]:

        # ----------------------------------------------------------------
        # Walk mode: bypass standard dataloader-based proposer generation
        # ----------------------------------------------------------------
        if self._proposer_mode == "walk":
            return self._walk_mode_step(timing_raw)

        # ----------------------------------------------------------------
        # Original seed mode (unchanged below)
        # ----------------------------------------------------------------
        dynamic_sampling_config = self.sp_config.get("dynamic_sampling", {})
        enable_dynamic_sampling = dynamic_sampling_config.get("enable", False)

        try:
            batch_dict = next(dataloader_iter)
        except StopIteration:
            return None, None, {}

        batch = DataProto.from_single_dict(batch_dict)

        if not enable_dynamic_sampling:
            proposer_gen_batch = self._generate_proposer_trajectories(batch, timing_raw)
            solver_batch = self._extract_and_assemble_solver_data(proposer_gen_batch, timing_raw)
            return proposer_gen_batch, solver_batch, {}

        max_retry_attempts = dynamic_sampling_config.get("max_retry_attempts", 3)
        min_valid_ratio = dynamic_sampling_config.get("min_valid_ratio", 1.0)
        target_batch_size = len(batch.batch)

        print(
            f"=== Dynamic Sampling Enabled: max_batches={max_retry_attempts + 1}, min_valid_ratio={min_valid_ratio}, target_size={target_batch_size} ==="
        )

        all_valid_problems = []
        all_proposer_batches = []
        sampling_metrics = {
            "dynamic_sampling/batches_used": 0,
            "dynamic_sampling/total_trajectories": 0,
            "dynamic_sampling/total_valid_problems": 0,
            "dynamic_sampling/final_valid_count": 0,
        }

        current_batch = batch
        batch_count = 0

        while batch_count <= max_retry_attempts:
            batch_count += 1
            sampling_metrics["dynamic_sampling/batches_used"] = batch_count

            print(f"Dynamic sampling batch {batch_count}/{max_retry_attempts + 1}")

            proposer_gen_batch = self._generate_proposer_trajectories(current_batch, timing_raw)

            trajectories = self._extract_trajectories_from_batch(proposer_gen_batch)
            extracted_problems = self._extract_and_process_problems(trajectories)

            valid_problems_this_batch = len(extracted_problems)
            total_trajectories_this_batch = len(trajectories)

            sampling_metrics["dynamic_sampling/total_trajectories"] += total_trajectories_this_batch
            sampling_metrics["dynamic_sampling/total_valid_problems"] += valid_problems_this_batch

            print(f"Batch {batch_count}: {valid_problems_this_batch}/{total_trajectories_this_batch} valid problems")

            if extracted_problems:
                valid_trajectory_indices = [problem["trajectory_index"] for problem in extracted_problems]
                valid_proposer_batch = self._filter_proposer_batch_by_indices(
                    proposer_gen_batch, valid_trajectory_indices
                )
                all_proposer_batches.append(valid_proposer_batch)

            all_valid_problems.extend(extracted_problems)

            current_valid_count = len(all_valid_problems)
            required_valid_count = int(target_batch_size * min_valid_ratio)

            if current_valid_count >= required_valid_count:
                print(
                    f"✅ Dynamic sampling succeeded! Collected {current_valid_count} valid problems (required: {required_valid_count})"
                )
                break

            if batch_count <= max_retry_attempts:
                try:
                    batch_dict = next(dataloader_iter)
                    current_batch = DataProto.from_single_dict(batch_dict)
                    print(
                        f"❌ Need more valid problems: {current_valid_count}/{required_valid_count}, using next batch..."
                    )
                except StopIteration:
                    print(f"❌ No more batches available, collected {current_valid_count} valid problems")
                    break
            else:
                print(f"❌ Max batches reached, collected {current_valid_count} valid problems")

        if len(all_valid_problems) == 0 or len(all_proposer_batches) == 0:
            print("❌ No valid problems found from any batch")
            return None, None, sampling_metrics

        final_proposer_batch = self._combine_proposer_batches(all_proposer_batches)

        final_batch_size = len(all_valid_problems)
        aligned_problems = self._align_problems_with_proposer_batch(all_valid_problems, final_batch_size)

        replicated_count = 0
        if final_batch_size < target_batch_size:
            print(f"🔄 Replicating {final_batch_size} valid problems to reach target size {target_batch_size}")

            needed_count = target_batch_size - final_batch_size
            replicated_count = needed_count

            import random

            replicated_problems = []
            replicated_proposer_trajectories = []

            for i in range(needed_count):
                source_idx = random.randint(0, final_batch_size - 1)

                replicated_problem = deepcopy(aligned_problems[source_idx])
                replicated_problem["trajectory_index"] = final_batch_size + i
                replicated_problems.append(replicated_problem)

                source_proposer_data = final_proposer_batch[source_idx : source_idx + 1]
                replicated_proposer_trajectories.append(source_proposer_data)

            aligned_problems.extend(replicated_problems)

            for replicated_traj in replicated_proposer_trajectories:
                final_proposer_batch = DataProto.concat([final_proposer_batch, replicated_traj])

            print(f"🔄 Replicated {replicated_count} problems, now have {len(aligned_problems)} total")

        else:
            if final_batch_size > target_batch_size:
                print(f"✂️ Truncating {final_batch_size} problems to target size {target_batch_size}")
                aligned_problems = aligned_problems[:target_batch_size]
                final_proposer_batch = final_proposer_batch[:target_batch_size]
                print(f"✂️ Truncated to {len(aligned_problems)} problems")
        self.generate_problem = aligned_problems

        solver_batch = self._prepare_solving_batch_from_data(aligned_problems)

        sampling_metrics["dynamic_sampling/original_valid_count"] = final_batch_size
        sampling_metrics["dynamic_sampling/final_valid_count"] = len(aligned_problems)
        sampling_metrics["dynamic_sampling/replicated_count"] = replicated_count

        if solver_batch is not None:
            solver_batch.meta_info.setdefault("metrics", {}).update(sampling_metrics)

        return final_proposer_batch, solver_batch, sampling_metrics

    def _apply_reward_dynamic_sampling(
        self, proposer_gen_batch: DataProto, solver_gen_batch: DataProto, timing_raw: Dict
    ) -> Tuple[DataProto, DataProto]:
        print("=== Applying Reward-based Dynamic Sampling ===")

        reward_config = self.sp_config.get("reward_dynamic_sampling", {})
        metric_name = reward_config.get("metric", "seq_final_reward")

        if metric_name == "seq_final_reward":
            metric_values = proposer_gen_batch.batch["token_level_scores"].sum(dim=-1).numpy()
        elif metric_name == "seq_reward":
            metric_values = proposer_gen_batch.batch["token_level_scores"].sum(dim=-1).numpy()
        else:
            raise ValueError(f"Unknown metric: {metric_name}")

        proposer_gen_batch.non_tensor_batch[metric_name] = metric_values

        uid2metric_vals = defaultdict(list)
        uid2traj_indices = defaultdict(list)

        for idx, (uid, metric_val) in enumerate(
            zip(proposer_gen_batch.non_tensor_batch["uid"], proposer_gen_batch.non_tensor_batch[metric_name])
        ):
            uid2metric_vals[uid].append(metric_val)
            uid2traj_indices[uid].append(idx)

        uid2variance = {}
        for uid, metric_vals in uid2metric_vals.items():
            uid2variance[uid] = np.std(metric_vals)

        kept_uids = [uid for uid, variance in uid2variance.items() if variance > 0 or len(uid2metric_vals[uid]) == 1]

        kept_traj_indices = []
        for uid in kept_uids:
            kept_traj_indices.extend(uid2traj_indices[uid])

        kept_traj_indices.sort()

        print(
            f"Reward filtering: {len(uid2metric_vals)} groups, {len(kept_uids)} kept, {len(kept_traj_indices)}/{len(proposer_gen_batch.batch)} trajectories"
        )

        filtered_proposer_batch = proposer_gen_batch[kept_traj_indices]

        solver_n = self.config.actor_rollout_ref.rollout.n
        kept_solver_indices = []

        for proposer_idx in kept_traj_indices:
            solver_start_idx = proposer_idx * solver_n
            solver_end_idx = solver_start_idx + solver_n
            kept_solver_indices.extend(range(solver_start_idx, solver_end_idx))

        filtered_solver_batch = solver_gen_batch[kept_solver_indices]

        print(f"Solver filtering: {len(kept_solver_indices)}/{len(solver_gen_batch.batch)} trajectories kept")

        filtering_metrics = {
            "reward_dynamic_sampling/original_proposer_groups": len(uid2metric_vals),
            "reward_dynamic_sampling/kept_proposer_groups": len(kept_uids),
            "reward_dynamic_sampling/original_proposer_trajectories": len(proposer_gen_batch.batch),
            "reward_dynamic_sampling/kept_proposer_trajectories": len(filtered_proposer_batch.batch),
            "reward_dynamic_sampling/original_solver_trajectories": len(solver_gen_batch.batch),
            "reward_dynamic_sampling/kept_solver_trajectories": len(filtered_solver_batch.batch),
            "reward_dynamic_sampling/filtering_ratio": (
                len(kept_traj_indices) / len(proposer_gen_batch.batch) if len(proposer_gen_batch.batch) > 0 else 0.0
            ),
        }

        pprint(filtering_metrics)

        return filtered_proposer_batch, filtered_solver_batch

    def _combine_proposer_batches(self, proposer_batches: List[DataProto]) -> DataProto:
        if len(proposer_batches) == 1:
            return proposer_batches[0]

        combined_batch = proposer_batches[0]
        for batch in proposer_batches[1:]:
            combined_batch = DataProto.concat([combined_batch, batch])

        return combined_batch

    def _filter_proposer_batch_by_indices(self, proposer_batch: DataProto, valid_indices: List[int]) -> DataProto:
        if not valid_indices:
            return None

        filtered_batch = proposer_batch[valid_indices]

        return filtered_batch

    def _align_problems_with_proposer_batch(
        self, all_valid_problems: List[Dict[str, Any]], target_size: int
    ) -> List[Dict[str, Any]]:
        aligned_problems = []

        for i, problem in enumerate(all_valid_problems[:target_size]):
            aligned_problem = problem.copy()
            aligned_problem["trajectory_index"] = i
            aligned_problems.append(aligned_problem)

        return aligned_problems

    # =========================================================================
    # Walk mode methods
    # =========================================================================

    def _walk_mode_step(
        self, timing_raw: Dict
    ) -> Tuple[DataProto, Optional[DataProto], Dict[str, Any]]:
        """
        Walk-mode proposer step with per-chain multi-candidate selection.

          1. Sample K=batch_size chains.
          2. For each chain, generate M=proposer.n candidate prompts (K*M total).
          3. For each chain group of M:
               a. Format-check all M.
               b. API quality-score the format-valid ones.
               c. Select the best candidate using the quality score.
               d. If all M fail format → use dummy problem.
          4. Build proposer_gen_batch from the K selected trajectories only.
          5. Store avg_format_rewards per chain for reward computation.
          6. Solver batch = K problems (one per chain).
        """
        batch_size = self.config.data.train_batch_size
        candidates_per_chain = int(self.sp_config.proposer.get("n", 1))
        num_chains = batch_size  # K chains = batch_size

        print(
            f"=== Walk Mode Step (step {self.global_steps}): "
            f"K={num_chains} chains × M={candidates_per_chain} candidates ==="
        )

        # ── Step 1 & 2: Generate K*M trajectories ────────────────────────────
        full_gen_batch, chain_ids = self._generate_proposer_trajectories_walk(
            num_chains, candidates_per_chain, timing_raw
        )

        # ── Step 2b: Parse <select_entity>/<select_hops> and patch ground truth ──
        # select_entity (new): 0-indexed position in chain
        # select_hops  (legacy): hop count (index = n-1)
        trajectories = self._extract_trajectories_from_batch(full_gen_batch)
        selected_idx_list = []   # entity index (0-based)
        for traj in trajectories:
            extra_info = traj.get("metadata", {}).get("extra_info", {})
            relations = extra_info.get("chain_relations", [])
            if relations:
                entity_idx, gt = extract_selected_entity(traj.get("output", ""), relations)
                traj["metadata"]["selected_entity_idx"] = entity_idx
                traj["metadata"]["selected_hops"] = entity_idx + 1   # legacy compat
                traj["metadata"]["dynamic_ground_truth"] = gt
                reward_model = traj["metadata"].get("reward_model", {})
                if isinstance(reward_model, dict) and "ground_truth" in reward_model:
                    reward_model["ground_truth"]["target"] = gt
                selected_idx_list.append(entity_idx)

        if selected_idx_list:
            avg_idx = sum(selected_idx_list) / len(selected_idx_list)
            dist = sorted(set(selected_idx_list))
            print(f"Entity selection: avg_idx={avg_idx:.1f}, dist={dist}")

        # ── Step 3: Per-chain selection (format → API → best) ────────────────
        (
            selected_batch_indices,
            selected_problems,
            avg_format_rewards,
            selected_map,
        ) = self._select_best_per_chain_with_quality(
            full_gen_batch, trajectories, chain_ids, num_chains, candidates_per_chain
        )

        # ── Step 4: Build proposer_gen_batch from K selected trajectories ─────
        # This is the batch that goes through the model update (K entries, not K*M).
        proposer_gen_batch = full_gen_batch[selected_batch_indices]

        # ── Step 5: Store state for reward computation ────────────────────────
        self._walk_selected_map = selected_map          # {chain_idx → solver_pos} (identity)
        self._walk_generate_n_last = num_chains          # K (for num_repeat in compute_advantage)
        self._chain_avg_format_rewards = avg_format_rewards  # K floats

        # Mark which chains succeeded so _compute_proposer_rewards applies format penalty
        self.current_extraction_success_mask = [
            not (p.get("extra_info", {}) or {}).get("extraction_failed", False)
            for p in selected_problems
        ]
        self._add_feedback_pool_problems(selected_problems)

        # ── Step 6: Inject process-consistency metadata and prepare solver batch ──
        self._inject_process_consistency_info(selected_problems)
        solver_batch = self._prepare_solving_batch_from_data(selected_problems)

        failed_count = sum(1 for p in selected_problems
                           if (p.get("extra_info", {}) or {}).get("extraction_failed", False))
        valid_count = len(selected_problems) - failed_count
        sampling_metrics = {
            "walk/num_chains": num_chains,
            "walk/candidates_per_chain": candidates_per_chain,
            "walk/valid_chains": valid_count,
            "walk/dummy_chains": failed_count,
            "walk/avg_format_reward": sum(avg_format_rewards) / len(avg_format_rewards),
            "walk/avg_selected_entity_idx": (
                sum(selected_idx_list) / len(selected_idx_list) if selected_idx_list else 0.0
            ),
            "walk/avg_selected_hops": (   # legacy compat
                (sum(selected_idx_list) / len(selected_idx_list) + 1) if selected_idx_list else 0.0
            ),
        }
        return proposer_gen_batch, solver_batch, sampling_metrics

    def _generate_proposer_trajectories_walk(
        self, num_chains: int, candidates_per_chain: int, timing_raw: Dict
    ) -> Tuple[DataProto, List[int]]:
        """
        Sample num_chains knowledge chains and generate candidates_per_chain
        proposer trajectories per chain.

        Returns:
            gen_batch      : DataProto of shape (num_chains * candidates_per_chain,)
            chain_ids      : list of length (num_chains * candidates_per_chain,)
                             chain_ids[i] = which chain trajectory i belongs to
        """
        total = num_chains * candidates_per_chain
        print(f"=== Walk Proposer Generation: {num_chains} chains × {candidates_per_chain} candidates = {total} ===")

        ray.get(self.sp_data_manager.switch_to_phase.remote(SelfPlayPhase.PROBLEM_GENERATION))

        chains = self.chain_pool.sample(num_chains, seed_entities=self._walk_seed_entities or None)
        proposer_data_one_per_chain = self.chain_pool.build_proposer_data_batch(chains)

        if len(proposer_data_one_per_chain) < num_chains:
            import random as _rnd
            logger.warning(
                f"Chain pool only produced {len(proposer_data_one_per_chain)} valid records "
                f"(requested {num_chains}). Padding by repeating."
            )
            while len(proposer_data_one_per_chain) < num_chains:
                proposer_data_one_per_chain.append(_rnd.choice(proposer_data_one_per_chain))

        proposer_data_one_per_chain = proposer_data_one_per_chain[:num_chains]

        # Replicate each chain record M times to get num_chains * M entries.
        # trajectory_index is assigned sequentially across the full K*M batch.
        from copy import deepcopy as _deepcopy
        proposer_data: List[Dict] = []
        chain_ids: List[int] = []
        for chain_idx, record in enumerate(proposer_data_one_per_chain):
            for _ in range(candidates_per_chain):
                proposer_data.append(_deepcopy(record))
                chain_ids.append(chain_idx)
        # Re-assign trajectory_index to be globally sequential (0 .. total-1)
        for traj_idx, record in enumerate(proposer_data):
            if "extra_info" in record and isinstance(record["extra_info"], dict):
                record["extra_info"]["trajectory_index"] = traj_idx

        # Tokenize into a DataProto (reuse _prepare_solving_batch_from_data)
        gen_batch = self._prepare_solving_batch_from_data(proposer_data)

        # Pop tensor keys to prepare for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
        for key in ["multi_modal_data", "raw_prompt", "tools_kwargs",
                    "interaction_kwargs", "index", "agent_name"]:
            if key in gen_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append(key)

        gen_batch_input = gen_batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )

        gen_batch_input.meta_info["global_steps"] = self.global_steps
        gen_batch_input.meta_info["phase"] = "proposer_walk"
        gen_batch_input.meta_info["do_sample"] = self.sp_config.get("proposer", {}).get("do_sample", True)
        gen_batch_input.meta_info["temperature"] = self.sp_config.get("proposer", {}).get("temperature", 0.8)

        if not self.async_rollout_mode:
            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_input)
        else:
            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_input)

        timing_raw.update(gen_batch_output.meta_info.get("timing", {}))
        gen_batch_output.meta_info.pop("timing", None)

        # Assign UIDs (one per trajectory — no repeats in walk mode)
        gen_batch.non_tensor_batch["uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(len(gen_batch.batch))], dtype=object
        )
        gen_batch = gen_batch.union(gen_batch_output)

        for key in gen_batch.batch.keys():
            if key not in ["old_log_probs", "ref_log_prob"]:
                gen_batch.batch[key] = gen_batch.batch[key].long()

        if "response_mask" not in gen_batch.batch.keys():
            gen_batch.batch["response_mask"] = compute_response_mask(gen_batch)

        print(f"Walk proposer generated {len(gen_batch.batch)} trajectories")
        return gen_batch, chain_ids

    def _select_walk_problems(
        self,
        extracted_problems: List[Dict[str, Any]],
        batch_size: int,
    ) -> Tuple[List[Dict[str, Any]], Dict[int, int], int]:
        """
        Select exactly batch_size solver problems from walk-extracted questions,
        padding with seed data when fewer than batch_size are valid.

        Returns:
            selected_problems  - list of batch_size problem dicts
            selected_map       - {proposer_traj_idx: solver_position}
            seed_pad_count     - how many seed problems were added
        """
        # Reset extraction success mask for walk mode
        self.current_extraction_success_mask = [False] * self._walk_generate_n_last

        # Take up to batch_size valid walk problems
        walk_selected = extracted_problems[:batch_size]
        selected_map: Dict[int, int] = {}

        for solver_pos, problem in enumerate(walk_selected):
            traj_idx = problem.get("trajectory_index", -1)
            if 0 <= traj_idx < self._walk_generate_n_last:
                selected_map[traj_idx] = solver_pos
                self.current_extraction_success_mask[traj_idx] = True

        # Pad with seed problems if needed
        seed_pad_count = max(0, batch_size - len(walk_selected))
        seed_problems: List[Dict[str, Any]] = []

        if seed_pad_count > 0:
            seed_problems = self._sample_seed_problems(
                seed_pad_count,
                start_traj_idx=len(walk_selected),
            )
            if len(seed_problems) < seed_pad_count:
                logger.warning(
                    f"Could only get {len(seed_problems)} seed problems "
                    f"(needed {seed_pad_count})"
                )

        selected_problems = walk_selected + seed_problems

        # Re-index trajectory_index to be sequential 0..len-1
        for i, prob in enumerate(selected_problems):
            prob["trajectory_index"] = i

        return selected_problems, selected_map, len(seed_problems)

    def _sample_seed_problems(
        self, count: int, start_traj_idx: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Directly convert seed data records into solver problems (no proposer run).
        Used as fallback padding in walk mode.
        """
        seed_pool = ray.get(self.sp_data_manager.get_current_phase_data.remote())
        if not seed_pool:
            logger.warning("Seed pool is empty; cannot pad walk problems with seeds.")
            return []

        import random as _random
        sampled = (
            _random.sample(seed_pool, min(count, len(seed_pool)))
            if len(seed_pool) >= count
            else [_random.choice(seed_pool) for _ in range(count)]
        )

        problems = []
        for i, seed_record in enumerate(sampled):
            prob = build_seed_solver_problem(
                seed_record=seed_record,
                trajectory_index=start_traj_idx + i,
                solver_system_prompt=self.problem_extractor.solver_system_prompt,
                solver_user_prompt_template=self.problem_extractor.solver_user_prompt,
            )
            if prob is not None:
                problems.append(prob)

        return problems

    # =========================================================================
    # Per-chain candidate selection with format check + API quality scoring
    # =========================================================================

    def _select_best_per_chain_with_quality(
        self,
        full_gen_batch: DataProto,
        trajectories: List[Dict[str, Any]],
        chain_ids: List[int],
        num_chains: int,
        candidates_per_chain: int,
    ) -> Tuple[List[int], List[Dict[str, Any]], List[float], Dict[int, int]]:
        """
        For each of the num_chains chains:
          1. Format-check all M candidates.
          2. For format-valid candidates, call quality API concurrently.
          3. Select the candidate with highest quality score.
             - If quality >= threshold: keep it as the selected candidate.
             - If all M fail format: use dummy problem.
          4. Compute avg_format_reward across all M candidates for this chain.

        Returns:
            selected_batch_indices : list[int] — indices into full_gen_batch (length = num_chains)
            selected_problems      : list[dict] — solver problem for each chain (length = num_chains)
            avg_format_rewards     : list[float] — per-chain average format reward (length = num_chains)
            selected_map           : dict{chain_idx → solver_pos} (chain_idx == solver_pos, identity)
        """
        format_bonus   = self.sp_config.proposer.get("format_bonus",   0.1)
        format_penalty = self.sp_config.proposer.get("format_penalty", -0.1)

        # Build per-chain groups: chain_idx → list of (global_traj_idx, traj_dict)
        chain_groups: Dict[int, List[Tuple[int, Dict]]] = {k: [] for k in range(num_chains)}
        for global_idx, traj in enumerate(trajectories):
            chain_idx = chain_ids[global_idx]
            chain_groups[chain_idx].append((global_idx, traj))

        selected_batch_indices: List[int] = []
        selected_problems: List[Dict[str, Any]] = []
        avg_format_rewards: List[float] = []
        selected_map: Dict[int, int] = {}

        _NODE_RE = re.compile(r"\[([^\]]+)\]:\s*(.+?)(?=\n\[|\Z)", re.DOTALL)

        # ── Pass 1: format check (sequential — uses shared extraction pipeline) ──
        # Collect per-chain format results without calling the quality API yet.
        per_chain_format: List[Dict] = []   # one entry per chain_idx
        for chain_idx in range(num_chains):
            candidates = chain_groups[chain_idx]
            format_valid: List[Tuple[int, Dict, Dict]] = []
            per_candidate_fmt_rewards: List[float] = []

            for global_idx, traj in candidates:
                problems_extracted, _ = extract_problems_batch([traj], self.problem_extractor)
                if problems_extracted and self._validate_extracted_problem(problems_extracted[0]):
                    prob = problems_extracted[0]
                    meta_extra = traj.get("metadata", {}).get("extra_info", {})
                    relations = meta_extra.get("chain_relations", [])
                    # Use entity_idx (new) with fallback to selected_hops-1 (legacy)
                    entity_idx = traj.get("metadata", {}).get(
                        "selected_entity_idx",
                        max(0, traj.get("metadata", {}).get("selected_hops", len(relations)) - 1)
                    )
                    entity_idx = min(entity_idx, len(relations) - 1) if relations else 0
                    prob.setdefault("extra_info", {})
                    prob["extra_info"]["intermediate_entities"] = relations[:entity_idx]
                    prob["extra_info"]["selected_entity_idx"] = entity_idx
                    prob["extra_info"]["selected_hops"] = entity_idx + 1  # legacy compat
                    format_valid.append((global_idx, traj, prob))
                    per_candidate_fmt_rewards.append(format_bonus)
                else:
                    per_candidate_fmt_rewards.append(format_penalty)

            avg_fmt = sum(per_candidate_fmt_rewards) / len(per_candidate_fmt_rewards)
            avg_format_rewards.append(avg_fmt)
            per_chain_format.append({"format_valid": format_valid, "avg_fmt": avg_fmt,
                                     "candidates": candidates})

        # ── Pass 2: quality check (concurrent across all chains — one API call per chain) ──
        # Build quality-check tasks for chains that have ≥1 format-valid candidate.
        # All M candidates of the same chain share chain context → single API call.
        import concurrent.futures as _cf
        import random as _rand_sel

        def _quality_task(chain_idx: int, format_valid_items) -> List[float]:
            """RAG-verify candidates for one chain, respecting per-candidate GT."""
            _, first_traj, _ = format_valid_items[0]
            node_docs: dict = {}

            # Prefer chain evidence from metadata. Search-mode proposer prompts
            # only include entity names, so prompt parsing would otherwise leave
            # RAG verification without documents.
            extra_info = first_traj.get("metadata", {}).get("extra_info", {})
            if isinstance(extra_info, dict):
                raw_nodes = extra_info.get("chain_nodes", {})
                if isinstance(raw_nodes, dict):
                    node_docs = {
                        str(entity): str(text).replace("\n", " ").strip()
                        for entity, text in raw_nodes.items()
                        if text
                    }

            # Backward compatibility for older data where node docs were embedded
            # in the user prompt as [Entity]: text lines.
            turns = first_traj.get("turns", [])
            if not node_docs and turns:
                for m in _NODE_RE.finditer(turns[0].get("content", "")):
                    entity = m.group(1).strip()
                    # Preserve full text (not just snippet) for RAG context
                    text = m.group(2).replace("\n", " ").strip()
                    node_docs[entity] = text

            grouped: Dict[str, List[Tuple[int, str]]] = {}
            for local_idx, (_, _, prob) in enumerate(format_valid_items):
                ground_truth = prob.get("reward_model", {}).get("ground_truth", {})
                if isinstance(ground_truth, dict):
                    gt_entity = ground_truth.get("target", "") or ""
                else:
                    gt_entity = str(ground_truth or "")
                grouped.setdefault(gt_entity, []).append(
                    (local_idx, prob.get("extracted_question", ""))
                )

            scores: List[Optional[float]] = [None] * len(format_valid_items)
            for gt_entity, items in grouped.items():
                questions = [question for _, question in items]
                group_scores = self.quality_checker.rag_verify_candidates(
                    questions,
                    ground_truth=gt_entity,
                    node_docs=node_docs if node_docs else None,
                )
                if group_scores is None:
                    # API unavailable: treat all candidates in this group as failed; do not auto-accept.
                    group_scores = [0.0] * len(questions)
                for (local_idx, _), score in zip(items, group_scores):
                    scores[local_idx] = score

            return [float(score) if score is not None else 0.0 for score in scores]

        # Submit all chains concurrently
        chain_quality_scores: List[Optional[List[float]]] = [None] * num_chains
        if self.quality_checker is not None:
            with _cf.ThreadPoolExecutor(max_workers=self._quality_check_max_workers) as _pool:
                fut_map = {
                    _pool.submit(_quality_task, cidx, per_chain_format[cidx]["format_valid"]): cidx
                    for cidx in range(num_chains)
                    if per_chain_format[cidx]["format_valid"]
                }
                for fut in _cf.as_completed(fut_map):
                    cidx = fut_map[fut]
                    try:
                        chain_quality_scores[cidx] = fut.result()
                    except Exception as e:
                        from coevokg.utils.api_provider_pool import FatalProviderPoolExhausted
                        if isinstance(e, FatalProviderPoolExhausted):
                            raise
                        logger.warning(f"Quality check failed for chain {cidx}: {e}")
                        chain_quality_scores[cidx] = None

        # ── Pass 3: top-k selection per chain ─────────────────────────────────
        selection_top_k = int(self.sp_config.proposer.get("selection_top_k", 3))
        # Apply threshold filtering only after a successful API call (None=API failure, 0.0=low quality).
        min_quality_score = float(self.sp_config.proposer.get("min_quality_score", 0.0))

        for chain_idx in range(num_chains):
            entry = per_chain_format[chain_idx]
            format_valid = entry["format_valid"]
            avg_fmt = entry["avg_fmt"]
            candidates = entry["candidates"]

            if format_valid:
                api_scores = chain_quality_scores[chain_idx]  # None or List[float]
                if self.quality_checker is not None and api_scores is not None:
                    quality_scores = api_scores
                else:
                    quality_scores = [0.0] * len(format_valid)  # API unavailable: validation failed, use dummy fallback.

                sorted_by_quality = sorted(
                    range(len(format_valid)),
                    key=lambda i: quality_scores[i],
                    reverse=True,
                )
                top_k_indices = sorted_by_quality[:min(selection_top_k, len(sorted_by_quality))]
                best_local_idx = _rand_sel.choice(top_k_indices)
                best_global_idx, best_traj, best_problem = format_valid[best_local_idx]
                best_quality = quality_scores[best_local_idx] if quality_scores else None

                # Threshold filtering: valid API score below threshold triggers seed fallback.
                if quality_scores is not None and best_quality < min_quality_score:
                    dummy_global_idx = candidates[0][0]
                    dummy_traj = candidates[0][1]
                    dummy_problem, _ = self._create_fallback_problem_for_failed_extraction(
                        dummy_traj, trajectory_index=chain_idx
                    )
                    selected_batch_indices.append(dummy_global_idx)
                    selected_problems.append(dummy_problem)
                    selected_map[chain_idx] = chain_idx
                    print(
                        f"  Chain {chain_idx}: quality={best_quality:.2f} < {min_quality_score} → seed fallback"
                    )
                else:
                    selected_batch_indices.append(best_global_idx)
                    selected_problems.append(best_problem)
                    selected_map[chain_idx] = chain_idx
                    q_str = f"{best_quality:.2f}" if best_quality is not None else "N/A(API fail)"
                    print(
                        f"  Chain {chain_idx}: {len(format_valid)}/{candidates_per_chain} valid, "
                        f"selected traj {best_global_idx} (quality={q_str}), "
                        f"avg_fmt={avg_fmt:.2f}"
                    )

            else:
                dummy_global_idx = candidates[0][0]
                dummy_traj = candidates[0][1]
                dummy_problem, _ = self._create_fallback_problem_for_failed_extraction(
                    dummy_traj, trajectory_index=chain_idx
                )
                selected_batch_indices.append(dummy_global_idx)
                selected_problems.append(dummy_problem)
                selected_map[chain_idx] = chain_idx
                print(
                    f"  Chain {chain_idx}: 0/{candidates_per_chain} valid → seed fallback "
                    f"(avg_fmt={avg_fmt:.2f})"
                )

        print(
            f"Per-chain selection complete: "
            f"{sum(1 for p in selected_problems if p.get('data_source') != 'dummy')}/{num_chains} valid, "
            f"avg_format_reward={sum(avg_format_rewards)/len(avg_format_rewards):.3f}"
        )
        return selected_batch_indices, selected_problems, avg_format_rewards, selected_map

    # =========================================================================
    # Original seed-mode proposer generation (unchanged)
    # =========================================================================

    def _generate_proposer_trajectories(self, batch: DataProto, timing_raw: Dict) -> DataProto:
        print(f"=== Step 1: Proposer Generation at step {self.global_steps} ===")

        ray.get(self.sp_data_manager.switch_to_phase.remote(SelfPlayPhase.PROBLEM_GENERATION))

        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
        if "multi_modal_data" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("multi_modal_data")
        if "raw_prompt" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("raw_prompt")
        if "tools_kwargs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("tools_kwargs")
        if "interaction_kwargs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("interaction_kwargs")
        if "index" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("index")
        if "agent_name" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("agent_name")

        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )

        gen_batch.meta_info["global_steps"] = self.global_steps
        gen_batch.meta_info["phase"] = "proposer"
        gen_batch.meta_info["do_sample"] = self.sp_config.get("proposer", {}).get("do_sample", True)
        gen_batch.meta_info["temperature"] = self.sp_config.get("proposer", {}).get("temperature", 0.8)

        self.rollout_n = self.config.actor_rollout_ref.rollout.n
        proposer_n = self.sp_config.get("proposer", {}).get("n", self.rollout_n)

        if SELF_PLAY_DEBUG:
            logger.debug(f"Proposer gen_batch size before repeat: {len(gen_batch)}, proposer_n: {proposer_n}")

        gen_batch = gen_batch.repeat(repeat_times=proposer_n, interleave=True)

        if SELF_PLAY_DEBUG:
            logger.debug(f"Proposer gen_batch size after repeat: {len(gen_batch)}")

        if not self.async_rollout_mode:
            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
        else:
            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)

        timing_raw.update(gen_batch_output.meta_info.get("timing", {}))
        gen_batch_output.meta_info.pop("timing", None)

        batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)

        batch = batch.repeat(repeat_times=proposer_n, interleave=True)

        batch = batch.union(gen_batch_output)

        for key in batch.batch.keys():
            if key not in ["old_log_probs", "ref_log_prob"]:
                batch.batch[key] = batch.batch[key].long()

        if "response_mask" not in batch.batch.keys():
            batch.batch["response_mask"] = compute_response_mask(batch)

        print(f"Generated {len(batch.batch)} proposer trajectories")

        return batch

    def _extract_and_assemble_solver_data(self, proposer_gen_batch: DataProto, timing_raw: Dict) -> Optional[DataProto]:
        print(f"=== Step 2: Problem Extraction and Assembly at step {self.global_steps} ===")

        trajectories = self._extract_trajectories_from_batch(proposer_gen_batch)
        batch_size = len(trajectories)

        extracted_problems = self._extract_and_process_problems(trajectories)

        extraction_success_mask = [False] * batch_size

        problem_to_trajectory_map = {}
        for problem in extracted_problems:
            traj_idx = problem.get("trajectory_index", -1)
            if 0 <= traj_idx < batch_size:
                extraction_success_mask[traj_idx] = True
                problem_to_trajectory_map[traj_idx] = problem

        solving_data = []
        failed_extraction_count = 0
        reused_problem_count = 0
        metrics = {}

        for i in range(batch_size):
            if i in problem_to_trajectory_map:
                solving_data.append(problem_to_trajectory_map[i])
            else:
                failed_extraction_count += 1
                fallback_problem, reused = self._create_fallback_problem_for_failed_extraction(trajectories[i], i)
                solving_data.append(fallback_problem)
                if reused:
                    reused_problem_count += 1

        print(
            f"Failed extractions: {failed_extraction_count} (reused {reused_problem_count} existing problems, created {failed_extraction_count - reused_problem_count} dummy problems)"
        )
        metrics["self_play/reused_problem_count"] = reused_problem_count
        metrics["self_play/dummy_problem_count"] = failed_extraction_count - reused_problem_count

        print(f"Extracted {len(extracted_problems)} valid problems, padded to {len(solving_data)} total")
        print(
            f"Extraction success rate: {sum(extraction_success_mask)}/{batch_size} = {sum(extraction_success_mask) / batch_size:.2%}"
        )
        metrics["self_play/extraction_success_rate"] = sum(extraction_success_mask) / batch_size

        self.current_extraction_success_mask = extraction_success_mask

        self._inject_process_consistency_info(solving_data)
        solver_batch = self._prepare_solving_batch_from_data(solving_data)

        self.generate_problem = solving_data

        solver_batch.meta_info.setdefault("metrics", {}).update(metrics)

        return solver_batch

    def _generate_solver_trajectories(self, solver_batch: DataProto, timing_raw: Dict) -> DataProto:
        print(f"=== Step 3: Solver Generation at step {self.global_steps} ===")

        if SELF_PLAY_DEBUG:
            logger.debug(f"Input solver_batch size: {len(solver_batch)}")

        # DataProto.pop mutates the object. Keep the extracted question batch reusable
        # across solver multi-step updates.
        solver_batch = solver_batch[:]

        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
        if "multi_modal_data" in solver_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("multi_modal_data")
        if "raw_prompt" in solver_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("raw_prompt")
        if "tools_kwargs" in solver_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("tools_kwargs")
        if "interaction_kwargs" in solver_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("interaction_kwargs")
        if "index" in solver_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("index")
        if "agent_name" in solver_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("agent_name")

        gen_batch = solver_batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )

        gen_batch.meta_info["global_steps"] = self.global_steps
        gen_batch.meta_info["phase"] = "solver"
        gen_batch.meta_info["do_sample"] = self.sp_config.get("solver", {}).get("do_sample", True)
        gen_batch.meta_info["temperature"] = self.sp_config.get("solver", {}).get("temperature", 0.7)

        self.rollout_n = self.config.actor_rollout_ref.rollout.n

        if SELF_PLAY_DEBUG:
            logger.debug(f"gen_batch size before repeat: {len(gen_batch)}, rollout_n: {self.rollout_n}")

        gen_batch = gen_batch.repeat(repeat_times=self.rollout_n, interleave=True)

        if SELF_PLAY_DEBUG:
            logger.debug(f"gen_batch size after repeat: {len(gen_batch)}")

        if not self.async_rollout_mode:
            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
        else:
            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)

        timing_raw.update(gen_batch_output.meta_info.get("timing", {}))
        gen_batch_output.meta_info.pop("timing", None)

        solver_batch.non_tensor_batch["uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(len(solver_batch.batch))], dtype=object
        )
        solver_batch = solver_batch.repeat(repeat_times=self.rollout_n, interleave=True)
        solver_batch = solver_batch.union(gen_batch_output)

        for key in solver_batch.batch.keys():
            if key not in ["old_log_probs", "ref_log_prob"]:
                solver_batch.batch[key] = solver_batch.batch[key].long()

        if "response_mask" not in solver_batch.batch.keys():
            solver_batch.batch["response_mask"] = compute_response_mask(solver_batch)

        print(f"Generated {len(solver_batch.batch)} solver trajectories")

        return solver_batch

    def _calculate_self_play_rewards(
        self, proposer_gen_batch: DataProto, solver_gen_batch: DataProto, timing_raw: Dict,
        *, _return_future: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        print(f"=== Step 4: Reward Calculation at step {self.global_steps} ===")

        with marked_timer("solver_reward_computation", timing_raw, color="yellow"):
            if self.config.reward_model.launch_reward_fn_async:
                future_solver_reward = compute_reward_async.remote(data=solver_gen_batch, reward_fn=self.reward_fn)
                if _return_future:
                    # Caller will resolve via _finalize_self_play_rewards after doing GPU work.
                    return future_solver_reward
                solver_rewards, solver_extra_infos = ray.get(future_solver_reward)
            else:
                solver_rewards, solver_extra_infos = compute_reward(solver_gen_batch, self.reward_fn)

        # Stash the reward-stage KG chain payloads (one JSON string per trajectory,
        # indexed like the batch) so write-back reuses the already-extracted chain
        # instead of calling the LLM a second time.
        self._solver_kg_payloads = (solver_extra_infos or {}).get("coevokg_score_kg")

        with marked_timer("proposer_reward_computation", timing_raw, color="orange"):
            proposer_rewards = self._compute_proposer_rewards(proposer_gen_batch, solver_rewards, solver_extra_infos)

        print(f"Computed rewards: proposer_mean={proposer_rewards.mean():.3f}, solver_mean={solver_rewards.mean():.3f}")

        return proposer_rewards, solver_rewards

    def _finalize_self_play_rewards(
        self, future_solver_reward, proposer_gen_batch: DataProto, solver_gen_batch: DataProto, timing_raw: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Block on an async reward future and compute proposer rewards.

        Used with the overlap pattern where the caller launched the reward remote task
        via _calculate_self_play_rewards(_return_future=True) and has since done useful
        GPU work (log-prob prefill).  The solver_reward timer here measures only the
        residual wait after that GPU work; the proposer timer is unchanged.
        """
        with marked_timer("solver_reward_computation", timing_raw, color="yellow"):
            solver_rewards, solver_extra_infos = ray.get(future_solver_reward)

        # Stash reward-stage KG chain payloads for write-back reuse (see _calculate_self_play_rewards).
        self._solver_kg_payloads = (solver_extra_infos or {}).get("coevokg_score_kg")

        with marked_timer("proposer_reward_computation", timing_raw, color="orange"):
            proposer_rewards = self._compute_proposer_rewards(proposer_gen_batch, solver_rewards, solver_extra_infos)

        print(f"Computed rewards: proposer_mean={proposer_rewards.mean():.3f}, solver_mean={solver_rewards.mean():.3f}")
        return proposer_rewards, solver_rewards

    def _attach_self_play_rewards(
        self,
        proposer_gen_batch: DataProto,
        solver_gen_batch: DataProto,
        proposer_rewards: torch.Tensor,
        solver_rewards: torch.Tensor,
        step_metrics: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        proposer_gen_batch.batch["token_level_scores"] = proposer_rewards
        solver_gen_batch.batch["solver_raw_scores"] = solver_rewards

        from coevokg.reward.score.coevokg_score import _compute_solver_format_reward_hard as _sfr

        fmt_rewards = torch.zeros_like(solver_rewards)
        if "response_mask" in solver_gen_batch.batch and hasattr(self, "tokenizer"):
            resp_ids = solver_gen_batch.batch["responses"]
            resp_masks = solver_gen_batch.batch["response_mask"]
            fmt_vals: List[float] = []
            for i in range(len(resp_ids)):
                valid_len = int(resp_masks[i].sum().item())
                text = self.tokenizer.decode(resp_ids[i, :valid_len], skip_special_tokens=True)
                fmt_reward = _sfr(text)
                last_pos = max(0, valid_len - 1)
                fmt_rewards[i, last_pos] = fmt_reward
                fmt_vals.append(fmt_reward)
            nonzero = [v for v in fmt_vals if v != 0]
            step_metrics["solver/format_reward_mean"] = (
                float(sum(nonzero) / len(nonzero)) if nonzero else 0.0
            )
            step_metrics["solver/format_good_rate"] = (
                sum(1 for v in fmt_vals if v > 0) / max(len(fmt_vals), 1)
            )

        solver_gen_batch.batch["solver_format_rewards"] = fmt_rewards
        solver_format_coef = self._get_solver_format_coef()
        solver_train_rewards = solver_rewards + solver_format_coef * fmt_rewards.to(solver_rewards.device)
        solver_gen_batch.batch["token_level_scores"] = solver_train_rewards

        proposer_seq_rewards = proposer_rewards.sum(-1)
        solver_seq_raw_rewards = solver_rewards.sum(-1)
        solver_seq_train_rewards = solver_train_rewards.sum(-1)
        correct_count = int((solver_seq_raw_rewards >= 1.0).sum().item())
        searched_count = 0
        if hasattr(solver_gen_batch, "non_tensor_batch"):
            num_searches = solver_gen_batch.non_tensor_batch.get("num_searches", None)
            if num_searches is not None:
                searched_count = int(sum(1 for v in num_searches if (v or 0) > 0))

        step_metrics["rewards/proposer_mean"] = float(proposer_seq_rewards.mean().item())
        step_metrics["rewards/solver_mean"] = float(solver_seq_train_rewards.mean().item())
        step_metrics["rewards/solver_max"] = float(solver_seq_train_rewards.max().item())
        step_metrics["rewards/proposer_token_mean"] = float(proposer_rewards.mean().item())
        step_metrics["rewards/solver_token_mean"] = float(solver_train_rewards.mean().item())
        step_metrics["solver/raw_reward_mean"] = float(solver_seq_raw_rewards.mean().item())
        step_metrics["solver/train_reward_mean"] = float(solver_seq_train_rewards.mean().item())
        step_metrics["solver/format_coef"] = float(solver_format_coef)
        step_metrics["solver/format_reward_contribution"] = float(
            (solver_format_coef * fmt_rewards.sum(-1)).mean().item()
        )
        step_metrics["solver/correct_rate"] = correct_count / max(len(solver_seq_raw_rewards), 1)
        step_metrics["solver/search_rate"] = searched_count / max(len(solver_seq_raw_rewards), 1)
        step_metrics["walk/dummy_rate"] = (
            step_metrics.get("walk/dummy_chains", 0) / max(step_metrics.get("walk/num_chains", 1), 1)
        )
        step_metrics["walk/extraction_failed_rate"] = step_metrics["walk/dummy_rate"]
        step_metrics.update(self._source_reward_metrics(solver_gen_batch, solver_seq_raw_rewards))

        return proposer_rewards, solver_train_rewards

    def _get_solver_format_coef(self) -> float:
        solver_cfg = self.sp_config.get("solver", {})
        start = float(solver_cfg.get("format_coef_start", 1.0))
        end = float(solver_cfg.get("format_coef_end", start))
        anneal_steps = int(solver_cfg.get("format_anneal_steps", 0))
        if anneal_steps <= 0:
            return end
        progress = min(max(self.global_steps, 0) / max(anneal_steps, 1), 1.0)
        return start + (end - start) * progress

    def _get_current_kl_loss_coef(self) -> float:
        adaptive_cfg = self.sp_config.get("adaptive_kl", {})
        base_coef = float(self.config.actor_rollout_ref.actor.kl_loss_coef)
        if not adaptive_cfg.get("enable", False):
            return base_coef
        if not hasattr(self, "_coevokg_dynamic_kl_coef"):
            self._coevokg_dynamic_kl_coef = float(adaptive_cfg.get("init_kl_coef", base_coef))
        return float(self._coevokg_dynamic_kl_coef)

    def _update_dynamic_kl_loss_coef(self, observed_kl: float) -> Dict[str, float]:
        adaptive_cfg = self.sp_config.get("adaptive_kl", {})
        if not adaptive_cfg.get("enable", False) or observed_kl is None:
            return {}

        current = self._get_current_kl_loss_coef()
        target = float(adaptive_cfg.get("target_kl_loss", 0.1))
        tolerance = float(adaptive_cfg.get("tolerance", 0.2))
        up_factor = float(adaptive_cfg.get("up_factor", 1.2))
        down_factor = float(adaptive_cfg.get("down_factor", 0.95))
        min_coef = float(adaptive_cfg.get("min_kl_coef", 0.0))
        max_coef = float(adaptive_cfg.get("max_kl_coef", max(current, 1.0)))

        next_coef = current
        if observed_kl > target * (1.0 + tolerance):
            next_coef = current * up_factor
        elif observed_kl < target * (1.0 - tolerance):
            next_coef = current * down_factor
        next_coef = min(max(next_coef, min_coef), max_coef)
        self._coevokg_dynamic_kl_coef = float(next_coef)
        return {
            "adaptive_kl/observed_kl": float(observed_kl),
            "adaptive_kl/target_kl": target,
            "adaptive_kl/coef": float(next_coef),
        }

    @staticmethod
    def _extract_doc_titles(info_text: str) -> List[str]:
        """Extract retrieved document titles from an <information> block.

        KILT-style retrieval results are formatted as:
          Doc 1 (Title: "National People's Congress") ...
        We pull the titles in order, deduplicating consecutive repeats while
        preserving first-seen order.
        """
        import re
        titles = re.findall(r'Title:\s*"([^"]+)"', info_text)
        seen, ordered = set(), []
        for t in titles:
            key = t.lower().strip()
            if key and key not in seen:
                seen.add(key)
                ordered.append(t.strip())
        return ordered

    def _title_passage_map(self, turns: List[Dict]):
        """From tool turns, return (ordered unique retrieved titles, {title: content}).

        Mirrors the title/passage collection in `_discover_chain_from_trajectory`;
        used at write-back to attach node documents to an LLM-extracted chain.
        """
        title_seq: List[str] = []
        title_text: Dict[str, str] = {}
        for turn in turns:
            if turn.get("role") != "tool":
                continue
            content = turn.get("content", "")
            for title in self._extract_doc_titles(content):
                if title not in title_text:
                    title_seq.append(title)
                title_text[title] = (title_text.get(title, "") + "\n" + content).strip()[:2000]
        return title_seq, title_text

    def _discover_chain_from_trajectory(
        self, turns: List[Dict], ground_truth, orig_relations: List[str],
    ) -> Optional[Dict]:
        """Construct a brand-new reasoning chain from a verified solver trajectory.

        The chain is built from the sequence of documents the solver actually
        retrieved (in search order) terminating at the ground-truth answer:

            [title_1, title_2, ..., title_k, answer_entity]

        Returns a chain dict {id, relations, nodes, source, step} or None when
        the trajectory does not yield a usable multi-hop chain.
        """
        # Collect retrieved titles in search order, with their evidence text.
        title_seq: List[str] = []
        title_text: Dict[str, str] = {}
        for turn in turns:
            if turn.get("role") != "tool":
                continue
            content = turn.get("content", "")
            for title in self._extract_doc_titles(content):
                if title not in title_text:
                    title_seq.append(title)
                title_text[title] = (title_text.get(title, "") + "\n" + content).strip()[:2000]

        if not title_seq:
            return None

        # Determine the answer entity from ground truth.
        gt = ground_truth
        if isinstance(gt, dict):
            gt = gt.get("target", "")
        if isinstance(gt, (list, tuple)):
            gt = gt[0] if gt else ""
        gt = str(gt).strip()
        if not gt:
            return None

        # Build the new entity sequence: retrieved titles + answer (if distinct).
        new_relations = list(title_seq)
        if gt.lower() not in {t.lower() for t in new_relations}:
            new_relations.append(gt)

        # Require a genuine multi-hop chain and that it differs from the source chain.
        min_hops = int(getattr(self.chain_pool, "min_hops", 2))
        if len(new_relations) < min_hops:
            return None
        if [r.lower() for r in new_relations] == [r.lower() for r in orig_relations]:
            return None

        new_nodes = {t: title_text.get(t, "") for t in new_relations}
        return {
            "id":        uuid.uuid4().hex,
            "relations": new_relations,
            "nodes":     new_nodes,
            "source":    "chain_discovered",
            "step":      self.global_steps,
        }

    def _solver_chain_from_turns(
        self, turns: List[Dict], ground_truth,
    ) -> List[str]:
        """Build the solver's retrieved entity chain for path-support scoring.

        Mirrors the title sequence used by ``_discover_chain_from_trajectory``:
        the documents the solver actually retrieved (in search order) terminating
        at the ground-truth answer. Returns [] when no usable chain is found.
        """
        title_seq: List[str] = []
        seen = set()
        for turn in turns:
            if turn.get("role") != "tool":
                continue
            for title in self._extract_doc_titles(turn.get("content", "")):
                if title.lower() not in seen:
                    seen.add(title.lower())
                    title_seq.append(title)

        gt = ground_truth
        if isinstance(gt, dict):
            gt = gt.get("target", "")
        if isinstance(gt, (list, tuple)):
            gt = gt[0] if gt else ""
        gt = str(gt).strip()

        chain = list(title_seq)
        if gt and gt.lower() not in {t.lower() for t in chain}:
            chain.append(gt)
        return chain

    def _write_correct_trajectories_to_kg(
        self, solver_gen_batch: DataProto, solver_rewards: torch.Tensor
    ) -> int:
        """Write verified solver knowledge back to the chain pool (co-evolution).

        For each correctly-answered trajectory (raw score >= 1.0) two write-back
        modes operate (each gated by config):

          1. Node enrichment (default): append retrieved <information> evidence to
             the matching nodes of the ORIGINAL sampled chain — enriches node text,
             entity sequence unchanged.

          2. Chain discovery (kg_writer.discover_new_chains=true): build a BRAND-NEW
             chain from the solver's retrieved document-title sequence terminating at
             the answer, adding genuinely new entity sequences to the pool.

        Returns the number of chains actually written (new or enriched).
        """
        if self.kg_writer is None or self.chain_pool is None:
            return 0

        correct_mask = solver_rewards.sum(-1) >= 1.0
        if not correct_mask.any():
            return 0

        kg_cfg = self.sp_config.get("kg_writer", {})
        enrich_nodes   = bool(kg_cfg.get("enrich_nodes", True))
        discover_chains = bool(kg_cfg.get("discover_new_chains", False))
        # When true, write back a COMPLETE chain (entities + relations + node docs)
        # extracted by the LLM from the correct trajectory; falls back to the
        # deterministic Mode 1 / Mode 2 logic below on empty/error.
        llm_extract = bool(kg_cfg.get("llm_extract", True))
        # Optional path-support gate: when >0, a correct trajectory is written back
        # only if its solver-retrieved chain is also evidence-supported (global_score
        # >= min_path_support). 0.0 disables the gate (default; current behaviour).
        min_path_support = float(kg_cfg.get("min_path_support", 0.0))
        gated_out = 0

        extra_infos  = solver_gen_batch.non_tensor_batch.get("extra_info", [])
        reward_models = solver_gen_batch.non_tensor_batch.get("reward_model", [])
        resp_ids     = solver_gen_batch.batch["responses"]
        resp_masks   = solver_gen_batch.batch["response_mask"]

        # ── Pass 1: gather per-correct-trajectory context (decode turns once) ──
        items = []
        for i in torch.where(correct_mask)[0].tolist():
            ei = extra_infos[i] if i < len(extra_infos) else {}
            valid_len = int(resp_masks[i].sum().item())
            response_text = self.tokenizer.decode(
                resp_ids[i, :valid_len], skip_special_tokens=True
            )
            turns = self._parse_response_turns(response_text)
            info_blocks = [
                t.get("content", "")
                for t in turns
                if t.get("role") == "tool" and t.get("content", "")
            ]
            if not info_blocks:
                continue
            rm = reward_models[i] if i < len(reward_models) else {}
            gt = rm.get("ground_truth") if isinstance(rm, dict) else None
            if gt is None:
                gt = ei.get("ground_truth", "")
            if isinstance(gt, dict):
                gt = gt.get("target", "")
            items.append({"i": i, "ei": ei, "turns": turns,
                          "info_blocks": info_blocks, "gt": str(gt)})

        # ── Pass 2: reuse the reward-stage chain payload (NO LLM call here) ──
        # compute_score already judged correctness AND extracted the canonical chain +
        # relations + path_support in one call; the payload was passed back via
        # reward_extra_info as a JSON string keyed by trajectory index. We reuse it here
        # instead of a second extraction, so the whole pipeline makes exactly one LLM
        # call per trajectory.  llm_out[i] = (chain, relations, path_support).
        llm_out: Dict[int, tuple] = {}
        kg_payloads = getattr(self, "_solver_kg_payloads", None)
        if llm_extract and kg_payloads is not None:
            for it in items:
                i = it["i"]
                raw = kg_payloads[i] if i < len(kg_payloads) else None
                if not raw:
                    continue
                try:
                    payload = json.loads(raw)
                except Exception:
                    continue
                if not payload.get("correct"):
                    continue
                chain = [str(e) for e in (payload.get("chain") or []) if str(e).strip()]
                rels = [str(r) for r in (payload.get("relations") or [])]
                ps = float(payload.get("path_support", 0.0) or 0.0)
                if len(chain) >= 2:
                    llm_out[i] = (chain, rels, ps)

        # ── Pass 3: gate + write (LLM complete chain, else deterministic Mode 1/2) ──
        futures = []
        for it in items:
            i = it["i"]; ei = it["ei"]; turns = it["turns"]
            gt = it["gt"]; info_blocks = it["info_blocks"]
            relations = ei.get("chain_relations", [])
            chain_path = ei.get("chain_data_path", "")

            llm_chain, llm_rels, ps = llm_out.get(i, ([], [], 0.0))
            if llm_extract and len(llm_chain) >= 2:
                # path_support was already computed at reward time on this same chain;
                # reuse it for the gate (no second verifier call).
                if min_path_support > 0.0 and ps < min_path_support:
                    gated_out += 1
                    continue
                # Assemble a COMPLETE writable chain: entities + relations + node docs.
                _titles, tmap = self._title_passage_map(turns)
                nodes = {e: tmap.get(e, "") for e in llm_chain}
                need = len(llm_chain) - 1
                rl = (list(llm_rels) + [""] * need)[:need]
                complete_chain: Dict = {
                    "id":               uuid.uuid4().hex,
                    "relations":        llm_chain,
                    "relation_labels":  rl,
                    "nodes":            nodes,
                    "chain_length":     len(llm_chain),
                    "answer_wiki_title": llm_chain[-1],
                    "source":           "llm_extracted_writeback",
                    "step":             self.global_steps,
                    "path_support":     round(float(ps), 3),
                }
                futures.append(self.kg_writer.add_chain.remote(complete_chain))
                continue

            # ── Fallback: deterministic path-support gate ──
            if min_path_support > 0.0:
                solver_chain = self._solver_chain_from_turns(turns, gt)
                if not (chain_path and len(solver_chain) >= 2):
                    gated_out += 1
                    continue
                try:
                    from coevokg.utils.path_support_verifier import get_global_verifier
                    ps = get_global_verifier(chain_path).verify_chain(solver_chain).global_score
                except Exception:
                    ps = 0.0
                if ps < min_path_support:
                    gated_out += 1
                    continue

            # ── Mode 1: enrich nodes of the original chain ──────────────────
            if enrich_nodes and relations:
                augmented_nodes: Dict[str, str] = dict(ei.get("chain_nodes", {}))
                combined_info = "\n".join(info_blocks)
                for entity in relations:
                    if entity.lower() in combined_info.lower():
                        prev = augmented_nodes.get(entity, "")
                        augmented_nodes[entity] = (prev + "\n" + combined_info).strip()[:2000]

                enriched_chain: Dict = {
                    "id":        uuid.uuid4().hex,
                    "relations": relations,
                    "nodes":     augmented_nodes,
                    "source":    "node_enriched",
                    "step":      self.global_steps,
                }
                rl = ei.get("relation_labels")
                if rl:
                    enriched_chain["relation_labels"] = rl
                futures.append(self.kg_writer.add_chain.remote(enriched_chain))

            # ── Mode 2: discover a brand-new chain from the trajectory ──────
            if discover_chains:
                discovered = self._discover_chain_from_trajectory(turns, gt, relations)
                if discovered is not None:
                    futures.append(self.kg_writer.add_chain.remote(discovered))

        if not futures:
            if gated_out:
                print(f"KGWriter: 0 chains written ({gated_out} gated out by path support)")
            return 0

        results = ray.get(futures)
        added = sum(1 for r in results if r)

        if added > 0:
            new_chains = ray.get(self.kg_writer.get_new_chains.remote())
            for c in new_chains:
                self.chain_pool.add_chain(c)
            print(
                f"KGWriter: +{added} chains written "
                f"(pool size={len(self.chain_pool.chains)}, gated_out={gated_out})"
            )

        return added

    def _apply_advantage_reweighting(self, gen_batch: DataProto, role: str) -> Dict[str, float]:
        cfg = self.sp_config.get("advantage_reweighting", {})
        if not cfg.get("enable", False):
            return {}
        if role == "proposer" and not cfg.get("proposer_enable", True):
            return {}
        if role == "solver" and not cfg.get("solver_enable", True):
            return {}
        if role not in {"proposer", "solver"}:
            return {}
        if "advantages" not in gen_batch.batch or "old_log_probs" not in gen_batch.batch:
            return {}

        alpha_key = "proposer_alpha" if role == "proposer" else "solver_alpha"
        alpha = float(cfg.get(alpha_key, cfg.get("alpha", 0.1)))
        min_prob = float(cfg.get("min_token_prob", 1e-6))
        max_prob = float(cfg.get("max_token_prob", 1.0))

        token_prob = gen_batch.batch["old_log_probs"].detach().float().exp()
        token_prob = torch.clamp(token_prob, min=min_prob, max=max_prob)
        weight = alpha * token_prob + (1.0 - alpha)
        if "response_mask" in gen_batch.batch:
            mask = gen_batch.batch["response_mask"].to(weight.device).float()
            weight = weight * mask + (1.0 - mask)
        gen_batch.batch["advantages"] = gen_batch.batch["advantages"] * weight.to(
            gen_batch.batch["advantages"].device
        )

        active_weight = weight
        if "response_mask" in gen_batch.batch:
            mask = gen_batch.batch["response_mask"].to(weight.device).bool()
            active_weight = weight[mask] if mask.any() else weight.reshape(-1)
        else:
            active_weight = weight.reshape(-1)
        return {
            "adv_reweight/alpha": alpha,
            "adv_reweight/weight_mean": float(active_weight.mean().item()),
            "adv_reweight/weight_min": float(active_weight.min().item()),
            "adv_reweight/weight_max": float(active_weight.max().item()),
        }

    def _feedback_pool_enabled(self) -> bool:
        return bool(self.sp_config.get("feedback_pool", {}).get("enable", False))

    def _add_feedback_pool_problems(self, problems: List[Dict[str, Any]]) -> None:
        if not self._feedback_pool_enabled():
            return
        reusable: List[Dict[str, Any]] = []
        for problem in problems:
            extra_info = problem.get("extra_info", {}) or {}
            data_source = str(problem.get("data_source", ""))
            if extra_info.get("extraction_failed", False):
                continue
            if data_source in {"dummy", "seed_fallback", "feedback_pool"}:
                continue
            reusable.append(deepcopy(problem))
        if not reusable:
            return
        try:
            ray.get(self.sp_data_manager.add_generated_problems.remote(reusable, self.global_steps))
        except Exception as exc:
            logger.warning(f"Failed to add problems to feedback pool: {exc}")

    def _sample_feedback_pool_problem(self, trajectory_index: int) -> Optional[Dict[str, Any]]:
        feedback_cfg = self.sp_config.get("feedback_pool", {})
        if not feedback_cfg.get("enable", False) or not feedback_cfg.get("prefer_for_fallback", True):
            return None
        try:
            problems = ray.get(self.sp_data_manager.get_random_existing_problems.remote(1))
        except Exception as exc:
            logger.warning(f"Failed to sample feedback pool problem: {exc}")
            return None
        if not problems:
            return None
        problem = None
        for candidate in problems:
            if not isinstance(candidate, dict):
                continue
            if "prompt" not in candidate or "reward_model" not in candidate:
                continue
            problem = deepcopy(candidate)
            break
        if problem is None:
            return None
        problem["data_source"] = "feedback_pool"
        problem["trajectory_index"] = trajectory_index
        problem.setdefault("extra_info", {})
        if isinstance(problem["extra_info"], dict):
            problem["extra_info"]["extraction_failed"] = True
            problem["extra_info"]["feedback_pool_reused"] = True
            problem["extra_info"]["trajectory_index"] = trajectory_index
        return problem

    def _inject_process_consistency_info(self, problems: List[Dict]) -> None:
        """
        Inject chain_data_path and beta_proc into each problem's extra_info so
        that compute_score() can access them inside the reward worker.

        Only injects when walk mode is active and chain_data_path is configured.
        """
        walk_cfg = self.sp_config.proposer.get("walk", {})
        local_chain_path = get_coevokg_env("CHAIN_DATA_LOCAL", "")
        chain_data_path = (
            local_chain_path
            if local_chain_path and os.path.exists(local_chain_path)
            else walk_cfg.get("chain_data_path", "")
        )
        beta_proc = float(self.sp_config.get("solver", {}).get("process_consistency_beta", 0.3))

        if not chain_data_path:
            return

        for problem in problems:
            problem.setdefault("extra_info", {})
            if isinstance(problem["extra_info"], dict):
                problem["extra_info"]["chain_data_path"] = chain_data_path
                problem["extra_info"]["beta_proc"] = beta_proc

    def _compute_proposer_rewards(
        self, proposer_gen_batch: DataProto, solver_rewards: torch.Tensor, solver_extra_infos: Dict
    ) -> torch.Tensor:
        # ------------------------------------------------------------------
        # Walk mode: proposer_batch_size (generate_n) != solver problems
        # ------------------------------------------------------------------
        if self._proposer_mode == "walk" and self._walk_generate_n_last > 0:
            return self._compute_proposer_rewards_walk(proposer_gen_batch, solver_rewards)

        # ------------------------------------------------------------------
        # Original seed mode (unchanged)
        # ------------------------------------------------------------------
        proposer_batch_size = len(proposer_gen_batch.batch)
        response_length = proposer_gen_batch.batch["responses"].size(1)

        solver_n_attempts = self.rollout_n
        solver_batch_size = solver_rewards.size(0)
        assert (
            solver_batch_size == proposer_batch_size * solver_n_attempts
        ), f"Expected {proposer_batch_size * solver_n_attempts} solver samples, got {solver_batch_size}"

        solver_rewards_per_attempt = solver_rewards.sum(-1)

        solver_rewards_grouped = solver_rewards_per_attempt.view(proposer_batch_size, solver_n_attempts)

        success_rates = (solver_rewards_grouped >= 1.0).float().mean(dim=1)

        reward_type = self.sp_config.get("proposer", {}).get("reward_type", "1-acc")

        if reward_type == "intermediate_difficulty":
            proposer_reward_values = torch.zeros_like(success_rates)

            proposer_right = self.sp_config.proposer.get("right", 1.0)
            proposer_left = self.sp_config.proposer.get("left", 0.0)
            intermediate_mask = (success_rates > 0) & (success_rates < 1)
            proposer_reward_values[intermediate_mask] = (
                4.0
                * (proposer_left + success_rates[intermediate_mask])
                * (proposer_right - success_rates[intermediate_mask])
            )
        elif reward_type == "quadratic_decay":
            # Smoother monotone-decreasing variant of 1-acc.
            # r = 1 - acc^2  →  acc=0→1.0, acc=0.5→0.75, acc=1→0.0
            # Gentler gradient in the mid-difficulty range vs. linear 1-acc.
            proposer_reward_values = 1.0 - success_rates ** 2
        elif reward_type == "format_only":
            proposer_reward_values = torch.ones_like(success_rates)
        else:  # "1-acc" (default) and "bell_curriculum" fall through here in seed mode
            proposer_reward_values = 1.0 - success_rates

        extraction_failure_penalty = self.sp_config.proposer.get("format_penalty", -0.1)

        if hasattr(self, "current_extraction_success_mask"):
            for i, extraction_success in enumerate(self.current_extraction_success_mask):
                if not extraction_success:
                    proposer_reward_values[i] = extraction_failure_penalty
                    if SELF_PLAY_DEBUG:
                        logger.debug(
                            f"Applying extraction failure penalty {extraction_failure_penalty} to proposer generation {i}"
                        )
        elif hasattr(self, "generate_problem") and self.generate_problem:
            assert len(self.generate_problem) == len(
                proposer_reward_values
            ), "generate_problem and proposer_reward_values must have the same length"
            for i, problem in enumerate(self.generate_problem):
                if i < len(proposer_reward_values):
                    data_source = problem.get("data_source", "")
                    if data_source == "dummy":
                        proposer_reward_values[i] = extraction_failure_penalty
                        if SELF_PLAY_DEBUG:
                            logger.debug(
                                f"Applying extraction failure penalty {extraction_failure_penalty} to proposer generation {i} (dummy data_source)"
                            )

        proposer_rewards = torch.zeros(
            (proposer_batch_size, response_length),
            dtype=torch.float32,
            device=solver_rewards.device,
        )

        if "response_mask" in proposer_gen_batch.batch:
            response_mask = proposer_gen_batch.batch["response_mask"]
            last_token_positions = response_mask.sum(-1) - 1
            for i in range(proposer_batch_size):
                last_pos = last_token_positions[i].item()
                if 0 <= last_pos < response_length:
                    proposer_rewards[i, last_pos] = proposer_reward_values[i].item()
        else:
            proposer_rewards[:, -1] = proposer_reward_values

        extraction_failures = 0
        if hasattr(self, "current_extraction_success_mask"):
            extraction_failures = sum(1 for x in self.current_extraction_success_mask if not x)

        print(f"Proposer reward computation (using formula: {reward_type}):")
        print(f"  - Success rates: mean={success_rates.mean():.3f}, std={success_rates.std():.3f}")
        print(
            f"  - Proposer rewards: mean={proposer_reward_values.mean():.3f}, std={proposer_reward_values.std():.3f}, max={proposer_reward_values.max():.3f}"
        )
        print(f"  - Total problems: {proposer_batch_size}")
        print(f"  - Extraction failures: {extraction_failures}/{proposer_batch_size}")

        print(f"  - Reward range: [{proposer_reward_values.min():.3f}, {proposer_reward_values.max():.3f}]")

        if reward_type == "intermediate_difficulty":
            zero_success = (success_rates == 0).sum().item()
            full_success = (success_rates == 1).sum().item()
            intermediate_count = ((success_rates > 0) & (success_rates < 1)).sum().item()
            print(f"  - Problems with success_rate=0 (zero reward): {zero_success}")
            print(f"  - Problems with success_rate=1 (zero reward): {full_success}")
            print(f"  - Problems with intermediate difficulty (non-zero reward): {intermediate_count}")
            if intermediate_count > 0:
                intermediate_rates = success_rates[(success_rates > 0) & (success_rates < 1)]
                intermediate_rewards = proposer_reward_values[(success_rates > 0) & (success_rates < 1)]
                print(
                    f"  - Intermediate problems success_rate range: [{intermediate_rates.min():.3f}, {intermediate_rates.max():.3f}]"
                )
                print(
                    f"  - Intermediate problems reward range: [{intermediate_rewards.min():.3f}, {intermediate_rewards.max():.3f}]"
                )
        else:
            print(f"  - Problems with success_rate=0 (max reward): {(success_rates == 0).sum().item()}")
            print(f"  - Problems with success_rate=1 (min reward): {(success_rates == 1).sum().item()}")
            print(f"  - Problems with 0<success_rate<1: {((success_rates > 0) & (success_rates < 1)).sum().item()}")

        return proposer_rewards

    def _compute_proposer_rewards_walk(
        self, proposer_gen_batch: DataProto, solver_rewards: torch.Tensor
    ) -> torch.Tensor:
        """
        Walk-mode proposer reward computation (K-candidate-selected version).

        proposer_gen_batch: K selected trajectories (one per chain).
        solver_rewards    : (K * rollout_n, response_length)

        For each chain k:
            difficulty_reward_k = f(solver_success_rate_k)   [quadratic_decay / etc.]
            reward_k = avg_format_reward_k + difficulty_reward_k

        Reinforce++ normalization is applied across all K rewards.
        Skill library update is done at selection time, NOT here.
        """
        K = len(proposer_gen_batch.batch)         # = batch_size = num_chains
        response_length = proposer_gen_batch.batch["responses"].size(1)
        solver_n_attempts = self.rollout_n
        reward_type = self.sp_config.get("proposer", {}).get("reward_type", "1-acc")

        # ── Solver success rates ──────────────────────────────────────────────
        solver_batch_size = solver_rewards.size(0)
        num_solver_problems = solver_batch_size // solver_n_attempts  # should equal K
        solver_rewards_per_attempt = solver_rewards.sum(-1)           # (K * rollout_n,)
        solver_grouped = solver_rewards_per_attempt.view(num_solver_problems, solver_n_attempts)
        # Use final-answer correctness for proposer difficulty.  Solver rewards
        # may contain small positive process rewards for wrong answers, so >0
        # would overestimate how often a question is truly solved.
        positive_rates = (solver_grouped > 0).float().mean(dim=1)      # (K,)
        success_rates = (solver_grouped >= 1.0).float().mean(dim=1)    # (K,)

        # ── Difficulty reward per chain ───────────────────────────────────────
        # Chains where extraction failed use a seed_fallback question; their solver
        # success rate reflects a random seed question, not the proposer's difficulty,
        # so we exclude them from the difficulty signal (set r=0).
        extraction_success_mask = getattr(self, "current_extraction_success_mask", None)

        difficulty_rewards = torch.zeros(K, dtype=torch.float32)
        valid_accs = []
        valid_positive_rates = []
        for k in range(min(K, num_solver_problems)):
            # Skip difficulty reward for failed extractions
            if extraction_success_mask is not None and k < len(extraction_success_mask) \
                    and not extraction_success_mask[k]:
                difficulty_rewards[k] = 0.0
                continue

            acc = success_rates[k].item()
            valid_accs.append(acc)
            valid_positive_rates.append(positive_rates[k].item())
            if reward_type == "quadratic_decay":
                r = 1.0 - acc ** 2
            elif reward_type == "intermediate_difficulty":
                proposer_right = self.sp_config.proposer.get("right", 1.0)
                proposer_left = self.sp_config.proposer.get("left", 0.0)
                r = 4.0 * (proposer_left + acc) * (proposer_right - acc) if 0 < acc < 1 else 0.0
            elif reward_type == "bell_curriculum":
                curriculum = self.sp_config.proposer.get("curriculum", {})
                start_t = curriculum.get("start_target_acc", 0.7)
                end_t = curriculum.get("end_target_acc", 0.2)
                sigma = curriculum.get("sigma", 0.15)
                progress = min(self.global_steps / max(self.total_training_steps, 1), 1.0)
                target_acc = start_t - (start_t - end_t) * progress
                import math as _math
                r = _math.exp(-(acc - target_acc) ** 2 / (2 * sigma ** 2))
            elif reward_type == "mixed_curriculum":
                # Weighted sum of Gaussians at easy/medium/hard difficulty targets.
                # Each tier contributes independently: Proposer is rewarded for questions
                # at any of the three difficulty levels, preventing collapse to a single target.
                import math as _math
                multi_level = self.sp_config.proposer.get("multi_level", {})
                tiers = multi_level.get("tiers", [
                    {"target_acc": 0.70, "sigma": 0.15, "weight": 0.20},
                    {"target_acc": 0.40, "sigma": 0.15, "weight": 0.50},
                    {"target_acc": 0.15, "sigma": 0.10, "weight": 0.30},
                ])
                r = sum(
                    t.get("weight", 1.0) * _math.exp(
                        -(acc - t.get("target_acc", 0.5)) ** 2
                        / (2 * t.get("sigma", 0.15) ** 2)
                    )
                    for t in tiers
                )
            elif reward_type == "format_only":
                r = 1.0
            else:  # "1-acc"
                r = 1.0 - acc
            difficulty_rewards[k] = r

        # ── avg_format_reward from selection phase ────────────────────────────
        avg_format_rewards = torch.tensor(
            getattr(self, "_chain_avg_format_rewards", [0.0] * K),
            dtype=torch.float32,
        )[:K]

        # ── Proposer reward = pure difficulty ────────────────────────────────
        # Format is enforced upstream as a HARD GATE (malformed candidates fail
        # extraction in Pass 1 and are never selected), not as a soft reward term,
        # so the proposer is optimized purely on difficulty. avg_format_rewards is
        # still computed above for logging/metrics only. Completely-failed
        # extractions get a fixed penalty below.
        combined_rewards = difficulty_rewards.clone()
        if extraction_success_mask is not None:
            extraction_failure_penalty = float(self.sp_config.proposer.get("format_penalty", -0.1))
            for k in range(min(K, len(extraction_success_mask))):
                if not extraction_success_mask[k]:
                    combined_rewards[k] = extraction_failure_penalty

        # ── Scatter to last token of each selected trajectory ─────────────────
        proposer_rewards = torch.zeros(
            (K, response_length), dtype=torch.float32, device=solver_rewards.device
        )
        if "response_mask" in proposer_gen_batch.batch:
            response_mask = proposer_gen_batch.batch["response_mask"]
            last_token_positions = response_mask.sum(-1) - 1
            for i in range(K):
                last_pos = last_token_positions[i].item()
                if 0 <= last_pos < response_length:
                    proposer_rewards[i, last_pos] = combined_rewards[i].item()
        else:
            proposer_rewards[:, -1] = combined_rewards

        valid_acc_mean = sum(valid_accs) / len(valid_accs) if valid_accs else 0.0
        valid_positive_mean = sum(valid_positive_rates) / len(valid_positive_rates) if valid_positive_rates else 0.0
        print(
            f"Walk proposer rewards ({reward_type}): "
            f"K={K}, valid_chains={len(valid_accs)}/{K}, "
            f"avg_fmt={avg_format_rewards.mean():.3f}, "
            f"avg_diff={difficulty_rewards.mean():.3f}, "
            f"avg_combined={combined_rewards.mean():.3f}, "
            f"std={combined_rewards.std():.3f}, "
            f"solver_correct_rate(valid only)={valid_acc_mean:.3f}, "
            f"solver_positive_rate(valid only)={valid_positive_mean:.3f}"
        )
        return proposer_rewards

    def _source_reward_metrics(self, solver_gen_batch: DataProto, solver_seq_rewards: torch.Tensor) -> Dict[str, Any]:
        data_sources = (solver_gen_batch.non_tensor_batch or {}).get("data_source", None)
        if data_sources is None:
            return {}

        rewards = solver_seq_rewards.detach().cpu().tolist()
        sources = [str(x) for x in data_sources]
        total = max(len(sources), 1)
        metrics: Dict[str, Any] = {}

        for source in ("walk_chain", "feedback_pool", "seed_fallback", "dummy"):
            vals = [r for r, s in zip(rewards, sources) if s == source]
            if not vals:
                continue
            prefix = f"solver/{source}"
            metrics[f"{prefix}_count"] = len(vals)
            metrics[f"{prefix}_rate"] = len(vals) / total
            metrics[f"{prefix}_reward_mean"] = float(np.mean(vals))
            metrics[f"{prefix}_correct_rate"] = float(np.mean([v >= 1.0 for v in vals]))

        metrics["walk/seed_fallback_rate"] = sources.count("seed_fallback") / total
        metrics["walk/real_dummy_rate"] = sources.count("dummy") / total
        return metrics

    def _prefill_policy_log_probs(self, gen_batch: DataProto, role: str, timing_raw: Dict) -> Tuple[DataProto, Dict[str, Any]]:
        metrics: Dict[str, Any] = {}

        if self.config.trainer.balance_batch and not gen_batch.meta_info.get("_coevokg_balanced_for_update", False):
            self._balance_batch(gen_batch, metrics=metrics)
            gen_batch.meta_info["_coevokg_balanced_for_update"] = True

        if "old_log_probs" not in gen_batch.batch:
            with marked_timer(f"{role}_old_log_prob", timing_raw, color="blue"):
                old_log_prob = self.actor_rollout_wg.compute_log_prob(gen_batch)
                entropys = old_log_prob.batch.get("entropys", None)
                if entropys is not None:
                    response_masks = gen_batch.batch["response_mask"]
                    loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                    entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                    metrics[f"{role}/actor/entropy"] = entropy_agg.detach().item()
                    old_log_prob.batch.pop("entropys", None)
                gen_batch = gen_batch.union(old_log_prob)

        if self.use_reference_policy and "ref_log_prob" not in gen_batch.batch:
            with marked_timer(f"{role}_ref", timing_raw, color="olive"):
                if not self.ref_in_actor:
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(gen_batch)
                else:
                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(gen_batch)
                gen_batch = gen_batch.union(ref_log_prob)

        return gen_batch, metrics

    def _update_on_trajectories(self, gen_batch: DataProto, role: str, timing_raw: Dict) -> Dict[str, Any]:
        print(f"=== Step 5: {role.title()} Update at step {self.global_steps} ===")

        gen_batch, metrics = self._prepare_trajectories_for_update(gen_batch, role, timing_raw)

        if self.use_critic:
            with marked_timer(f"{role}_update_critic", timing_raw, color="pink"):
                critic_output = self.critic_wg.update_critic(gen_batch)
            critic_metrics = reduce_metrics(critic_output.meta_info["metrics"])
            metrics.update({f"{role}_critic_{k}": v for k, v in critic_metrics.items()})

        metrics.update(self._update_actor_on_prepared_trajectories(gen_batch, role, timing_raw))
        metrics.update(self._reward_summary_metrics(gen_batch, role))

        return metrics

    def _prepare_trajectories_for_update(
        self, gen_batch: DataProto, role: str, timing_raw: Dict
    ) -> Tuple[DataProto, Dict[str, Any]]:
        metrics: Dict[str, Any] = {}

        # Dummy solver samples (placeholder chains with no real question) must not
        # contribute to the gradient: they all share one identical prompt, polluting
        # entropy and response-length statistics and injecting noise into policy loss.
        if role == "solver" and "response_mask" in gen_batch.batch:
            data_sources = (gen_batch.non_tensor_batch or {}).get("data_source", None)
            if data_sources is not None:
                dummy_flags = torch.tensor(
                    [str(s) == "dummy" for s in data_sources],
                    dtype=torch.bool,
                    device=gen_batch.batch["response_mask"].device,
                )
                if dummy_flags.any():
                    gen_batch.batch["response_mask"][dummy_flags] = 0
                    print(f"  [dummy mask] zeroed response_mask for {dummy_flags.sum().item()}"
                          f"/{len(dummy_flags)} dummy solver samples")
                    metrics["solver/dummy_masked_count"] = int(dummy_flags.sum().item())

        if self.config.trainer.balance_batch and not gen_batch.meta_info.get("_coevokg_balanced_for_update", False):
            self._balance_batch(gen_batch, metrics=metrics)

        gen_batch.meta_info["global_token_num"] = torch.sum(gen_batch.batch["attention_mask"], dim=-1).tolist()

        for key in gen_batch.batch.keys():
            if key not in [
                "old_log_probs",
                "ref_log_prob",
                "token_level_scores",
                "token_level_rewards",
                "advantages",
                "returns",
                "values",
                "solver_raw_scores",
                "solver_format_rewards",
            ]:
                gen_batch.batch[key] = gen_batch.batch[key].long()

        gen_batch, policy_metrics = self._prefill_policy_log_probs(gen_batch, role, timing_raw)
        metrics.update(policy_metrics)

        if self.use_critic:
            with marked_timer(f"{role}_values", timing_raw, color="cyan"):
                values = self.critic_wg.compute_values(gen_batch)
                gen_batch = gen_batch.union(values)

        with marked_timer(f"{role}_adv", timing_raw, color="brown"):
            if self.config.algorithm.use_kl_in_reward:
                gen_batch, kl_metrics = apply_kl_penalty(
                    gen_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                )
                metrics.update({f"{role}_{k}": v for k, v in kl_metrics.items()})
            else:
                gen_batch.batch["token_level_rewards"] = gen_batch.batch["token_level_scores"]

            # Legacy path: older batches stored solver format reward outside token_level_scores.
            # New self-play batches already include format reward in token_level_scores.
            if (
                role == "solver"
                and "solver_format_rewards" in gen_batch.batch
                and "solver_raw_scores" not in gen_batch.batch
            ):
                fmt = gen_batch.batch["solver_format_rewards"].to(
                    gen_batch.batch["token_level_rewards"].device
                )
                gen_batch.batch["token_level_rewards"] = gen_batch.batch["token_level_rewards"] + fmt
                _nz = fmt[fmt != 0]
                print(f"  [format_reward] added to token_level_rewards: "
                      f"mean={float(_nz.mean()):.3f}" if _nz.numel() > 0 else
                      "  [format_reward] no non-zero entries")

            norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
            if role == "solver":
                num_repeat = self.rollout_n
                adv_estimator = self.config.algorithm.adv_estimator
            else:
                if self._proposer_mode == "walk" and self._walk_generate_n_last > 0:
                    # Walk mode: all generate_n trajectories form one group for GRPO advantage
                    num_repeat = self._walk_generate_n_last
                    adv_estimator = self.sp_config.proposer.adv_estimator
                else:
                    num_repeat = self.sp_config.proposer.n
                    if num_repeat == 1:
                        adv_estimator = self.sp_config.proposer.adv_estimator
                    else:
                        adv_estimator = self.config.algorithm.adv_estimator
            gen_batch = compute_advantage(
                gen_batch,
                adv_estimator=adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=num_repeat,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                config=self.config.algorithm,
            )
            metrics.update(self._apply_advantage_reweighting(gen_batch, role))
            adv_weight = 1.0
            if role == "proposer":
                adv_weight = float(self.sp_config.get("proposer_adv_weight", 1.0))
            elif role == "solver":
                adv_weight = float(self.sp_config.get("solver_adv_weight", 1.0))
            if adv_weight != 1.0 and "advantages" in gen_batch.batch:
                gen_batch.batch["advantages"] = gen_batch.batch["advantages"] * adv_weight
            metrics["adv_weight"] = adv_weight

        return gen_batch, metrics

    def _update_actor_on_prepared_trajectories(
        self, gen_batch: DataProto, role: str, timing_raw: Dict
    ) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}

        if self.config.trainer.balance_batch and not gen_batch.meta_info.get("_coevokg_balanced_for_update", False):
            self._balance_batch(gen_batch, metrics=metrics)

        gen_batch.meta_info["global_token_num"] = torch.sum(gen_batch.batch["attention_mask"], dim=-1).tolist()
        if self.config.trainer.critic_warmup <= self.global_steps:
            with marked_timer(f"{role}_update_actor", timing_raw, color="red"):
                gen_batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                gen_batch.meta_info["kl_loss_coef"] = self._get_current_kl_loss_coef()
                actor_output = self.actor_rollout_wg.update_actor(gen_batch)
            actor_metrics = reduce_metrics(actor_output.meta_info["metrics"])
            observed_kl = actor_metrics.get("actor/kl_loss", None)
            metrics.update(self._update_dynamic_kl_loss_coef(observed_kl))
            metrics.update({f"{role}_actor_{k}": v for k, v in actor_metrics.items()})

        return metrics

    def _reward_summary_metrics(self, gen_batch: DataProto, role: str) -> Dict[str, Any]:
        scores = gen_batch.batch["token_level_scores"].sum(-1).cpu().tolist()
        return {
            f"{role}_reward_mean": np.mean(scores),
            f"{role}_reward_std": np.std(scores),
            f"{role}_reward_max": np.max(scores),
            f"{role}_reward_min": np.min(scores),
        }

    def _concat_unprepared_update_batches(self, batches: List[DataProto], role: str) -> DataProto:
        """Concat same-role batches before prefill/advantage with defensive key alignment."""
        if not batches:
            raise ValueError(f"No {role} batches to concat")
        if len(batches) == 1:
            return batches[0]

        common_tensor_keys = set(batches[0].batch.keys())
        common_non_tensor_keys = set(batches[0].non_tensor_batch.keys())
        for batch in batches[1:]:
            common_tensor_keys &= set(batch.batch.keys())
            common_non_tensor_keys &= set(batch.non_tensor_batch.keys())

        required_tensor_keys = {
            "input_ids",
            "attention_mask",
            "position_ids",
            "responses",
            "response_mask",
            "token_level_scores",
        }
        missing = required_tensor_keys - common_tensor_keys
        if missing:
            raise ValueError(
                f"Cannot concat {role} update batches before advantage, missing tensor keys: {sorted(missing)}"
            )

        dropped_tensor_keys = set()
        dropped_non_tensor_keys = set()
        for batch in batches:
            for key in list(batch.batch.keys()):
                if key not in common_tensor_keys:
                    dropped_tensor_keys.add(key)
                    del batch.batch[key]
            for key in list(batch.non_tensor_batch.keys()):
                if key not in common_non_tensor_keys:
                    dropped_non_tensor_keys.add(key)
                    del batch.non_tensor_batch[key]

        if dropped_tensor_keys or dropped_non_tensor_keys:
            logger.warning(
                f"Dropped non-common keys before concat {role} batches: "
                f"tensor={sorted(dropped_tensor_keys)}, non_tensor={sorted(dropped_non_tensor_keys)}"
            )

        combined_batch = DataProto.concat(batches)
        combined_batch.meta_info["_coevokg_balanced_for_update"] = False
        return combined_batch

    def _concat_prepared_update_batches(self, batches: List[DataProto]) -> DataProto:
        """Concat role-prepared batches after dropping role-only temporary fields."""
        if not batches:
            raise ValueError("No prepared batches to combine")
        if len(batches) == 1:
            return batches[0]

        common_tensor_keys = set(batches[0].batch.keys())
        common_non_tensor_keys = set(batches[0].non_tensor_batch.keys())
        for batch in batches[1:]:
            common_tensor_keys &= set(batch.batch.keys())
            common_non_tensor_keys &= set(batch.non_tensor_batch.keys())

        required_tensor_keys = {
            "input_ids",
            "attention_mask",
            "position_ids",
            "responses",
            "response_mask",
            "old_log_probs",
            "advantages",
        }
        if self.config.actor_rollout_ref.actor.use_kl_loss:
            required_tensor_keys.add("ref_log_prob")
        missing = required_tensor_keys - common_tensor_keys
        if missing:
            raise ValueError(f"Cannot combine prepared update batches, missing tensor keys: {sorted(missing)}")

        for batch in batches:
            for key in list(batch.batch.keys()):
                if key not in common_tensor_keys:
                    del batch.batch[key]
            for key in list(batch.non_tensor_batch.keys()):
                if key not in common_non_tensor_keys:
                    del batch.non_tensor_batch[key]

        combined_batch = DataProto.concat(batches)
        combined_batch.meta_info["_coevokg_balanced_for_update"] = False
        return combined_batch

    @staticmethod
    def _parse_response_turns(response_text: str) -> List[Dict]:
        """
        Parse a multi-turn response string into a list of structured turns.

        The format (qwen2p5 template, skip_special_tokens=True) uses:
          - <think>...</think>       : model reasoning
          - <search>...</search>     : search query issued by model
          - <information>...</information> : search result (injected by tool)
          - <answer>...</answer>     : final answer

        Tool turns are identified by <information> blocks; everything between
        two consecutive information blocks (or from start / to end) is an
        assistant turn.
        """
        import re
        turns: List[Dict] = []

        # Split on information blocks to interleave assistant and tool turns
        info_pattern = re.compile(r'(<information>.*?</information>)', re.DOTALL)
        segments = info_pattern.split(response_text)

        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue
            if seg.startswith('<information>'):
                content = re.sub(r'<information>(.*?)</information>', r'\1', seg, flags=re.DOTALL).strip()
                turns.append({"role": "tool", "content": content})
            else:
                # Assistant segment — may contain think / search / answer
                think_m = re.search(r'<think>(.*?)</think>', seg, re.DOTALL)
                search_m = re.search(r'<search>(.*?)</search>', seg, re.DOTALL)
                answer_m = re.search(r'<answer>(.*?)</answer>', seg, re.DOTALL)
                turn: Dict = {"role": "assistant"}
                if think_m:
                    turn["think"] = think_m.group(1).strip()
                if search_m:
                    turn["search"] = search_m.group(1).strip()
                if answer_m:
                    turn["answer"] = answer_m.group(1).strip()
                if not turn.get("think") and not turn.get("search") and not turn.get("answer"):
                    # Raw content (proposer single-turn or unparseable)
                    turn["content"] = seg
                turns.append(turn)

        return turns

    def _dump_trajectories(self, inputs, outputs, scores, role: str, step: int, reward_extra_infos_dict=None):
        """
        Save proposer/solver trajectories in structured JSONL format.

        Each line is a JSON object with:
          step, role, idx, score, is_dummy, timestamp
          + role-specific metadata (question/chain/ground_truth)
          + turns: list of {"role": "user/assistant/tool", ...} dicts
        """
        try:
            base_dir = self.config.trainer.get("rollout_data_dir", "/tmp/coevokg_rollout")
            role_dir = os.path.join(base_dir, role)
            os.makedirs(role_dir, exist_ok=True)

            filename = f"{role}_step_{step:06d}.jsonl"
            filepath = os.path.join(role_dir, filename)

            n = len(inputs)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            lines = []
            for i in range(n):
                # ── Base fields ───────────────────────────────────────────
                entry: Dict = {
                    "step": step,
                    "role": role,
                    "idx": i,
                    "score": scores[i] if i < len(scores) else None,
                    "timestamp": ts,
                }

                # ── Metadata from non_tensor_batch ─────────────────────
                extra_info = {}
                reward_model = {}
                data_source = ""
                uid = ""
                if reward_extra_infos_dict:
                    if "extra_info" in reward_extra_infos_dict and i < len(reward_extra_infos_dict["extra_info"]):
                        raw = reward_extra_infos_dict["extra_info"][i]
                        extra_info = raw if isinstance(raw, dict) else {}
                    if "reward_model" in reward_extra_infos_dict and i < len(reward_extra_infos_dict["reward_model"]):
                        raw = reward_extra_infos_dict["reward_model"][i]
                        reward_model = raw if isinstance(raw, dict) else {}
                    if "data_source" in reward_extra_infos_dict and i < len(reward_extra_infos_dict["data_source"]):
                        data_source = str(reward_extra_infos_dict["data_source"][i])
                    if "uid" in reward_extra_infos_dict and i < len(reward_extra_infos_dict["uid"]):
                        uid = str(reward_extra_infos_dict["uid"][i])

                entry["data_source"] = data_source
                entry["uid"] = uid
                entry["is_dummy"] = data_source == "dummy"

                gt = ""
                if isinstance(reward_model.get("ground_truth"), dict):
                    gt = reward_model["ground_truth"].get("target", "")
                elif "ground_truth" in reward_model:
                    gt = str(reward_model["ground_truth"])
                entry["ground_truth"] = gt

                # ── Role-specific metadata ────────────────────────────
                if role == "proposer":
                    entry["chain"] = extra_info.get("chain_relations", [])
                    entry["selected_hops"] = extra_info.get("selected_hops", None)
                    entry["intermediate_entities"] = extra_info.get("intermediate_entities", [])
                elif role == "solver":
                    entry["question"] = extra_info.get("question", extra_info.get("extracted_question", ""))
                    entry["intermediate_entities"] = extra_info.get("intermediate_entities", [])

                # ── Parse multi-turn response into structured turns ───
                user_turn = {"role": "user", "content": inputs[i] if i < len(inputs) else ""}
                response_turns = self._parse_response_turns(outputs[i] if i < len(outputs) else "")

                # Extract surface-level info for quick inspection
                all_searches = [t["search"] for t in response_turns if t.get("search")]
                final_answer = next((t["answer"] for t in reversed(response_turns) if t.get("answer")), None)

                entry["num_turns"] = len(response_turns)
                entry["num_searches"] = len(all_searches)
                entry["searches"] = all_searches
                entry["final_answer"] = final_answer
                entry["turns"] = [user_turn] + response_turns

                lines.append(json.dumps(entry, ensure_ascii=False, default=str))

            with open(filepath, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")

            print(f"Dumped {n} {role} trajectories → {filepath} "
                  f"(avg_turns={sum(json.loads(l)['num_turns'] for l in lines)/max(n,1):.1f})")

        except Exception as e:
            logger.warning(f"Failed to dump {role} trajectories for step {step}: {e}")

    def _prepare_solving_batch_from_data(self, solving_data: List[Dict]) -> DataProto:
        assert solving_data is not None, "No solving data available"

        import os
        import tempfile

        import pandas as pd

        df = pd.DataFrame(solving_data)

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            temp_file = f.name
            df.to_parquet(temp_file, index=False)

        try:
            from verl.utils.dataset.rl_dataset import RLHFDataset

            if SELF_PLAY_DEBUG:
                logger.debug(f"Input solving_data length: {len(solving_data)}")

            temp_config = deepcopy(self.config.data)
            temp_config.filter_overlong_prompts = False

            temp_dataset = RLHFDataset(
                data_files=[temp_file], tokenizer=self.tokenizer, config=temp_config, processor=self.processor
            )

            if SELF_PLAY_DEBUG:
                logger.debug(f"RLHFDataset length after creation: {len(temp_dataset)}")

            from torchdata.stateful_dataloader import StatefulDataLoader
            from verl.utils.dataset.rl_dataset import collate_fn

            temp_dataloader = StatefulDataLoader(
                dataset=temp_dataset,
                batch_size=len(solving_data),
                num_workers=0,
                drop_last=False,
                collate_fn=collate_fn,
                shuffle=False,
            )

            batch_dict = next(iter(temp_dataloader))
            batch = DataProto.from_single_dict(batch_dict)

            if SELF_PLAY_DEBUG:
                logger.debug(f"Final batch size after processing: {len(batch)}")

            for key in batch.batch.keys():
                if key not in [
                    "old_log_probs",
                    "ref_log_prob",
                    "token_level_scores",
                    "token_level_rewards",
                    "advantages",
                    "returns",
                    "values",
                ]:
                    batch.batch[key] = batch.batch[key].long()

            return batch

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def _extract_trajectories_from_batch(self, batch: DataProto) -> List[Dict[str, Any]]:
        trajectories = []

        assert "prompts" in batch.batch and "responses" in batch.batch, "prompts and responses must be in batch"

        if SELF_PLAY_DEBUG:
            logger.debug(f"Extracting trajectories from batch with {len(batch.batch['prompts'])} samples")
            logger.debug(f"Batch keys: {list(batch.batch.keys())}")
            if hasattr(batch, "non_tensor_batch") and batch.non_tensor_batch:
                logger.debug(f"Non-tensor batch keys: {list(batch.non_tensor_batch.keys())}")

        input_ids = batch.batch["prompts"]
        output_ids = batch.batch["responses"]

        input_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        output_texts = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        for i, (input_text, output_text) in enumerate(zip(input_texts, output_texts)):
            metadata = {"trajectory_index": i, "global_step": self.global_steps, "phase": self.current_phase}

            if hasattr(batch, "non_tensor_batch") and batch.non_tensor_batch is not None:
                if "reward_model" in batch.non_tensor_batch and i < len(batch.non_tensor_batch["reward_model"]):
                    metadata["reward_model"] = batch.non_tensor_batch["reward_model"][i]
                    if SELF_PLAY_DEBUG:
                        logger.debug(f"Trajectory {i}: reward_model = {metadata['reward_model']}")

                if "data_source" in batch.non_tensor_batch and i < len(batch.non_tensor_batch["data_source"]):
                    metadata["data_source"] = batch.non_tensor_batch["data_source"][i]
                    if SELF_PLAY_DEBUG:
                        logger.debug(f"Trajectory {i}: data_source = {metadata['data_source']}")

                if "extra_info" in batch.non_tensor_batch and i < len(batch.non_tensor_batch["extra_info"]):
                    metadata["extra_info"] = batch.non_tensor_batch["extra_info"][i]
                    if SELF_PLAY_DEBUG:
                        logger.debug(f"Trajectory {i}: extra_info = {metadata['extra_info']}")

            trajectory = {
                "input": input_text,
                "output": output_text,
                "metadata": metadata,
            }
            trajectories.append(trajectory)

            if SELF_PLAY_DEBUG:
                logger.debug(f"Trajectory {i}: input[:100] = {input_text[:100]}...")
                logger.debug(f"Trajectory {i}: output[:200] = {output_text[:200]}...")
                logger.debug(f"Trajectory {i}: metadata = {metadata}")

        if SELF_PLAY_DEBUG:
            logger.debug(f"Extracted {len(trajectories)} trajectories total")

        return trajectories

    def _extract_and_process_problems(self, trajectories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if SELF_PLAY_DEBUG:
            logger.debug(f"Starting problem extraction from {len(trajectories)} trajectories")
            for i, traj in enumerate(trajectories[:3]):
                logger.debug(f"Trajectory {i} output for extraction: {traj['output'][:500]}...")

        ray.get(self.sp_data_manager.reset_current_step_stats.remote())

        start_time = time.time()
        extracted_problems, extraction_stats = extract_problems_batch(trajectories, self.problem_extractor)
        end_time = time.time()
        if SELF_PLAY_DEBUG:
            logger.debug(f"Extracted {len(extracted_problems)} problems in {end_time - start_time} seconds")
            logger.debug(f"Extraction statistics: {extraction_stats}")
            for i, problem in enumerate(extracted_problems[:3]):
                logger.debug(f"Raw problem {i}: {problem}")

        valid_problems = []
        for i, problem in enumerate(extracted_problems):
            if self._validate_extracted_problem(problem):
                valid_problems.append(problem)
                if SELF_PLAY_DEBUG:
                    logger.debug(f"Problem {i} is VALID: {problem.get('extracted_question', '')[:100]}...")
            else:
                if SELF_PLAY_DEBUG:
                    logger.debug(f"Problem {i} is INVALID: {problem}")

        print(f"Extracted {len(extracted_problems)} problems, {len(valid_problems)} valid")

        ray.get(
            self.sp_data_manager.record_extraction_stats.remote(
                trajectories_count=extraction_stats["trajectories_count"],
                answer_matches_count=extraction_stats["answer_matches_count"],
                valid_questions_count=extraction_stats["valid_questions_count"],
                format_error_count=extraction_stats["format_error_count"],
                successful_problems_count=len(valid_problems),
            )
        )

        if SELF_PLAY_DEBUG:
            logger.debug(f"Final valid problems count: {len(valid_problems)}")
            for i, problem in enumerate(valid_problems[:2]):  # Debug first 2 valid problems
                logger.debug(f"Valid problem {i}: {problem}")

        return valid_problems

    def _build_seed_question_index(self):
        """Build a shuffled circular fallback pool from data.train_files.

        When extraction fails the solver gets a randomly sampled seed question
        instead of a dummy placeholder.  A circular pointer advances through the
        shuffled list so each question is used at most once per pass (epoch-level
        deduplication); when the pool is exhausted it reshuffles and restarts.
        """
        self._seed_question_index: Dict[str, Dict] = {}
        self._seed_fallback_list: List[Dict] = []
        self._seed_fallback_ptr: int = 0
        try:
            import ast
            import random as _random
            import pandas as pd

            train_files = self.config.data.get("train_files", [])
            if isinstance(train_files, str):
                train_files = [train_files]

            loaded = 0
            all_rows: List[Dict] = []
            for path in train_files:
                if not path or not os.path.exists(path):
                    continue
                df = pd.read_parquet(path) if path.endswith(".parquet") else None
                if df is None:
                    continue
                for _, row in df.iterrows():
                    rm = row.get("reward_model", {})
                    if isinstance(rm, str):
                        try:
                            rm = ast.literal_eval(rm)
                        except Exception:
                            continue
                    target = rm.get("ground_truth", {}).get("target", "") if isinstance(rm, dict) else ""
                    if not target:
                        continue
                    row_dict = row.to_dict()
                    key = target.lower().strip()
                    if key not in self._seed_question_index:
                        self._seed_question_index[key] = row_dict
                        loaded += 1
                    all_rows.append(row_dict)

            _random.shuffle(all_rows)
            self._seed_fallback_list = all_rows
            self._seed_fallback_ptr = 0

            logger.info(
                f"[seed_question_index] Built index with {loaded} unique answers "
                f"from {len(train_files)} train file(s); "
                f"fallback pool: {len(all_rows)} records (shuffled)"
            )
        except Exception as e:
            logger.warning(f"[seed_question_index] Failed to build index: {e}; fallback will use dummy")
            self._seed_question_index = {}
            self._seed_fallback_list = []
            self._seed_fallback_ptr = 0

    def _create_fallback_problem_for_failed_extraction(
        self, trajectory: Dict[str, Any], trajectory_index: int
    ) -> Tuple[Dict[str, Any], bool]:
        """Extraction failed: pop the next seed question from the shuffled circular pool.

        Advances a pointer through _seed_fallback_list so each question is used at
        most once per pass (epoch-level deduplication).  When the pool is exhausted
        it reshuffles and restarts.  Falls back to a dummy placeholder only when the
        pool is completely empty.
        """
        feedback_problem = self._sample_feedback_pool_problem(trajectory_index)
        if feedback_problem is not None:
            return feedback_problem, True

        if not self._seed_fallback_list:
            return self._create_dummy_problem_for_failed_extraction(trajectory, trajectory_index), False

        if self._seed_fallback_ptr >= len(self._seed_fallback_list):
            import random as _random
            _random.shuffle(self._seed_fallback_list)
            self._seed_fallback_ptr = 0
            logger.info("[seed_fallback] Exhausted fallback pool, reshuffling for next pass")

        seed_record = self._seed_fallback_list[self._seed_fallback_ptr]
        self._seed_fallback_ptr += 1

        import ast
        prompt = seed_record.get("prompt", [])
        if isinstance(prompt, str):
            try:
                prompt = ast.literal_eval(prompt)
            except Exception:
                prompt = [{"role": "user", "content": prompt}]

        rm = seed_record.get("reward_model", {})
        if isinstance(rm, str):
            try:
                rm = ast.literal_eval(rm)
            except Exception:
                rm = {"style": "rule", "ground_truth": {"target": ""}}

        ei = seed_record.get("extra_info", {})
        if isinstance(ei, str):
            try:
                ei = ast.literal_eval(ei)
            except Exception:
                ei = {}

        question_text = ei.get("question", rm.get("ground_truth", {}).get("target", ""))
        # Unify with walk_chain: rebuild the prompt with the standard solver system
        # prompt instead of the dataset's baked-in legacy prompt, so seed-fallback
        # trajectories follow the same <think>/<search>/<answer> format and are
        # scored consistently with walk_chain trajectories.
        prompt = [
            {"role": "system", "content": self.problem_extractor.solver_system_prompt},
            {"role": "user", "content": f"Question: {question_text}"},
        ]
        fallback_problem = {
            "data_source": "seed_fallback",
            "prompt": prompt,
            "ability": seed_record.get("ability", "fact-reasoning"),
            "reward_model": rm,
            "extra_info": {
                **ei,
                "question": question_text,
                "need_tools_kwargs": True,
                "split": "train",
                "tools_kwargs": ei.get("tools_kwargs", {
                    "search": {
                        "create_kwargs": {
                            "data_source": "seed_fallback",
                            "question": question_text,
                            "ground_truth": rm.get("ground_truth"),
                        }
                    }
                }),
                "extraction_failed": True,
            },
            "metadata": None,
            "extracted_question": question_text,
            "formatted_prompt": prompt,
            "problem_type": "search",
            "trajectory_index": trajectory_index,
        }
        if SELF_PLAY_DEBUG:
            logger.debug(
                f"Seed fallback (ptr={self._seed_fallback_ptr - 1}) for trajectory {trajectory_index}: "
                f"question={repr(question_text)[:80]}"
            )
        return fallback_problem, True  # reused=True

    def _create_dummy_problem_for_failed_extraction(
        self, trajectory: Dict[str, Any], trajectory_index: int
    ) -> Dict[str, Any]:
        metadata = trajectory.get("metadata", {})

        dummy_prompt = [
            {"role": "system", "content": "You are a helpful and harmless assistant."},
            {
                "role": "user",
                "content": "This is a dummy problem. Please directly output </answer>",
            },
        ]

        dummy_reward_model = metadata.get("reward_model", {"ground_truth": {"target": "dummy_answer", "style": "rule"}})

        dummy_problem = {
            "data_source": "dummy",
            "prompt": dummy_prompt,
            "ability": "fact-reasoning",
            "reward_model": dummy_reward_model,
            "extra_info": {
                "question": "This is a dummy problem. Please directly output </answer>",
                "need_tools_kwargs": True,
                "split": "train",
                "tools_kwargs": {
                    "search": {
                        "create_kwargs": {
                            "data_source": "dummy",
                            "question": "This is a dummy problem. Please directly output </answer>",
                            "ground_truth": dummy_reward_model.get("ground_truth"),
                        }
                    }
                },
                "extraction_failed": True,
            },
            "metadata": None,
            "extracted_question": "This is a dummy problem. Please directly output </answer>",
            "formatted_prompt": dummy_prompt,
            "problem_type": "search",
            "trajectory_index": trajectory_index,
        }

        if SELF_PLAY_DEBUG:
            logger.debug(f"Created dummy problem for failed extraction at trajectory {trajectory_index}")

        return dummy_problem

    def _validate_extracted_problem(self, problem: Dict[str, Any]) -> bool:
        question = problem["extracted_question"]
        if len(question.strip()) < 10:
            return False

        answer = problem.get("reward_model", {}).get("ground_truth", {}).get("target", None)
        if answer in question:
            return False

        return True

    def _balance_batch(self, batch: DataProto, metrics):
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()
        world_size = self.actor_rollout_wg.world_size

        from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance

        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix="global_seqlen"
        )
        metrics.update(global_balance_stats)
