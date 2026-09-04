import logging
import os
from functools import partial
from pprint import pprint

import hydra
import ray
from verl.trainer.ppo.reward import get_custom_reward_fn
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.fs import copy_to_local


def setup_logging():
    log_level_str = os.getenv("LOG_LEVEL", "WARNING").upper()
    log_level = getattr(logging, log_level_str, logging.WARNING)

    logging.getLogger().setLevel(log_level)

    logging.getLogger("coevokg").setLevel(log_level)
    logging.getLogger("verl").setLevel(log_level)

    for logger_name in [
        "verl.workers.rollout.sglang_rollout",
    ]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)
        logger.propagate = True

    print(f"Logging level set to: {log_level_str} ({log_level})")

setup_logging()


@hydra.main(config_path="config", config_name="rl_config", version_base=None)
def main(config):
    run_ppo(config)


def init_ray(config) -> None:
    """
    Initialize Ray with an explicit policy. Supported modes:

      connect  -- connect to an existing cluster (address="auto" or explicit address).
      local    -- start a local embedded cluster for single-node debugging.
      auto     -- prefer an existing cluster and raise an actionable error on failure.

    Address priority: config.ray_init.address > env RAY_ADDRESS > "auto".

    Do not pass runtime_env when connecting to an existing cluster; this avoids
    silent hangs during GCS uploads. Launch scripts should export env_vars
    such as TOKENIZERS_PARALLELISM, NCCL_DEBUG, and VLLM_LOGGING_LEVEL before
    ray start so workers inherit them automatically.
    """
    if ray.is_initialized():
        return

    ray_cfg = config.get("ray_init", {})
    mode = ray_cfg.get("mode", "auto")
    explicit_address = ray_cfg.get("address", None) or os.getenv("RAY_ADDRESS")

    print(f"[Ray] init mode={mode}, explicit_address={explicit_address}")

    if mode == "connect":
        address = explicit_address or "auto"
        print(f"[Ray] Connecting to existing cluster: {address}")
        ray.init(address=address)

    elif mode == "local":
        num_cpus = ray_cfg.get("num_cpus", None)
        print(f"[Ray] Starting local embedded cluster (num_cpus={num_cpus})")
        init_kwargs = {}
        if num_cpus is not None:
            init_kwargs["num_cpus"] = num_cpus
        ray.init(**init_kwargs)

    else:  # auto by default
        address = explicit_address or "auto"
        print(f"[Ray] Auto mode: connecting to existing cluster: {address}")
        try:
            ray.init(address=address)
        except Exception as e:
            raise RuntimeError(
                f"[Ray] Failed to connect to existing Ray cluster (address={address}). "
                "If you want a local embedded cluster, set ray_init.mode=local explicitly."
            ) from e

    print(f"[Ray] Initialized. Resources: {ray.cluster_resources()}")


def run_ppo(config) -> None:
    init_ray(config)

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def run(self, config):
        # print initial config

        from omegaconf import OmegaConf

        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        # instantiate tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # used for multimodal LLM, could be none
        processor = hf_processor(local_path, use_fast=True)

        if config.coevokg.task_type == "coevokg_deep_search":
            from coevokg.utils.patch.metric_patch import apply_metric_patch

            apply_metric_patch(tokenizer)

        # cls should be imported after apply_metric_patch
        # from verl.trainer.ppo.ray_trainer import RayPPOTrainer

        # Check if self-play is enabled
        use_self_play = config.get("self_play", {}).get("enable", False)
        if use_self_play:
            from coevokg.trainer.ppo.coevokg_ray_trainer import CoEvoKGRayPPOTrainer as TrainerClass

            print("Using Self-Play PPO Trainer")
        else:
            from verl.trainer.ppo.ray_trainer import RayPPOTrainer as TrainerClass

            print("Using Standard PPO Trainer")

        # define worker classes
        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            if config.coevokg.task_type == "coevokg_deep_search":
                from coevokg.worker.fsdp_worker import CoEvoKGActorRolloutRefWorker

                actor_rollout_cls = CoEvoKGActorRolloutRefWorker
            elif config.actor_rollout_ref.rollout.mode == "async":
                actor_rollout_cls = AsyncActorRolloutRefWorker
            else:
                actor_rollout_cls = ActorRolloutRefWorker

            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # we should adopt a multi-source reward function here
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # - finally, we combine all the rewards together
        # - The reward type depends on the tag of the data
        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker

            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # use reference model
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        if (
            config.coevokg.task_type == "coevokg_deep_search"
            and config.coevokg.diff_val_reward_fn_config.custom_reward_function.path
        ):
            # use a different reward function for validation
            val_reward_fn = load_reward_manager(config.coevokg.diff_val_reward_fn_config, tokenizer, num_examine=1)
        else:
            val_reward_fn = load_reward_manager(config, tokenizer, num_examine=1)
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        from verl.utils.dataset.rl_dataset import collate_fn

        from torch.utils.data import Subset
        train_dataset = create_rl_dataset(config.data.train_files, config)
        val_dataset = create_rl_dataset(config.data.val_files, config)
        if (n := config.data.get("train_size", None)) is not None and n < len(train_dataset):
            train_dataset = Subset(train_dataset, list(range(n)))
            print(f"Train dataset truncated to {n} samples")
        if (n := config.data.get("val_size", None)) is not None and n < len(val_dataset):
            val_dataset = Subset(val_dataset, list(range(n)))
            print(f"Val dataset truncated to {n} samples")
        train_sampler = create_rl_sampler(config.data, train_dataset)
        trainer = TrainerClass(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()
        trainer.fit()


def create_rl_dataset(data_paths, config, tokenizer=None, processor=None):
    """Create a dataset.

    Arguments:
        config.data: The data config.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.

    Returns:
        dataset (Dataset): The dataset.
    """
    from torch.utils.data import Dataset
    from verl.utils import hf_processor, hf_tokenizer
    from verl.utils.dataset.rl_dataset import RLHFDataset
    from verl.utils.fs import copy_to_local

    local_path = copy_to_local(config.actor_rollout_ref.model.path)
    trust_remote_code = config.data.get("trust_remote_code", False)

    if tokenizer is None:
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    if processor is None:
        processor = hf_processor(local_path, use_fast=True)

    if config.coevokg.task_type == "coevokg_deep_search":
        from coevokg.utils.chat_template import CoEvoKGChatTemplate as QSCT
        from coevokg.utils.patch.chat_template import switch_apply_chat_template

        print("[create_rl_dataset]: CoEvoKGChatTemplate._current_template =", QSCT._current_template)

        tokenizer = switch_apply_chat_template(tokenizer, QSCT.apply_chat_template)
        processor = switch_apply_chat_template(processor, QSCT.apply_chat_template)

    dataset_cls = RLHFDataset

    print(f"Using dataset class: {dataset_cls.__name__}")

    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=config.data,
    )

    return dataset


def load_reward_manager(config, tokenizer, num_examine, **reward_kwargs):
    reward_manager_name = config.reward_model.get("reward_manager", "naive")
    print(f"Loading reward manager: {reward_manager_name}")
    if reward_manager_name == "naive":
        from verl.workers.reward_manager import NaiveRewardManager

        reward_manager_cls = NaiveRewardManager
    elif reward_manager_name == "prime":
        from verl.workers.reward_manager import PrimeRewardManager

        reward_manager_cls = PrimeRewardManager
    elif reward_manager_name == "batch":
        from verl.workers.reward_manager import BatchRewardManager

        reward_manager_cls = BatchRewardManager
    elif reward_manager_name == "dapo":
        from verl.workers.reward_manager import DAPORewardManager

        reward_manager_cls = DAPORewardManager

    elif reward_manager_name == "coevokg":
        from coevokg.reward import get_custom_reward_fns
        from coevokg.reward.manager import CoEvoKGRewardManager

        return CoEvoKGRewardManager(
            tokenizer=tokenizer,
            num_examine=num_examine,
            reward_fns=get_custom_reward_fns(config),
            reward_fn_key=config.data.reward_fn_key,
            save_records=False,
            save_path=config.reward_model.get("save_path", None),
        )

    elif reward_manager_name == "naive_with_prompt":
        from coevokg.reward.manager import NaiveRewardManagerComputeWithPrompt

        reward_manager_cls = NaiveRewardManagerComputeWithPrompt

    else:
        raise NotImplementedError

    compute_score = get_custom_reward_fn(config)

    if reward_manager_name == "naive_with_prompt":

        return reward_manager_cls(
            tokenizer=tokenizer,
            num_examine=num_examine,
            compute_score=compute_score,
            reward_fn_key=config.data.reward_fn_key,
            save_records=False,
            save_path=config.reward_model.get("save_path", None),
            **reward_kwargs,
        )
    else:
        return reward_manager_cls(
            tokenizer=tokenizer,
            num_examine=num_examine,
            compute_score=compute_score,
            reward_fn_key=config.data.reward_fn_key,
            **reward_kwargs,
        )


def create_rl_sampler(data_config, dataset):
    """Create a sampler for the dataset.

    Arguments:
        data_config: The data config.
        dataset (Dataset): The dataset.

    Returns:
        sampler (Sampler): The sampler.
    """
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    # use sampler for better ckpt resume
    if data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=dataset)

    return sampler


if __name__ == "__main__":
    main()
