import inspect
import os
from collections import defaultdict, deque
from contextlib import nullcontext
from functools import partial
from typing import Any, Optional, Union

import datasets
import torch
import torch.utils.data
import transformers
from accelerate import logging
from accelerate.state import AcceleratorState
from accelerate.utils import (
    broadcast_object_list,
    gather_object,
    is_peft_model,
    set_seed,
)
from datasets import Dataset, IterableDataset
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.functional import kl_div, log_softmax
from torch.utils.data import DataLoader, Sampler
from transformers import (
    AutoConfig,
    AutoProcessor,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
    is_wandb_available,
)
from transformers.trainer_utils import seed_worker
from transformers.utils import (
    is_datasets_available,
    is_flash_attn_2_available,
    is_peft_available,
)
from trl.data_utils import (
    is_conversational,
    maybe_apply_chat_template,
    prepare_multimodal_messages,
)
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.import_utils import is_vllm_available
from trl.models import (
    prepare_deepspeed,
    prepare_fsdp,
    prepare_peft_model,
    unwrap_model_for_generation,
)
from trl.trainer.base_trainer import BaseTrainer
from trl.trainer.utils import (
    RepeatSampler,
    disable_dropout_in_model,
    ensure_master_addr_port,
    entropy_from_logits,
    identity,
    nanmax,
    nanmin,
    pad,
    selective_log_softmax,
    shuffle_sequence_dict,
    split_pixel_values_by_grid,
    split_tensor_dict,
    unsplit_pixel_values_by_grid,
)

from .distil_config import DistilConfig

if is_peft_available():
    from peft import PeftConfig, PeftModel
if is_wandb_available():
    import wandb
logger = logging.get_logger(__name__)


class MemoryEfficientSyncRefModelCallback(TrainerCallback):
    def __init__(
        self, ref_model: Union[PreTrainedModel, nn.Module], accelerator: Optional[Any]
    ):
        self.accelerator = accelerator
        self.ref_model = ref_model

    @staticmethod
    def _sync_param(model_param, ref_param, alpha):
        ref_param.data.mul_(1.0 - alpha).add_(model_param.data, alpha=alpha)

    @staticmethod
    def sync_target_model_memory_efficient(model, target_model, alpha):
        deepspeed_plugin = AcceleratorState().deepspeed_plugin
        is_zero3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if is_zero3:
            import deepspeed

            for (name, model_param), (_, ref_param) in zip(
                model.named_parameters(), target_model.named_parameters()
            ):
                with deepspeed.zero.GatheredParameters(
                    [model_param, ref_param], modifier_rank=0
                ):
                    if deepspeed.comm.get_rank() == 0:
                        MemoryEfficientSyncRefModelCallback._sync_param(
                            model_param, ref_param, alpha
                        )
        else:
            for model_param, ref_param in zip(
                model.parameters(), target_model.parameters()
            ):
                MemoryEfficientSyncRefModelCallback._sync_param(
                    model_param, ref_param, alpha
                )

    def on_step_end(self, args, state, control, **kwargs):
        model: PreTrainedModel = kwargs["model"]
        if (
            self.ref_model is not None
            and state.global_step % args.ref_model_sync_steps == 0
        ):
            if self.accelerator:
                model = self.accelerator.unwrap_model(model)
            self.sync_target_model_memory_efficient(
                model, self.ref_model, args.ref_model_mixup_alpha
            )


class DistilTrainer(BaseTrainer):
    _tag_names = ["trl", "distil"]
    _name = "Distil"

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        ref_model: Union[str, PreTrainedModel],
        args: Optional[DistilConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[
            Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]
        ] = None,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, ProcessorMixin]
        ] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[
            Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]
        ] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
    ):
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = DistilConfig(f"{model_name}-Distil")
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            dtype = model_init_kwargs.get("dtype")
            if isinstance(dtype, torch.dtype) or dtype == "auto" or dtype is None:
                pass
            elif isinstance(dtype, str):
                dtype = getattr(torch, dtype)
                model_init_kwargs["dtype"] = dtype
            else:
                raise ValueError(
                    f"Invalid `dtype` passed to `DistilConfig`. Expected either 'auto' or a string representing a `torch.dtype` (e.g., 'float32'), but got {dtype}."
                )
            config = AutoConfig.from_pretrained(model_id)
            architecture = getattr(transformers, config.architectures[0])
            model = architecture.from_pretrained(model_id, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                logger.warning(
                    "You passed `model_init_kwargs` to the `DistilConfig`, but your model is already instantiated. The `model_init_kwargs` will be ignored."
                )
        self.model_kwarg_keys = (
            inspect.signature(model.forward).parameters.keys()
            if not hasattr(model, "get_base_model")
            else inspect.signature(model.get_base_model().forward).parameters.keys()
        )
        if peft_config is not None or (
            is_peft_available() and isinstance(model, PeftModel)
        ):
            model = prepare_peft_model(model, peft_config, args)
        if processing_class is None:
            processing_class = AutoProcessor.from_pretrained(
                model.config._name_or_path, truncation_side="left"
            )
        if isinstance(processing_class, ProcessorMixin):
            tokenizer = processing_class.tokenizer
        elif isinstance(processing_class, PreTrainedTokenizerBase):
            tokenizer = processing_class
        else:
            raise TypeError(
                "The `processing_class` must be either a `PreTrainedTokenizerBase` or a `ProcessorMixin`"
            )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.pad_token = tokenizer.pad_token
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.min_p = args.min_p
        self.repetition_penalty = args.repetition_penalty
        self.use_transformers_paged = args.use_transformers_paged
        self.use_vllm = args.use_vllm
        self.vllm_mode = args.vllm_mode
        self.vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization
        self.vllm_tensor_parallel_size = args.vllm_tensor_parallel_size
        self.vllm_importance_sampling_correction = (
            args.vllm_importance_sampling_correction
        )
        self.vllm_importance_sampling_cap = args.vllm_importance_sampling_cap
        self.mask_truncated_completions = args.mask_truncated_completions
        self.top_entropy_quantile = args.top_entropy_quantile
        self.num_loss_tokens_to_skip = args.num_loss_tokens_to_skip
        self.use_safety_weighting = args.use_safety_weighting
        self.safety_weighting_top_k = args.safety_weighting_top_k
        self.dpsw_top_k = args.dpsw_top_k
        self.shuffle_dataset = args.shuffle_dataset
        if (
            isinstance(train_dataset, IterableDataset)
            or isinstance(eval_dataset, IterableDataset)
            or (
                isinstance(eval_dataset, dict)
                and any(
                    (isinstance(ds, IterableDataset) for ds in eval_dataset.values())
                )
            )
        ):
            raise NotImplementedError(
                "Iterable datasets are not yet supported in DistilTrainer. Please use a standard dataset instead."
            )
        self.num_iterations = args.num_iterations
        self._step = 0
        self._buffered_inputs = None
        model.warnings_issued["estimate_tokens"] = True
        super().__init__(
            model=model,
            args=args,
            data_collator=identity,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            compute_loss_func="non-None value to disable scaling",
        )
        self.alpha = args.alpha
        self.generate_from_teacher = args.generate_from_teacher
        if ref_model is not None:
            self.ref_model = ref_model
            self.ref_model.requires_grad_(False)
            self.ref_model.eval()
        elif is_peft_model(model):
            self.ref_model = None
        else:
            config = AutoConfig.from_pretrained(model_id)
            architecture = getattr(transformers, config.architectures[0])
            self.ref_model = architecture.from_pretrained(model_id, **model_init_kwargs)
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self.log_completions = args.log_completions
        self.wandb_log_unique_prompts = args.wandb_log_unique_prompts
        self._logs = {
            "images": deque(maxlen=args.generation_batch_size),
            "prompt": deque(maxlen=args.generation_batch_size),
            "completion": deque(maxlen=args.generation_batch_size),
            "safety_weights": deque(maxlen=args.generation_batch_size),
        }
        set_seed(args.seed, device_specific=True)
        if self.use_vllm:
            from vllm import LLM

            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with `pip install trl[vllm]` to use it."
                )
            if self.vllm_mode == "server":
                from trl.extras.vllm_client import VLLMClient

                if self.accelerator.is_main_process:
                    if args.vllm_server_base_url is not None:
                        base_url = args.vllm_server_base_url
                    else:
                        base_url = (
                            f"http://{args.vllm_server_host}:{args.vllm_server_port}"
                        )
                    self.vllm_client = VLLMClient(
                        base_url=base_url, connection_timeout=args.vllm_server_timeout
                    )
                    self.vllm_client.init_communicator(
                        device=torch.cuda.current_device()
                    )
            elif self.vllm_mode == "colocate":
                if (
                    not self.accelerator.num_processes % self.vllm_tensor_parallel_size
                    == 0
                ):
                    raise ValueError(
                        f"vllm_tensor_parallel_size ({self.vllm_tensor_parallel_size}) must divide world size ({self.accelerator.num_processes}) evenly."
                    )
                if self.vllm_tensor_parallel_size > 1:
                    self.tp_group, _ = torch.distributed.new_subgroups_by_enumeration(
                        [
                            list(
                                range(
                                    i * self.vllm_tensor_parallel_size,
                                    (i + 1) * self.vllm_tensor_parallel_size,
                                )
                            )
                            for i in range(
                                self.accelerator.num_processes
                                // self.vllm_tensor_parallel_size
                            )
                        ]
                    )
                os.environ["RANK"] = str(self.accelerator.process_index)
                os.environ["LOCAL_RANK"] = str(self.accelerator.local_process_index)
                os.environ["WORLD_SIZE"] = str(self.accelerator.num_processes)
                ensure_master_addr_port()
                if (
                    self.max_prompt_length is not None
                    and self.max_completion_length is not None
                ):
                    max_model_len = self.max_prompt_length + self.max_completion_length
                else:
                    max_model_len = None
                vllm_model_path = (
                    ref_model.name_or_path
                    if self.generate_from_teacher and ref_model is not None
                    else model.name_or_path
                )
                self.llm = LLM(
                    model=vllm_model_path,
                    tensor_parallel_size=args.vllm_tensor_parallel_size,
                    gpu_memory_utilization=self.vllm_gpu_memory_utilization,
                    max_num_seqs=self.args.per_device_train_batch_size
                    * self.vllm_tensor_parallel_size
                    * self.args.steps_per_generation,
                    max_model_len=max_model_len,
                    distributed_executor_backend="external_launcher",
                    seed=self.accelerator.process_index
                    // self.vllm_tensor_parallel_size,
                    max_num_batched_tokens=4096,
                    model_impl=self.args.vllm_model_impl,
                    enable_sleep_mode=self.args.vllm_enable_sleep_mode,
                    logprobs_mode="processed_logprobs",
                )
                if self.args.vllm_enable_sleep_mode:
                    self.llm.sleep(level=1)
            else:
                raise ValueError(
                    f"vllm_mode must be either 'server' or 'colocate', got '{self.vllm_mode}'."
                )
            self._last_loaded_step = -1
            self.accelerator.wait_for_everyone()
        else:
            generation_kwargs = {
                "max_new_tokens": self.max_completion_length,
                "do_sample": True,
                "pad_token_id": tokenizer.pad_token_id,
                "bos_token_id": tokenizer.bos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "min_p": self.min_p,
                "repetition_penalty": self.repetition_penalty,
                "cache_implementation": args.cache_implementation,
            }
            if args.generation_kwargs is not None:
                generation_kwargs.update(args.generation_kwargs)
            self.generation_config = GenerationConfig(**generation_kwargs)
        self.model_accepts_loss_kwargs = False
        self.model.add_model_tags(self._tag_names)
        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            elif self.is_fsdp_enabled:
                self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(
                    self.ref_model, evaluation_mode=True
                )
        if args.sync_ref_model:
            self.add_callback(
                MemoryEfficientSyncRefModelCallback(
                    ref_model=self.ref_model, accelerator=self.accelerator
                )
            )

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt", "teacher_prompt", "image", "images"]

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(
                train_dataset, description="training"
            )
        else:
            data_collator = self._get_collator_with_removed_columns(
                data_collator, description="training"
            )
        dataloader_params = {
            "batch_size": self._train_batch_size * self.args.steps_per_generation,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }
        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = partial(
                seed_worker,
                num_workers=self.args.dataloader_num_workers,
                rank=self.args.process_index,
            )
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def _get_train_sampler(self, dataset: Optional[Dataset] = None) -> Sampler:
        if dataset is None:
            dataset = self.train_dataset
        return RepeatSampler(
            data_source=dataset,
            mini_repeat_count=self.num_generations,
            batch_size=self.args.generation_batch_size // self.num_generations,
            repeat_count=self.num_iterations * self.args.steps_per_generation,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        return RepeatSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations,
            seed=self.args.seed,
        )

    def get_high_entropy_mask(
        self, entropies: torch.Tensor, mask: torch.Tensor, threshold: float
    ) -> torch.Tensor:
        local = entropies[mask.bool()].float()
        pad_value = -1000000000.0
        padded = self.accelerator.pad_across_processes(
            local, dim=0, pad_index=pad_value
        )
        gathered = self.accelerator.gather(padded)
        gathered = gathered[gathered != pad_value]
        if gathered.numel() == 0:
            return torch.zeros_like(entropies, dtype=torch.bool)
        entropy_threshold = torch.quantile(gathered, threshold)
        masked_entropies = entropies * mask.float()
        entropy_mask = masked_entropies >= entropy_threshold
        return entropy_mask & mask.bool()

    @profiling_decorator
    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        pixel_values=None,
        image_grid_thw=None,
        num_images=None,
        pixel_attention_mask=None,
        image_sizes=None,
        token_type_ids=None,
        compute_all_logps=True,
    ):
        batch_size = batch_size or input_ids.size(0)
        all_selected_logps = []
        all_logps = []
        all_entropies = []
        for start in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[start : start + batch_size]
            attention_mask_batch = attention_mask[start : start + batch_size]
            model_inputs = {
                "input_ids": input_ids_batch,
                "attention_mask": attention_mask_batch,
            }
            if image_grid_thw is not None and pixel_values is not None:
                rows_per_image = image_grid_thw.prod(dim=-1)
                rows_per_sample = torch.split(rows_per_image, num_images)
                rows_per_sample = torch.stack([s.sum() for s in rows_per_sample])
                cum_rows = torch.cat(
                    [
                        torch.tensor([0], device=rows_per_sample.device),
                        rows_per_sample.cumsum(0),
                    ]
                )
                row_start, row_end = (
                    cum_rows[start].item(),
                    cum_rows[start + batch_size].item(),
                )
                model_inputs["pixel_values"] = pixel_values[row_start:row_end]
                cum_imgs = torch.tensor([0] + num_images).cumsum(0)
                img_start, img_end = (cum_imgs[start], cum_imgs[start + batch_size])
                model_inputs["image_grid_thw"] = image_grid_thw[img_start:img_end]
            elif pixel_values is not None:
                model_inputs["pixel_values"] = pixel_values[start : start + batch_size]
            if pixel_attention_mask is not None:
                model_inputs["pixel_attention_mask"] = pixel_attention_mask[
                    start : start + batch_size
                ]
            if image_sizes is not None:
                model_inputs["image_sizes"] = image_sizes[start : start + batch_size]
            if token_type_ids is not None:
                model_inputs["token_type_ids"] = token_type_ids[
                    start : start + batch_size
                ]
            if "logits_to_keep" in self.model_kwarg_keys:
                model_inputs["logits_to_keep"] = logits_to_keep + 1
            model_inputs["use_cache"] = False
            outputs = model(**model_inputs)
            logits = outputs.logits
            logits = logits[:, :-1, :]
            logits = logits[:, -logits_to_keep:, :]
            logits = logits / self.temperature
            completion_ids = input_ids_batch[:, -logits_to_keep:]
            selected_logps = selective_log_softmax(logits, completion_ids)
            if compute_all_logps:
                logps = log_softmax(logits.float(), dim=-1)
            else:
                logps = None
            all_selected_logps.append(selected_logps)
            all_logps.append(logps)
            if compute_entropy:
                with torch.no_grad():
                    entropies = entropy_from_logits(logits)
                all_entropies.append(entropies)
        selected_logps = torch.cat(all_selected_logps, dim=0)
        if compute_all_logps:
            logps = torch.cat(all_logps, dim=0)
        else:
            logps = None
        entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
        return (selected_logps, logps, entropies)

    def _fix_param_name_to_vllm(self, name, extra_prefixes: Optional[list[str]] = None):
        extra_prefixes = extra_prefixes or []
        prefixes = ["_checkpoint_wrapped_module."] + extra_prefixes
        for prefix in prefixes:
            name = name.replace(prefix, "")
        return name

    def _sync_fsdp1_params_to_vllm(
        self, module: nn.Module, prefix: str = "", visited=None
    ):
        if visited is None:
            visited = set()
        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            self._sync_fsdp1_params_to_vllm(
                child_module, prefix=child_prefix, visited=visited
            )
        if isinstance(module, FSDP):
            with FSDP.summon_full_params(module, recurse=False, writeback=False):
                for param_name, param in module.named_parameters():
                    full_name = f"{prefix}.{param_name}" if prefix else param_name
                    full_name = self._fix_param_name_to_vllm(
                        full_name, extra_prefixes=["_fsdp_wrapped_module."]
                    )
                    if full_name in visited:
                        continue
                    visited.add(full_name)
                    if self.vllm_mode == "server" and self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(full_name, param.data)
                    elif self.vllm_mode == "colocate":
                        llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                        llm_model.load_weights([(full_name, param.data)])

    def _sync_fsdp2_params_to_vllm(self, module: nn.Module):
        for name, param in module.state_dict().items():
            if param.is_cpu:
                param = param.to(torch.device("cuda"))
            param = param.full_tensor()
            if self.vllm_mode == "server" and self.accelerator.is_main_process:
                self.vllm_client.update_named_param(name, param)
            elif self.vllm_mode == "colocate":
                llm_model = (
                    self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                )
                llm_model.load_weights([(name, param)])

    @profiling_decorator
    def _move_model_to_vllm(self):
        model_to_sync = self.ref_model if self.generate_from_teacher else self.model
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3:
            import deepspeed

            gather_if_zero3 = deepspeed.zero.GatheredParameters
        else:
            gather_if_zero3 = nullcontext
        if is_peft_model(self.model) and (not self.generate_from_teacher):
            with gather_if_zero3(list(self.model.parameters())):
                self.model.merge_adapter()
                if self.is_fsdp_enabled:
                    fsdp_plugin = getattr(self.accelerator.state, "fsdp_plugin", None)
                    fsdp_version = (
                        getattr(fsdp_plugin, "fsdp_version", 1) if fsdp_plugin else 1
                    )
                    if fsdp_version == 1:
                        self._sync_fsdp1_params_to_vllm(self.model)
                    elif fsdp_version == 2:
                        self._sync_fsdp2_params_to_vllm(self.model)
                else:
                    for name, param in self.model.named_parameters():
                        name = name.removeprefix("base_model.model.").replace(
                            ".base_layer", ""
                        )
                        if self.model.prefix in name:
                            continue
                        if "original_module" in name:
                            continue
                        name = self._fix_param_name_to_vllm(
                            name, extra_prefixes=["modules_to_save.default."]
                        )
                        if (
                            self.vllm_mode == "server"
                            and self.accelerator.is_main_process
                        ):
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                            llm_model.load_weights([(name, param.data)])
                self.model.unmerge_adapter()
        elif self.is_fsdp_enabled:
            fsdp_plugin = getattr(self.accelerator.state, "fsdp_plugin", None)
            fsdp_version = getattr(fsdp_plugin, "fsdp_version", 1) if fsdp_plugin else 1
            if fsdp_version == 1:
                self._sync_fsdp1_params_to_vllm(model_to_sync)
            elif fsdp_version == 2:
                self._sync_fsdp2_params_to_vllm(model_to_sync)
        else:
            for name, param in model_to_sync.named_parameters():
                name = self._fix_param_name_to_vllm(name)
                with gather_if_zero3([param]):
                    if self.vllm_mode == "server" and self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(name, param.data)
                    elif self.vllm_mode == "colocate":
                        llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                        llm_model.load_weights([(name, param.data)])
        if self.vllm_mode == "server" and self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()
        elif self.vllm_mode == "colocate":
            self.llm.reset_prefix_cache()

    @profiling_decorator
    def _prepare_inputs(
        self, generation_batch: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        mode = "train" if self.model.training else "eval"
        if mode == "train":
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                generation_batch = self._generate_and_score_completions(
                    generation_batch
                )
                generation_batch = split_pixel_values_by_grid(generation_batch)
                generation_batch = shuffle_sequence_dict(generation_batch)
                generation_batches = split_tensor_dict(
                    generation_batch, self.args.steps_per_generation
                )
                self._buffered_inputs = [
                    unsplit_pixel_values_by_grid(batch) for batch in generation_batches
                ]
            inputs = self._buffered_inputs[self._step % self.args.steps_per_generation]
            self._step += 1
        else:
            inputs = self._generate_and_score_completions(generation_batch)
        return inputs

    def _generate_single_turn(self, prompts: list[str], images: Optional[list]):
        device = self.accelerator.device
        kwargs = {}
        if images is not None:
            kwargs = {"images": images}
            for prompt, image_list in zip(prompts, images):
                if isinstance(prompt, list):
                    prepare_multimodal_messages(prompt, num_images=len(image_list))
        prompts_text = [
            maybe_apply_chat_template({"prompt": prompt}, self.processing_class)[
                "prompt"
            ]
            for prompt in prompts
        ]
        if images is not None:
            prompt_inputs = self.processing_class(
                text=prompts_text, padding=True, return_tensors="pt", **kwargs
            )
            prompt_inputs = super()._prepare_inputs(prompt_inputs)
            forward_kwargs = {
                k: v
                for k, v in prompt_inputs.items()
                if k not in ["input_ids", "attention_mask"]
            }
        else:
            forward_kwargs = {}
        if self.use_vllm:
            if self.vllm_mode == "colocate" and self.args.vllm_enable_sleep_mode:
                torch.cuda.empty_cache()
                self.llm.wake_up()
            should_sync = self.state.global_step != self._last_loaded_step
            if self.generate_from_teacher and (not self.args.sync_ref_model):
                should_sync = False
            if should_sync:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step
            if self.vllm_mode == "server":
                all_prompts_text = gather_object(prompts_text)
                if images is not None:
                    all_images = gather_object(images)
                if self.accelerator.is_main_process:
                    ordered_set_of_prompts = all_prompts_text[:: self.num_generations]
                    if images is not None:
                        ordered_set_of_images = all_images[:: self.num_generations]
                    else:
                        ordered_set_of_images = None
                    with profiling_context(self, "vLLM.generate"):
                        output = self.vllm_client.generate(
                            prompts=ordered_set_of_prompts,
                            images=ordered_set_of_images,
                            n=self.num_generations,
                            repetition_penalty=self.repetition_penalty,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            top_k=-1 if self.top_k is None else self.top_k,
                            min_p=0.0 if self.min_p is None else self.min_p,
                            max_tokens=self.max_completion_length,
                            truncate_prompt_tokens=self.max_prompt_length,
                            generation_kwargs=self.args.generation_kwargs,
                        )
                        payload = (
                            output["prompt_ids"],
                            output["completion_ids"],
                            output["logprobs"],
                        )
                else:
                    payload = None
                obj_list = [payload]
                broadcast_object_list(obj_list, from_process=0)
                all_prompt_ids, all_completion_ids, all_logprobs = obj_list[0]
                all_prompt_ids = [
                    ids for ids in all_prompt_ids for _ in range(self.num_generations)
                ]
                process_slice = slice(
                    self.accelerator.process_index * len(prompts),
                    (self.accelerator.process_index + 1) * len(prompts),
                )
                prompt_ids = all_prompt_ids[process_slice]
                completion_ids = all_completion_ids[process_slice]
                logprobs = all_logprobs[process_slice]
            elif self.vllm_mode == "colocate":
                from vllm import SamplingParams

                generation_kwargs = {
                    "n": 1,
                    "repetition_penalty": self.repetition_penalty,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": -1 if self.top_k is None else self.top_k,
                    "min_p": 0.0 if self.min_p is None else self.min_p,
                    "max_tokens": self.max_completion_length,
                    "truncate_prompt_tokens": self.max_prompt_length,
                    "logprobs": 0,
                }
                if self.args.generation_kwargs is not None:
                    generation_kwargs.update(self.args.generation_kwargs)
                sampling_params = SamplingParams(**generation_kwargs)
                if self.vllm_tensor_parallel_size > 1:
                    orig_size = len(prompts_text)
                    gathered_prompts = [
                        None for _ in range(self.vllm_tensor_parallel_size)
                    ]
                    torch.distributed.all_gather_object(
                        gathered_prompts, prompts_text, group=self.tp_group
                    )
                    all_prompts_text = [
                        p for sublist in gathered_prompts for p in sublist
                    ]
                    if images is not None:
                        gathered_images = [
                            None for _ in range(self.vllm_tensor_parallel_size)
                        ]
                        torch.distributed.all_gather_object(
                            gathered_images, images, group=self.tp_group
                        )
                        all_images = [
                            img for sublist in gathered_images for img in sublist
                        ]
                    else:
                        all_images = None
                else:
                    all_prompts_text = prompts_text
                    all_images = images
                if images is not None and all_images:
                    vllm_inputs = []
                    for prompt, image_list in zip(all_prompts_text, all_images):
                        vllm_inputs.append(
                            {
                                "prompt": prompt,
                                "multi_modal_data": {"image": image_list},
                            }
                        )
                else:
                    vllm_inputs = all_prompts_text
                with profiling_context(self, "vLLM.generate"):
                    all_outputs = self.llm.generate(
                        vllm_inputs, sampling_params=sampling_params, use_tqdm=False
                    )
                all_prompt_ids = [output.prompt_token_ids for output in all_outputs]
                all_completion_ids = [
                    output.token_ids
                    for outputs in all_outputs
                    for output in outputs.outputs
                ]
                all_logprobs = [
                    [next(iter(lp.values())).logprob for lp in output.logprobs]
                    for outputs in all_outputs
                    for output in outputs.outputs
                ]
                if self.vllm_tensor_parallel_size > 1:
                    local_rank_in_group = torch.distributed.get_rank(
                        group=self.tp_group
                    )
                    tp_slice = slice(
                        local_rank_in_group * orig_size,
                        (local_rank_in_group + 1) * orig_size,
                    )
                    prompt_ids = all_prompt_ids[tp_slice]
                    completion_ids = all_completion_ids[tp_slice]
                    logprobs = all_logprobs[tp_slice]
                else:
                    prompt_ids = all_prompt_ids
                    completion_ids = all_completion_ids
                    logprobs = all_logprobs
                if self.args.vllm_enable_sleep_mode:
                    self.llm.sleep(level=1)
        elif self.use_transformers_paged:
            paged_prompt_inputs = self.processing_class(text=prompts_text, **kwargs)
            previous_attn = self.model_wrapped.config._attn_implementation
            if is_flash_attn_2_available():
                self.model_wrapped.config._attn_implementation = "paged_attention"
            else:
                self.model_wrapped.config._attn_implementation = "sdpa_paged"
            with (
                profiling_context(self, "transformers.generate_batch"),
                unwrap_model_for_generation(
                    self.ref_model
                    if self.generate_from_teacher
                    else self.model_wrapped,
                    self.accelerator,
                    gather_deepspeed3_params=self.args.ds3_gather_for_generation,
                ) as unwrapped_model,
                torch.no_grad(),
                FSDP.summon_full_params(self.model_wrapped, recurse=False)
                if self.is_fsdp_enabled
                else nullcontext(),
            ):
                if self.args.bf16:
                    unwrapped_model.to(torch.bfloat16)
                elif self.args.fp16:
                    unwrapped_model.to(torch.float16)
                with torch.inference_mode():
                    all_outputs = unwrapped_model.generate_batch(
                        paged_prompt_inputs.input_ids,
                        generation_config=self.generation_config,
                        progress_bar=False,
                    )
                    unwrapped_model.train(not self.generate_from_teacher)
            completion_ids = [
                output.generated_tokens for output in all_outputs.values()
            ]
            prompt_ids = paged_prompt_inputs.input_ids
            self.model_wrapped.config._attn_implementation = previous_attn
            logprobs = None
        else:
            generate_inputs = self.processing_class(
                text=prompts_text,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                max_length=self.max_prompt_length,
                truncation=True,
                add_special_tokens=False,
                **kwargs,
            )
            generate_inputs = super()._prepare_inputs(generate_inputs)
            with (
                profiling_context(self, "transformers.generate"),
                unwrap_model_for_generation(
                    self.ref_model
                    if self.generate_from_teacher
                    else self.model_wrapped,
                    self.accelerator,
                    gather_deepspeed3_params=self.args.ds3_gather_for_generation,
                ) as unwrapped_model,
                torch.no_grad(),
                FSDP.summon_full_params(self.model_wrapped, recurse=False)
                if self.is_fsdp_enabled
                else nullcontext(),
            ):
                was_training = unwrapped_model.training
                unwrapped_model.eval()
                prompt_completion_ids = unwrapped_model.generate(
                    **generate_inputs,
                    generation_config=self.generation_config,
                    disable_compile=True,
                )
                unwrapped_model.train(was_training)
            prompt_ids, prompt_mask = (
                generate_inputs["input_ids"],
                generate_inputs["attention_mask"],
            )
            prompt_length = prompt_ids.size(1)
            completion_ids = prompt_completion_ids[:, prompt_length:]
            is_eos = completion_ids == self.eos_token_id
            eos_idx = torch.full(
                (is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device
            )
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            sequence_indices = torch.arange(is_eos.size(1), device=device).expand(
                is_eos.size(0), -1
            )
            completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
            prompt_ids = [p[m].tolist() for p, m in zip(prompt_ids, prompt_mask.bool())]
            completion_ids = [
                c[m].tolist() for c, m in zip(completion_ids, completion_mask.bool())
            ]
            logprobs = None
        return (prompt_ids, completion_ids, logprobs, forward_kwargs)

    def _generate(self, prompts: list[str], images: Optional[list]):
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        prompt_ids, completion_ids, logprobs, forward_kwargs = (
            self._generate_single_turn(prompts, images)
        )
        prompt_lengths = torch.tensor([len(ids) for ids in prompt_ids], device=device)
        completion_lengths = torch.tensor(
            [len(ids) for ids in completion_ids], device=device
        )
        agg_prompt_lengths = self.accelerator.gather(prompt_lengths)
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        total_prompt_tokens = agg_prompt_lengths.sum()
        total_completion_tokens = agg_completion_lengths.sum()
        if mode == "train":
            self.state.num_input_tokens_seen += (
                total_prompt_tokens + total_completion_tokens
            ).item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]
        self._metrics[mode]["completions/mean_length"].append(
            agg_completion_lengths.float().mean().item()
        )
        self._metrics[mode]["completions/min_length"].append(
            agg_completion_lengths.float().min().item()
        )
        self._metrics[mode]["completions/max_length"].append(
            agg_completion_lengths.float().max().item()
        )
        eos_and_pad = [self.eos_token_id, self.pad_token_id]
        is_truncated = torch.tensor(
            [not ids or ids[-1] not in eos_and_pad for ids in completion_ids],
            device=device,
        )
        agg_is_truncated = self.accelerator.gather(is_truncated)
        self._metrics[mode]["completions/clipped_ratio"].append(
            agg_is_truncated.float().mean().item()
        )
        term_completion_lengths = agg_completion_lengths[~agg_is_truncated]
        if len(term_completion_lengths) == 0:
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(
            term_completion_lengths.float().mean().item()
        )
        self._metrics[mode]["completions/min_terminated_length"].append(
            term_completion_lengths.float().min().item()
        )
        self._metrics[mode]["completions/max_terminated_length"].append(
            term_completion_lengths.float().max().item()
        )
        return (
            prompt_ids,
            completion_ids,
            total_completion_tokens,
            logprobs,
            forward_kwargs,
        )

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        prompts = [x["prompt"] for x in inputs]
        teacher_prompts = [x["teacher_prompt"] for x in inputs]
        if "images" in inputs[0]:
            images = [example.get("images") for example in inputs]
        elif "image" in inputs[0]:
            images = [
                [example.get("image")] if example.get("image") is not None else None
                for example in inputs
            ]
        else:
            images = None
        if images is not None and all((img_list == [] for img_list in images)):
            images = None
        generation_prompts = teacher_prompts if self.generate_from_teacher else prompts
        (
            _generation_prompt_ids_list,
            completion_ids_list,
            num_items_in_batch,
            sampling_per_token_logps_list,
            forward_kwargs,
        ) = self._generate(generation_prompts, images)
        prompts_text = [
            maybe_apply_chat_template({"prompt": prompt}, self.processing_class)[
                "prompt"
            ]
            for prompt in prompts
        ]
        if self.use_vllm:
            self.processing_class.truncation_side = "left"
        student_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            max_length=self.max_prompt_length,
            truncation=True,
            add_special_tokens=False,
        )
        student_inputs = super()._prepare_inputs(student_inputs)
        student_prompt_ids, student_prompt_mask = (
            student_inputs["input_ids"],
            student_inputs["attention_mask"],
        )
        prompt_ids_list = [
            p[m].tolist()
            for p, m in zip(student_prompt_ids, student_prompt_mask.bool())
        ]
        teacher_prompts_text = [
            maybe_apply_chat_template({"prompt": prompt}, self.processing_class)[
                "prompt"
            ]
            for prompt in teacher_prompts
        ]
        teacher_inputs = self.processing_class(
            text=teacher_prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            max_length=self.max_prompt_length,
            truncation=True,
            add_special_tokens=False,
        )
        teacher_inputs = super()._prepare_inputs(teacher_inputs)
        if self.use_vllm:
            self.processing_class.truncation_side = "left"
        teacher_prompt_ids, teacher_prompt_mask = (
            teacher_inputs["input_ids"],
            teacher_inputs["attention_mask"],
        )
        teacher_prompt_ids_list = [
            p[m].tolist()
            for p, m in zip(teacher_prompt_ids, teacher_prompt_mask.bool())
        ]
        prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(
            prompt_ids, padding_value=self.pad_token_id, padding_side="left"
        )
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")
        teacher_prompt_ids = [
            torch.tensor(ids, device=device) for ids in teacher_prompt_ids_list
        ]
        teacher_prompt_mask = [
            torch.ones_like(ids, dtype=torch.long) for ids in teacher_prompt_ids
        ]
        teacher_prompt_ids = pad(
            teacher_prompt_ids, padding_value=self.pad_token_id, padding_side="left"
        )
        teacher_prompt_mask = pad(
            teacher_prompt_mask, padding_value=0, padding_side="left"
        )
        completion_ids = [
            torch.tensor(ids, device=device) for ids in completion_ids_list
        ]
        completion_mask = [
            torch.ones_like(ids, dtype=torch.long) for ids in completion_ids
        ]
        completion_ids = pad(
            completion_ids, padding_value=self.pad_token_id, padding_side="right"
        )
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right")
        if sampling_per_token_logps_list is not None:
            sampling_per_token_logps = [
                torch.tensor(logps, device=device)
                for logps in sampling_per_token_logps_list
            ]
            sampling_per_token_logps = pad(
                sampling_per_token_logps, padding_value=0.0, padding_side="right"
            )
        else:
            sampling_per_token_logps = None
        if self.mask_truncated_completions:
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor(
                [not ids or ids[-1] not in eos_and_pad for ids in completion_ids_list],
                device=device,
            )
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        if "token_type_ids" in forward_kwargs:
            token_type_ids = forward_kwargs["token_type_ids"]
            forward_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids.new_zeros(completion_ids.shape)], dim=1
            )
        logits_to_keep = completion_ids.size(1)
        batch_size = (
            self.args.per_device_train_batch_size
            if mode == "train"
            else self.args.per_device_eval_batch_size
        )
        num_images = (
            [len(img_list) for img_list in images] if images is not None else None
        )
        with torch.no_grad():
            generate_every = self.args.steps_per_generation * self.num_iterations
            if not self.generate_from_teacher and (
                self.args.gradient_accumulation_steps % generate_every != 0
                or (self.use_vllm and self.vllm_importance_sampling_correction)
            ):
                old_per_token_logps, _, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    num_images=num_images,
                    compute_all_logps=False,
                    **forward_kwargs,
                )
            else:
                old_per_token_logps = None
            if (
                self.use_vllm
                and self.vllm_importance_sampling_correction
                and (not self.generate_from_teacher)
            ):
                importance_sampling_ratio = torch.exp(
                    old_per_token_logps - sampling_per_token_logps
                )
                importance_sampling_ratio = torch.clamp(
                    importance_sampling_ratio, max=self.vllm_importance_sampling_cap
                )
            else:
                importance_sampling_ratio = None
        prompts_text = self.processing_class.batch_decode(
            prompt_ids, skip_special_tokens=True
        )
        completions_text = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = (
                    prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                )
                completions.append(
                    [{"role": "assistant", "content": bootstrap + completion}]
                )
        else:
            completions = completions_text
        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        if images is not None:
            self._logs["images"].extend(gather_object(images))
        if importance_sampling_ratio is not None:
            delta = torch.abs(old_per_token_logps - sampling_per_token_logps)
            delta = delta[completion_mask.bool()]
            mean_delta = (
                torch.mean(delta)
                if delta.numel() > 0
                else torch.tensor(0.0, device=device)
            )
            max_delta = (
                torch.max(delta)
                if delta.numel() > 0
                else torch.tensor(0.0, device=device)
            )
            self._metrics[mode]["sampling/sampling_logp_difference/mean"].append(
                self.accelerator.gather(mean_delta).mean().item()
            )
            self._metrics[mode]["sampling/sampling_logp_difference/max"].append(
                self.accelerator.gather(max_delta).max().item()
            )
            flat_is_ratio = importance_sampling_ratio[completion_mask.bool()]
            min_importance_sampling_ratio = (
                torch.min(flat_is_ratio)
                if flat_is_ratio.numel() > 0
                else torch.tensor(0.0, device=device)
            )
            mean_importance_sampling_ratio = (
                torch.mean(flat_is_ratio)
                if flat_is_ratio.numel() > 0
                else torch.tensor(0.0, device=device)
            )
            max_importance_sampling_ratio = (
                torch.max(flat_is_ratio)
                if flat_is_ratio.numel() > 0
                else torch.tensor(0.0, device=device)
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/min"].append(
                nanmin(self.accelerator.gather(min_importance_sampling_ratio)).item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/mean"].append(
                self.accelerator.gather(mean_importance_sampling_ratio).nanmean().item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/max"].append(
                nanmax(self.accelerator.gather(max_importance_sampling_ratio)).item()
            )
        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "teacher_prompt_ids": teacher_prompt_ids,
            "teacher_prompt_mask": teacher_prompt_mask,
            "num_items_in_batch": num_items_in_batch,
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if importance_sampling_ratio is not None:
            output["importance_sampling_ratio"] = importance_sampling_ratio
        if "pixel_values" in forward_kwargs:
            output["pixel_values"] = forward_kwargs["pixel_values"]
        if "image_grid_thw" in forward_kwargs:
            output["image_grid_thw"] = forward_kwargs["image_grid_thw"]
        if "pixel_attention_mask" in forward_kwargs:
            output["pixel_attention_mask"] = forward_kwargs["pixel_attention_mask"]
        if "image_sizes" in forward_kwargs:
            output["image_sizes"] = forward_kwargs["image_sizes"]
        if "token_type_ids" in forward_kwargs:
            output["token_type_ids"] = forward_kwargs["token_type_ids"]
        if images is not None:
            output["num_images"] = num_images
        return output

    @profiling_decorator
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if return_outputs:
            raise ValueError("The DistilTrainer does not support returning outputs")
        return self._compute_loss(model, inputs)

    def _compute_loss(self, model, inputs):
        mode = "train" if self.model.training else "eval"
        prompt_ids, prompt_mask = (inputs["prompt_ids"], inputs["prompt_mask"])
        completion_ids, completion_mask = (
            inputs["completion_ids"],
            inputs["completion_mask"],
        )
        teacher_prompt_ids, teacher_prompt_mask = (
            inputs["teacher_prompt_ids"],
            inputs["teacher_prompt_mask"],
        )
        loss_completion_mask = completion_mask
        if self.num_loss_tokens_to_skip > 0:
            batch_size, seq_len = completion_mask.shape
            token_positions = (
                torch.arange(seq_len, device=completion_mask.device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
            skip_mask = (token_positions >= self.num_loss_tokens_to_skip).int()
            loss_completion_mask = completion_mask * skip_mask
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        teacher_input_ids = torch.cat([teacher_prompt_ids, completion_ids], dim=1)
        teacher_attention_mask = torch.cat(
            [teacher_prompt_mask, completion_mask], dim=1
        )
        logits_to_keep = completion_ids.size(1)
        per_token_logps, all_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            num_images=inputs.get("num_images"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
            token_type_ids=inputs.get("token_type_ids"),
        )
        with torch.no_grad():
            teacher_per_token_logps, teacher_all_logps, teacher_entropies = (
                self._get_per_token_logps_and_entropies(
                    self.ref_model,
                    teacher_input_ids,
                    teacher_attention_mask,
                    logits_to_keep,
                    compute_entropy=True,
                    pixel_values=inputs.get("pixel_values"),
                    image_grid_thw=inputs.get("image_grid_thw"),
                    num_images=inputs.get("num_images"),
                    pixel_attention_mask=inputs.get("pixel_attention_mask"),
                    image_sizes=inputs.get("image_sizes"),
                    token_type_ids=inputs.get("token_type_ids"),
                )
            )
        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(
                entropies, loss_completion_mask, 1 - self.top_entropy_quantile
            )
        else:
            entropy_mask = None
        if self.use_safety_weighting:
            with torch.no_grad():
                K = min(self.dpsw_top_k, teacher_all_logps.shape[-1])
                topk_logps, _ = torch.topk(teacher_all_logps, k=K, dim=-1)
                topk_probs = topk_logps.exp()
                topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp(
                    min=1e-10
                )
                topk_entropy = -(topk_probs * torch.log(topk_probs + 1e-10)).sum(dim=-1)
                log_K = torch.log(
                    torch.tensor(float(K), device=teacher_all_logps.device)
                )
                C_T = 1.0 - topk_entropy / log_K
                C_T = C_T.clamp(min=0.0, max=1.0)
                teacher_top_token = teacher_all_logps.argmax(dim=-1)
                student_logp_for_teacher_top = all_logps.gather(
                    dim=-1, index=teacher_top_token.unsqueeze(-1)
                ).squeeze(-1)
                student_prob_for_teacher_top = student_logp_for_teacher_top.exp()
                D_S = 1.0 - student_prob_for_teacher_top
                D_S = D_S.clamp(min=0.0, max=1.0)
                w_n = C_T * D_S
                w_sum = (
                    (w_n * loss_completion_mask)
                    .sum(dim=-1, keepdim=True)
                    .clamp(min=1e-08)
                )
                seq_len_per_sample = loss_completion_mask.sum(
                    dim=-1, keepdim=True
                ).clamp(min=1.0)
                w_tilde = seq_len_per_sample * w_n / w_sum
                w_for_topk = w_n * loss_completion_mask + -1000000000.0 * (
                    1 - loss_completion_mask
                )
                k = min(self.safety_weighting_top_k, w_n.shape[-1])
                topk_vals, topk_indices = torch.topk(w_for_topk, k=k, dim=-1)
                safety_weight_logs = []
                for b in range(completion_ids.shape[0]):
                    seq_log = []
                    for j in range(k):
                        idx = topk_indices[b, j].item()
                        token_id = completion_ids[b, idx].item()
                        token_text = self.processing_class.decode([token_id])
                        teacher_top_id = teacher_top_token[b, idx].item()
                        teacher_top_text = self.processing_class.decode(
                            [teacher_top_id]
                        )
                        seq_log.append(
                            {
                                "pos": idx,
                                "token": token_text,
                                "teacher_top_token": teacher_top_text,
                                "w_n": w_n[b, idx].item(),
                                "C_T": C_T[b, idx].item(),
                                "D_S": D_S[b, idx].item(),
                            }
                        )
                    safety_weight_logs.append(seq_log)
                self._logs["safety_weights"].extend(gather_object(safety_weight_logs))
                mean_C_T = (
                    C_T * loss_completion_mask
                ).sum() / loss_completion_mask.sum().clamp(min=1.0)
                mean_D_S = (
                    D_S * loss_completion_mask
                ).sum() / loss_completion_mask.sum().clamp(min=1.0)
                mean_w_n = (
                    w_n * loss_completion_mask
                ).sum() / loss_completion_mask.sum().clamp(min=1.0)
                self._metrics[mode]["safety_weighting/C_T_mean"].append(
                    self.accelerator.gather(mean_C_T).nanmean().item()
                )
                self._metrics[mode]["safety_weighting/D_S_mean"].append(
                    self.accelerator.gather(mean_D_S).nanmean().item()
                )
                self._metrics[mode]["safety_weighting/w_n_mean"].append(
                    self.accelerator.gather(mean_w_n).nanmean().item()
                )
        else:
            w_tilde = None
        if self.alpha == 0:
            kl_loss = kl_div(
                all_logps, teacher_all_logps, reduction="none", log_target=True
            )
        elif self.alpha == 1:
            kl_loss = kl_div(
                teacher_all_logps, all_logps, reduction="none", log_target=True
            )
        else:
            alpha = torch.tensor(
                self.alpha, dtype=all_logps.dtype, device=all_logps.device
            )
            mixture_log_probs = torch.logsumexp(
                torch.stack(
                    [
                        all_logps + torch.log(1 - alpha),
                        teacher_all_logps + torch.log(alpha),
                    ]
                ),
                dim=0,
            )
            kl_teacher = kl_div(
                mixture_log_probs, teacher_all_logps, reduction="none", log_target=True
            )
            kl_student = kl_div(
                mixture_log_probs, all_logps, reduction="none", log_target=True
            )
            kl_loss = alpha * kl_teacher + (1 - alpha) * kl_student
        per_token_loss = kl_loss.sum(-1)
        if (
            self.use_vllm
            and self.vllm_importance_sampling_correction
            and (not self.generate_from_teacher)
        ):
            ratio = inputs["importance_sampling_ratio"]
            importance_weights = (ratio * loss_completion_mask).sum(
                -1
            ) / loss_completion_mask.sum(-1).clamp(min=1.0)
            importance_weights = importance_weights.unsqueeze(-1)
            per_token_loss = per_token_loss * importance_weights
        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask
        if w_tilde is not None:
            per_token_loss = per_token_loss * w_tilde
        loss = (
            (per_token_loss * loss_completion_mask).sum(-1)
            / loss_completion_mask.sum(-1).clamp(min=1.0)
        ).mean()
        loss = loss / self.current_gradient_accumulation_steps
        with torch.no_grad():
            kl_approx = (
                per_token_logps
                - teacher_per_token_logps
                + torch.exp(teacher_per_token_logps - per_token_logps)
                - 1
            )
            kl_approx_mean = (
                kl_approx * loss_completion_mask
            ).sum() / loss_completion_mask.sum().clamp(min=1.0)
        self._metrics[mode]["kl_approx"].append(
            self.accelerator.gather(kl_approx_mean).nanmean().item()
        )
        loss_completion_token_count = loss_completion_mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:
                return x.mean()
            else:
                return (x * loss_completion_mask).sum() / loss_completion_token_count

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(
            self.accelerator.gather(mean_entropy).nanmean().item()
        )
        return loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys: Optional[list[str]] = None,
    ):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return (loss, None, None)

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}
        logs = {**logs, **metrics}
        super().log(logs, start_time)
        self._metrics[mode].clear()
        if self.accelerator.is_main_process and self.log_completions:
            if (
                self.args.report_to
                and "wandb" in self.args.report_to
                and (wandb.run is not None)
            ):
                import pandas as pd

                table = {
                    "step": [str(self.state.global_step)] * len(self._logs["prompt"]),
                    "prompt": self._logs["prompt"],
                    "completion": self._logs["completion"],
                }
                if self._logs["images"]:
                    table["images"] = []
                    for image_list in self._logs["images"]:
                        table["images"].append(
                            [wandb.Image(image) for image in image_list]
                        )
                df = pd.DataFrame(table)
                if self.wandb_log_unique_prompts:
                    df = df.drop_duplicates(subset=["prompt"])
                wandb.log({"completions": wandb.Table(dataframe=df)})
                if self.use_safety_weighting and self._logs["safety_weights"]:
                    sw_rows = []
                    for i, seq_log in enumerate(self._logs["safety_weights"]):
                        if not isinstance(seq_log, list):
                            continue
                        for entry in seq_log:
                            sw_rows.append(
                                {
                                    "step": str(self.state.global_step),
                                    "seq_idx": i,
                                    "pos": entry["pos"],
                                    "token": entry["token"],
                                    "teacher_top_token": entry["teacher_top_token"],
                                    "w_n": round(entry["w_n"], 4),
                                    "C_T": round(entry["C_T"], 4),
                                    "D_S": round(entry["D_S"], 4),
                                }
                            )
                    if sw_rows:
                        sw_df = pd.DataFrame(sw_rows)
                        wandb.log({"safety_weights": wandb.Table(dataframe=sw_df)})
