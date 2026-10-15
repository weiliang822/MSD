from dataclasses import dataclass
from typing import Optional, Union

from transformers import TrainingArguments


@dataclass
class DistilConfig(TrainingArguments):
    gradient_checkpointing: bool = True
    bf16: Optional[bool] = None
    model_init_kwargs: Optional[Union[dict, str]] = None
    disable_dropout: bool = False
    remove_unused_columns: Optional[bool] = False
    max_prompt_length: Optional[int] = 512
    num_generations: Optional[int] = 1
    max_completion_length: Optional[int] = 256
    shuffle_dataset: Optional[bool] = True
    generation_batch_size: Optional[int] = None
    steps_per_generation: Optional[int] = None
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    generation_kwargs: Optional[dict] = None
    repetition_penalty: float = 1.0
    use_transformers_paged: bool = False
    cache_implementation: Optional[str] = None
    ds3_gather_for_generation: bool = True
    use_vllm: bool = False
    vllm_mode: str = "colocate"
    vllm_model_impl: str = "vllm"
    vllm_enable_sleep_mode: bool = False
    vllm_server_base_url: Optional[str] = None
    vllm_server_host: str = "127.0.0.1"
    vllm_server_port: int = 8000
    vllm_server_timeout: float = 240.0
    vllm_gpu_memory_utilization: float = 0.3
    vllm_tensor_parallel_size: int = 1
    alpha: float = 1.0
    generate_from_teacher: bool = False
    num_iterations: int = 1
    mask_truncated_completions: bool = False
    sync_ref_model: bool = False
    ref_model_mixup_alpha: float = 0.6
    ref_model_sync_steps: int = 512
    top_entropy_quantile: float = 1.0
    num_loss_tokens_to_skip: int = 0
    use_safety_weighting: bool = True
    safety_weighting_top_k: int = 5
    dpsw_top_k: int = 32
    vllm_importance_sampling_correction: bool = True
    vllm_importance_sampling_cap: float = 2.0
    log_completions: bool = False
    wandb_log_unique_prompts: Optional[bool] = False
    save_only_model: Optional[bool] = False

    def __post_init__(self):
        self.bf16 = not self.fp16 if self.bf16 is None else self.bf16
        super().__post_init__()
        num_processes = self.world_size
        if self.generation_batch_size is None and self.steps_per_generation is None:
            self.steps_per_generation = self.gradient_accumulation_steps
            self.generation_batch_size = (
                self.per_device_train_batch_size
                * num_processes
                * self.steps_per_generation
            )
        elif (
            self.generation_batch_size is not None and self.steps_per_generation is None
        ):
            if (
                self.generation_batch_size
                % (self.per_device_train_batch_size * num_processes)
                != 0
            ):
                raise ValueError(
                    f"generation_batch_size ({self.generation_batch_size}) must be divisible by the global batch size ({self.per_device_train_batch_size * num_processes})."
                )
            self.steps_per_generation = self.generation_batch_size // (
                self.per_device_train_batch_size * num_processes
            )
        elif (
            self.generation_batch_size is None and self.steps_per_generation is not None
        ):
            self.generation_batch_size = (
                self.per_device_train_batch_size
                * num_processes
                * self.steps_per_generation
            )
        else:
            raise ValueError(
                "'generation_batch_size' and 'steps_per_generation' can not be both configured at the same time"
            )
        if self.do_eval and self.eval_strategy != "no":
            if (
                self.per_device_eval_batch_size * num_processes % self.num_generations
                != 0
            ):
                raise ValueError(
                    f"The global eval batch size ({self.per_device_eval_batch_size} * {num_processes}) must be divisible by num_generations ({self.num_generations})."
                )
        if self.generation_batch_size % self.num_generations != 0:
            raise ValueError(
                f"generation_batch_size ({self.generation_batch_size}) must be divisible by num_generations ({self.num_generations})."
            )
        if not 0 <= self.alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        if self.dpsw_top_k < 2:
            raise ValueError("dpsw_top_k must be at least 2")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
