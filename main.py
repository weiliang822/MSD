import argparse
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Multilingual Self-Distillation")
    parser.add_argument("--model_name", default="Qwen/Qwen3-8B")
    parser.add_argument("--train_data", default="data/train.csv")
    parser.add_argument("--languages", default="zh,ar,bn,de,fr,hi,ja,ru,sp")
    parser.add_argument(
        "--sampling_policy", choices=["on_policy", "off_policy"], default="on_policy"
    )
    parser.add_argument("--output_dir", default="outputs/msd")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_train_epochs", type=float, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--save_steps", type=int, default=400)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_completion_length", type=int, default=512)
    parser.add_argument("--num_generations", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="0: forward KL; 1: reverse KL; (0, 1): generalized Jensen-Shannon divergence",
    )
    parser.add_argument(
        "--use_safety_weighting", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--dpsw_top_k", type=int, default=32)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument(
        "--use_vllm", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.3)
    parser.add_argument("--deepspeed", default=None)
    parser.add_argument("--use_peft", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--report_to", default="none")
    parser.add_argument("--log_completions", action="store_true")
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for name in (
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "max_prompt_length",
        "max_completion_length",
        "num_generations",
        "save_steps",
        "vllm_tensor_parallel_size",
        "lora_r",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be positive")
    if args.temperature <= 0 or args.learning_rate <= 0 or args.num_train_epochs <= 0:
        parser.error("temperature, learning rate and epochs must be positive")
    if not 0 <= args.alpha <= 1 or args.dpsw_top_k < 2:
        parser.error("alpha must be in [0, 1] and dpsw_top_k must be at least 2")
    if not 0 < args.vllm_gpu_memory_utilization < 1:
        parser.error("vllm_gpu_memory_utilization must be in (0, 1)")
    return args


def load_safety_examples(train_path, languages):
    languages = list(dict.fromkeys(lang.strip() for lang in languages if lang.strip()))
    if not languages:
        raise ValueError("At least one target language is required")
    with open(train_path, encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(["en", *languages]) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing language columns: {sorted(missing)}")
        examples = []
        for line, row in enumerate(reader, 2):
            english = (row["en"] or "").strip()
            if not english:
                raise ValueError(f"Missing English query at CSV line {line}")
            for language in languages:
                question = (row[language] or "").strip()
                if not question:
                    raise ValueError(f"Missing {language} query at CSV line {line}")
                teacher_content = (
                    f"You are a helpful assistant.\n"
                    f"User query: '{question}'\n\n"
                    f"The question in English is: '{english}'.\n"
                    f"Instructions:\n"
                    f"1. Analyze the query in English to determine if it violates safety guidelines (e.g., illegal, harmful, dangerous).\n"
                    f"2. If unsafe, generate a polite refusal. If safe, answer helpfully. Always respond in the same language as the original question.\n\n"
                    f"Please output your response:"
                )
                examples.append(
                    {
                        "prompt": [{"role": "user", "content": question}],
                        "teacher_prompt": [
                            {"role": "user", "content": teacher_content}
                        ],
                    }
                )
    if not examples:
        raise ValueError("The training dataset is empty")
    return examples


def main():
    args = parse_args()
    examples = load_safety_examples(args.train_data, args.languages.split(","))

    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    from src.distil_config import DistilConfig
    from src.distil_trainer import DistilTrainer

    set_seed(args.seed)
    config = DistilConfig(
        output_dir=args.output_dir,
        seed=args.seed,
        use_vllm=args.use_vllm,
        vllm_mode="colocate",
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_enable_sleep_mode=args.use_vllm,
        generate_from_teacher=args.sampling_policy == "off_policy",
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        bf16=args.precision == "bf16",
        fp16=args.precision == "fp16",
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        max_grad_norm=1.0,
        logging_steps=1,
        alpha=args.alpha,
        use_safety_weighting=args.use_safety_weighting,
        dpsw_top_k=args.dpsw_top_k,
        num_generations=args.num_generations,
        temperature=args.temperature,
        report_to=args.report_to,
        log_completions=args.log_completions,
        deepspeed=args.deepspeed,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
    )
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        args.precision
    ]
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype)
    teacher = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, padding_side="left", truncation_side="left"
    )
    peft_config = None
    if args.use_peft:
        from peft import LoraConfig

        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )
    dataset = Dataset.from_list(examples).shuffle(seed=args.seed)
    print(f"Training on {len(dataset)} multilingual queries ({args.sampling_policy})")
    trainer = DistilTrainer(
        model=model,
        ref_model=teacher,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(Path(args.output_dir) / "final"))
    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(Path(args.output_dir) / "final")


if __name__ == "__main__":
    main()
