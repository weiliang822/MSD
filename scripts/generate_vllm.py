import argparse
import csv
import hashlib
import json
from pathlib import Path

LANGUAGES = ["en", "zh", "it", "vi", "ar", "ko", "th", "bn", "sw", "jv"]
DATASETS = {"multijail": "data/MultiJail.csv", "pku": "data/PKU-SafeRLHF.csv"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate MSD safety benchmark responses"
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--datasets", default="multijail,pku")
    parser.add_argument("--multijail_data", default=DATASETS["multijail"])
    parser.add_argument("--pku_data", default=DATASETS["pku"])
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--model_name", default=None, help="Unique output folder name")
    parser.add_argument("--languages", default=",".join(LANGUAGES))
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument(
        "--dtype", choices=["auto", "bfloat16", "float16"], default="auto"
    )
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if min(args.batch_size, args.max_new_tokens, args.tensor_parallel_size) < 1:
        parser.error(
            "batch_size, max_new_tokens and tensor_parallel_size must be positive"
        )
    if (
        args.temperature < 0
        or not 0 < args.top_p <= 1
        or not 0 < args.gpu_memory_utilization < 1
    ):
        parser.error("Invalid sampling or GPU memory settings")
    if args.top_k != -1 and args.top_k < 1:
        parser.error("top_k must be -1 or positive")
    return args


def load_dataset(path, languages):
    with open(path, encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = {"id", *languages} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty dataset: {path}")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate example IDs in {path}")
    for row in rows:
        if any(not (row[language] or "").strip() for language in languages):
            raise ValueError(f"Missing query at id={row['id']}")
    return {
        language: [
            {
                "id": row["id"],
                "question": row[language].strip(),
                "tags": row.get("tags", ""),
            }
            for row in rows
        ]
        for language in languages
    }


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def model_signature(model_path):
    path = Path(model_path)
    if not path.is_dir():
        return model_path
    return [
        (p.name, p.stat().st_size, p.stat().st_mtime_ns)
        for p in sorted(path.iterdir())
        if p.suffix in {".json", ".safetensors", ".bin", ".model"}
    ]


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main():
    args = parse_args()
    datasets = list(
        dict.fromkeys(x.strip() for x in args.datasets.split(",") if x.strip())
    )
    languages = list(
        dict.fromkeys(x.strip() for x in args.languages.split(",") if x.strip())
    )
    if not datasets or set(datasets) - DATASETS.keys():
        raise ValueError("Select datasets from multijail,pku")
    if not languages or set(languages) - set(LANGUAGES):
        raise ValueError(f"Select languages from {LANGUAGES}")
    model_name = args.model_name or Path(args.adapter_path or args.model_path).name
    if (
        not model_name
        or model_name in {".", ".."}
        or "/" in model_name
        or "\\" in model_name
    ):
        raise ValueError("model_name must be a single folder name")
    settings = {
        k: v
        for k, v in vars(args).items()
        if k not in {"overwrite", "output_dir", "batch_size", "languages", "datasets"}
    }
    settings["model_signature"] = model_signature(args.model_path)
    if args.adapter_path:
        settings["adapter_signature"] = model_signature(args.adapter_path)
    pending = []
    for dataset in datasets:
        data = load_dataset(getattr(args, f"{dataset}_data"), languages)
        for language, rows in data.items():
            run_id = fingerprint(
                {
                    "settings": settings,
                    "rows": rows,
                    "language": language,
                    "dataset": dataset,
                }
            )
            destination = (
                Path(args.output_dir)
                / dataset
                / model_name
                / language
                / "results.jsonl"
            )
            if destination.exists() and not args.overwrite:
                with destination.open(encoding="utf-8") as stream:
                    existing = [json.loads(line) for line in stream if line.strip()]
                if len(existing) == len(rows) and all(
                    record.get("run_id") == run_id for record in existing
                ):
                    print(f"Already generated: {dataset}/{model_name}/{language}")
                    continue
                raise ValueError(
                    f"Existing results do not match this run: {destination}. Use a new output directory or --overwrite."
                )
            pending.append((dataset, language, rows, destination, run_id))
    if not pending:
        return

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path or args.model_path)
    model_kwargs = dict(
        model=args.model_path,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
    )
    lora_request = None
    if args.adapter_path:
        from peft import PeftConfig
        from vllm.lora.request import LoRARequest

        adapter_config = PeftConfig.from_pretrained(args.adapter_path)
        model_kwargs.update(enable_lora=True, max_lora_rank=max(8, adapter_config.r))
        lora_request = LoRARequest("msd", 1, args.adapter_path)
    llm = LLM(**model_kwargs)
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    for dataset, language, rows, destination, run_id in pending:
        records = []
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": row["question"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for row in batch
            ]
            outputs = llm.generate(prompts, sampling, lora_request=lora_request)
            if len(outputs) != len(batch):
                raise RuntimeError("The generation engine returned an incomplete batch")
            for row, prompt, output in zip(batch, prompts, outputs):
                records.append(
                    {
                        **row,
                        "dataset": dataset,
                        "model": model_name,
                        "language": language,
                        "prompt": prompt,
                        "response": output.outputs[0].text,
                        "run_id": run_id,
                    }
                )
        write_jsonl(destination, records)
        print(f"Saved {len(records)} responses: {destination}")


if __name__ == "__main__":
    main()
