set -euo pipefail
exec bash "$(dirname "$0")/run_train.sh" \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --max_steps 60 \
  --output_dir outputs/msd-qwen2.5-7b \
  "$@"
