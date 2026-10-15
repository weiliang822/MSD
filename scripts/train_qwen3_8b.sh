set -euo pipefail
exec bash "$(dirname "$0")/run_train.sh" \
  --model_name Qwen/Qwen3-8B \
  --max_steps 300 \
  --output_dir outputs/msd-qwen3-8b \
  "$@"
