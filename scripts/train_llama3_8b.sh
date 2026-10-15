set -euo pipefail
exec bash "$(dirname "$0")/run_train.sh" \
  --model_name meta-llama/Meta-Llama-3-8B-Instruct \
  --max_steps 20 \
  --output_dir outputs/msd-llama3-8b \
  "$@"
