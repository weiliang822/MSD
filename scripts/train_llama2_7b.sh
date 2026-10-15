set -euo pipefail
exec bash "$(dirname "$0")/run_train.sh" \
  --model_name meta-llama/Llama-2-7b-chat-hf \
  --max_steps 20 \
  --output_dir outputs/msd-llama2-7b \
  "$@"
