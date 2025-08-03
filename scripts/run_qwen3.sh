reasons=(
  "direct"
  "few-shot"
  "cot"
  "grammar"
)

for reason in "${reasons[@]}"; do
  python src/llm_inference/run_qwen3.py \
      --data dataset/sysml/dataset.json \
      --reason "${reason}" \
      --prompt_dir prompts \
      --model_type general \
      --model qwen3 \
      --moda greedy \
      --output_dir predict
done