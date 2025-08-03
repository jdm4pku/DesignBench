reasons=(
  "direct"
  "few-shot"
  "cot"
  "grammar"
)

for reason in "${reasons[@]}"; do
  python src/llm_inference/run_gpt_4.1.py \
      --data dataset/sysml/dataset.json \
      --reason "${reason}" \
      --prompt_dir prompts \
      --model_type general \
      --model gpt_4.1 \
      --moda greedy \
      --output_dir predict
done
