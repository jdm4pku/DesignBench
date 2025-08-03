reasons=(
  "direct"
  "few-shot"
  "cot"
  "grammar"
)

for reason in "${reasons[@]}"; do
  CUDA_VISIBLE_DEVICES=4,5 python src/llm_inference/run_wizardCoder.py \
      --data dataset/sysml/dataset.json \
      --reason "${reason}" \
      --prompt_dir prompts \
      --model_type general \
      --model WizardCoder-15B-V1.0 \
      --moda greedy \
      --output_dir predict
done
