reasons=(
  "direct"
  "few-shot"
  "cot"
  "grammar"
)

for reason in "${reasons[@]}"; do
  CUDA_VISIBLE_DEVICES=0,1 python src/llm_inference/run_magicoder.py \
      --data dataset/sysml/dataset.json \
      --reason "${reason}" \
      --prompt_dir prompts \
      --model_type general \
      --model Magicoder-S-CL-7B \
      --moda greedy \
      --output_dir predict
done
