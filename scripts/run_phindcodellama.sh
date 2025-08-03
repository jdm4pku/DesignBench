reasons=(
  "direct"
  "few-shot"
  "cot"
  "grammar"
)

for reason in "${reasons[@]}"; do
  CUDA_VISIBLE_DEVICES=0,1,2,3 python src/llm_inference/run_phindcodellama.py \
      --data dataset/sysml/dataset.json \
      --reason "${reason}" \
      --prompt_dir prompts \
      --model_type general \
      --model Phind-CodeLlama-34B-v2 \
      --moda greedy \
      --output_dir predict
done
