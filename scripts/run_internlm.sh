reasons=(
  "direct"
  "few-shot"
  "cot"
  "grammar"
)

for reason in "${reasons[@]}"; do
  CUDA_VISIBLE_DEVICES=0,1,2,3 python src/llm_inference/run_internlm3.py \
      --data dataset/sysml/dataset.json \
      --reason "${reason}" \
      --prompt_dir prompts \
      --model_type general \
      --model internlm3-8b-instruct \
      --moda greedy \
      --output_dir predict
done
