reasons=(
  "direct"
  "few-shot"
  "cot"
  "grammar"
)

for reason in "${reasons[@]}"; do
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python src/llm_inference/run_mistral.py \
      --data dataset/sysml/dataset.json \
      --reason "${reason}" \
      --prompt_dir prompts \
      --model_type general \
      --model Mistral-7B-Instruct-v0.2 \
      --moda greedy \
      --output_dir predict
done
