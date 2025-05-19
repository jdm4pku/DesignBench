models=(
  "mistral-small-3.1-24B-instruct"
  "Qwen3-32B"
  "gemma-3-27b-it"
  "Llama-3.1-8B-Instruct"
  "internlm3-8b-instruct"
  "Baichuan2-13B-Chat"
  "ChatGLM3-6B"
)

# -------- 提示策略 (reason) 列表 ----------
reasons=(
  "direct"
  "few-shot"
  "cot"
  "grammar"
)

# -------- 主循环 ----------
for model in "${models[@]}"; do
  for reason in "${reasons[@]}"; do
    echo "========== ${model}  |  reason=${reason} =========="
    CUDA_VISIBLE_DEVICES=0,1,2,3 python evaluate_llm.py \
      --data dataset/sysml/dataset.json \
      --reason "${reason}" \
      --prompt_dir prompts \
      --model_type general \
      --model "${model}" \
      --moda greedy \
      --output_dir result
  done
done