reasons=(
    # "direct"
    # "few-shot"
    # "cot"
    "grammar"
)

# get traditional metrics 
# for reason in "${reasons[@]}"; do
#     python src/metrics/metrics.py \
#         --preference_path dataset/sysml/dataset.json \
#         --predict_dir predict \
#         --reason "${reason}" \
#         --model claude3 \
#         --output_dir result
# done

# get sysm-eval metrics
# for reason in "${reasons[@]}"; do
#     python src/metrics/get_sysm_eval.py \
#         --preference_path dataset/sysml/dataset.json \
#         --predict_dir predict \
#         --reason "${reason}" \
#         --model claude3 \
#         --output_dir result
# done

# # # parse sysm-eval metrics
for reason in "${reasons[@]}"; do
    python src/metrics/parse_sysm_eval.py \
        --preference_path dataset/sysml/dataset.json \
        --predict_dir predict \
        --reason "${reason}" \
        --model claude3 \
        --output_dir result
done