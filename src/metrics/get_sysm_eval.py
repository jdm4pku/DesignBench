import os
import json
from argparse import ArgumentParser
from openai import OpenAI
from tqdm import tqdm
import time
import re

def eval_parser_args():
    parser = ArgumentParser()
    parser.add_argument("--preference_path",type=str,default="dataset/sysml/dataset.json")
    parser.add_argument("--predict_dir", type=str, default="predict")
    parser.add_argument("--reason",type=str,default="direct",choices=["direct","few-shot","cot","grammar"])
    parser.add_argument("--model",type=str,required=True)
    parser.add_argument("--output_dir",type=str,default="result")
    return parser.parse_args()

def parser_sysm_eval_p(text):
    match = re.search(r'Score:\s*(\d+)\s*/\s*(\d+)', text)
    if match:
        correct = int(match.group(1))
        total = int(match.group(2))
        accuracy = correct / total
        return correct, total, accuracy
    else:
        print("Score not found.")

def parser_sysm_eval_r(text):
    match = re.search(r'Score:\s*(\d+)\s*/\s*(\d+)', text)
    if match:
        matched = int(match.group(1))         # 正确召回数量
        total_gold = int(match.group(2))      # 总参考数量
        recall = round(matched / total_gold, 4)
        return matched, total_gold, recall
    else:
        print("Score not found.")

def get_sysm_eval_p(candidate,reference):
    client = OpenAI(
        api_key="sk-BwTI1iSg83soUQ6u2d1096B8A27848E5B3E4141154Dc592b",  # replace with your key
        base_url="https://api.yesapikey.com/v1",
    )
    with open("src/metrics/sysm-eval-p.txt","r", encoding="utf-8") as file:
        prompt_template = file.read()
    prompt = prompt_template.format(reference_model=reference, generated_model=candidate)
    flag = False
    while not flag:
        try:
            response = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="gpt-4.1-2025-04-14",
                    temperature=0   
            )
            flag=True
        except Exception as e:
            print(e)
            time.sleep(0.5)
    return response.choices[0].message.content

def get_sysm_eval_r(candidate,reference):
    client = OpenAI(
        api_key="sk-BwTI1iSg83soUQ6u2d1096B8A27848E5B3E4141154Dc592b",  # replace with your key
        base_url="https://api.yesapikey.com/v1",
    )
    with open("src/metrics/sysm-eval-r.txt","r", encoding="utf-8") as file:
        prompt_template = file.read()
    prompt = prompt_template.format(reference_model=reference, generated_model=candidate)
    flag = False
    while not flag:
        try:
            response = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="gpt-4.1-2025-04-14",
                    temperature=0   
            )
            flag=True
        except Exception as e:
            print(e)
            time.sleep(0.5)
    return response.choices[0].message.content

def compute_metrics(args):
    preference_data = json.load(open(args.preference_path, 'r', encoding='utf-8'))
    predict_path = f"{args.predict_dir}/{args.model}/{args.reason}.json"
    predict_data = json.load(open(predict_path, 'r', encoding='utf-8'))
    result = []
    for i,preference in tqdm(enumerate(preference_data),total=len(preference_data),desc="Computing metrics"):
        preference_answer = preference["design"]
        predict_answer = predict_data[i]
        sysm_eval_p = get_sysm_eval_p(preference_answer,predict_answer)
        # correct, total_generated, accuracy = parser_sysm_eval_p(sysm_eval_p)
        sysm_eval_r = get_sysm_eval_r(preference_answer,predict_answer)
        # matched, total_reference, recall = parser_sysm_eval_r(sysm_eval_r)
        result.append({
            "sysm_eval_p": sysm_eval_p,
            "sysm_eval_r": sysm_eval_r,
            # "p_correct": correct,
            # "p_total": total_generated,
            # "r_matched": matched,
            # "r_total": total_reference
        })
    # 计算一个平均值
    # sys_eval_p_avg = sum([item["sys_eval_p"] for item in result])
    # sys_eval_r_avg = sum([item["sys_eval_r"] for item in result])
    # bleu_score_avg = sum([item["bleu_score"] for item in result]) / len(result)
    # rouge1_f1_avg = sum([item["rouge1_f1"] for item in result]) / len(result)
    # rouge2_f1_avg = sum([item["rouge2_f1"] for item in result]) / len(result)
    # rougeL_f1_avg = sum([item["rougeL_f1"] for item in result]) / len(result)
    # bertscore_avg = sum([item["bertscore"] for item in result]) / len(result)
    # result.append({
    #     "sys_eval_p_avg": sys_eval_p_avg,
    #     "sys_eval_r_avg": sys_eval_r_avg
        # "bleu_score_avg":bleu_score_avg,
        # "rouge1_f1_avg":rouge1_f1_avg,
        # "rouge2_f1_avg":rouge2_f1_avg,
        # "rougeL_f1_avg":rougeL_f1_avg,
        # "bertscore_avg":bertscore_avg
    # })
    result_dir = f"{args.output_dir}/{args.model}"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_dir = f"{result_dir}/sysm_eval"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    with open(f"{result_dir}/response_{args.reason}.json", 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    print(f"Result saved to {result_dir}/response_{args.reason}.json")

if __name__=="__main__":
    args = eval_parser_args()
    print(f"==={args.model}==={args.reason}===")
    compute_metrics(args)