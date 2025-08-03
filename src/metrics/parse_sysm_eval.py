import json
import os
import re
from argparse import ArgumentParser
from tqdm import tqdm

def parse_parser_args():
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
        accuracy = correct / total if total!=0 else 0
        return correct, total, accuracy
    else:
        print("Score not found.")

def parser_sysm_eval_r(text):
    match = re.search(r'Score:\s*(\d+)\s*/\s*(\d+)', text)
    if match:
        matched = int(match.group(1))         # 正确召回数量
        total_gold = int(match.group(2))      # 总参考数量
        recall = round(matched / total_gold, 4) if total_gold!=0 else 0
        return matched, total_gold, recall
    else:
        print("Score not found.")

def parse_sysm_eval(args):
    respond_path = f"{args.output_dir}/{args.model}/sysm_eval/response_{args.reason}.json"
    respond_data = json.load(open(respond_path, 'r', encoding='utf-8'))
    result = []
    for item in tqdm(respond_data):
        p_response = item["sysm_eval_p"]
        r_response = item["sysm_eval_r"]
        try:
            correct, total_generated, accuracy = parser_sysm_eval_p(p_response)
        except Exception as e:
            print("*****p****")
            print(e)
            print(p_response)
            print("*****p****")
            exit()
        try:
            matched, total_reference, recall = parser_sysm_eval_r(r_response)
        except Exception as e:
            print("*****r****")
            print(e)
            print(r_response)
            print("*****r****")
            exit()
        f1 = 0.0 if (accuracy + recall) == 0 else 2 * accuracy * recall / (accuracy + recall)
        print(f"--accuracy:{accuracy}--correct:{correct}--total_generated:{total_generated}")
        print(f"--recall:{recall}--matched:{matched}--total_reference:{total_reference}")
        print(f"--f:{f1}")
        result.append({
            "sysm_eval_p": accuracy,
            "sysm_eval_r": recall,
            "sysm_eval_f1": f1
        })
    # 计算一个平均值
    sysm_eval_p_avg = sum([item["sysm_eval_p"] for item in result]) / len(result)
    sysm_eval_r_avg = sum([item["sysm_eval_r"] for item in result]) / len(result)
    sysm_eval_f1_avg = 0.0 if (sysm_eval_p_avg + sysm_eval_r_avg) == 0 else 2 * sysm_eval_p_avg * sysm_eval_r_avg / (sysm_eval_p_avg + sysm_eval_r_avg)
    print(f"--total_p:{sysm_eval_p_avg}--total_r:{sysm_eval_r_avg}--total_f1:{sysm_eval_f1_avg}")
    result.append({
        "sysm_eval_p_avg": sysm_eval_p_avg,
        "sysm_eval_r_avg": sysm_eval_r_avg,
        "sysm_eval_f1_avg": sysm_eval_f1_avg
    })
    result_path = f"{args.output_dir}/{args.model}/sysm_eval/metrics_{args.reason}.json"
    with open(result_path, 'w', encoding = 'utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    print(f"Result saved to {result_path}")

if __name__=="__main__":
    args = parse_parser_args()
    parse_sysm_eval(args)




