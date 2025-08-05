import os
import json

def get_grammar_id():
    dataset_path = "dataset/sysml/dataset.json"
    with open(dataset_path,"r",encoding="utf-8") as f:
        data = json.load(f)
    result = {}
    for idx, sample in enumerate(data):
        grammar = sample["grammar"]
        if grammar not in result:
            result[grammar] = [idx]
        else:
            result[grammar].append(idx)
    return result

def get_metrics_various_grammar(grammar2id):
    traditional_metrics_path = "result/qwen3/direct.json"
    sysm_eval_metrics_path = "result/qwen3/sysm_eval/metrics_direct.json"
    with open(traditional_metrics_path,"r",encoding="utf-8") as f:
        traditional_metrics = json.load(f)
    with open(sysm_eval_metrics_path,"r",encoding="utf-8") as f:
        sysm_eval_metrics = json.load(f)
    grammar_result = {}
    for grammar, sample_id_list in grammar2id.items():
        total_blue = 0
        total_rouge = 0
        total_bertscore = 0
        total_sysm_eval_p = 0
        total_sysm_eval_r = 0
        for sample_id in sample_id_list:
            total_blue = total_blue + traditional_metrics[sample_id]["sentence_bleu_score"]
            total_rouge = total_rouge + traditional_metrics[sample_id]["rougeL_f1"]
            total_bertscore = total_bertscore + traditional_metrics[sample_id]["bertscore"]
            total_sysm_eval_p = total_sysm_eval_p + sysm_eval_metrics[sample_id]["sysm_eval_p"]
            total_sysm_eval_r = total_sysm_eval_r + sysm_eval_metrics[sample_id]["sysm_eval_r"]
        grammar_item = {
            "blue": total_blue / len(sample_id_list),
            "rouge": total_rouge / len(sample_id_list),
            "bertscore": total_bertscore / len(sample_id_list),
            "sysm_eval_p": total_sysm_eval_p / len(sample_id_list),
            "sysm_eval_r": total_sysm_eval_r / len(sample_id_list)
        }
        grammar_result[grammar] = grammar_item
    result_path = "grammar_result.json"
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(grammar_result, f, ensure_ascii=False, indent=4)
    print(f"Qwen Domain Result saved to {result_path}")

if __name__=="__main__":
    grammar2sampleid = get_grammar_id()
    get_metrics_various_grammar(grammar2sampleid)
    pass