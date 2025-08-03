import os
import json

def get_domain_id():
    dataset_path = "dataset/sysml/dataset.json"
    with open(dataset_path,"r",encoding="utf-8") as f:
        data = json.load(f)
    result = {}
    for idx, sample in enumerate(data):
        domain = sample["domain"]
        if domain not in result:
            result[domain] = [idx]
        else:
            result[domain].append(idx)
    return result

def get_metrics_various_domain(domain2sampleid):
    traditional_metrics_path = "result/qwen3/direct.json"
    sysm_eval_metrics_path = "result/qwen3/sysm_eval/metrics_direct.json"
    with open(traditional_metrics_path,"r",encoding="utf-8") as f:
        traditional_metrics = json.load(f)
    with open(sysm_eval_metrics_path,"r",encoding="utf-8") as f:
        sysm_eval_metrics = json.load(f)
    domain_result = {}
    for domain,sample_id_list in domain2sampleid.items():
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
        domain_item = {
            "blue": total_blue / len(sample_id_list),
            "rouge": total_rouge / len(sample_id_list),
            "bertscore": total_bertscore / len(sample_id_list),
            "sysm_eval_p": total_sysm_eval_p / len(sample_id_list),
            "sysm_eval_r": total_sysm_eval_r / len(sample_id_list)
        }
        domain_result[domain] = domain_item
    result_path = "domain_result.json"
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(domain_result, f, ensure_ascii=False, indent=4)
    print(f"Qwen Domain Result saved to {result_path}")
    

if __name__=="__main__":
    domain2sampleid = get_domain_id()
    get_metrics_various_domain(domain2sampleid)
    pass