import os
import json

def count_lines(code: str) -> int:
    """
    统计代码字符串的行数。
    同时兼容以下两种情况：
    1) 字符串中是真实换行符（\n、\r\n、\r）
    2) 字符串中是转义形式的“\\n”“\\r\\n”（如示例）

    :param code: 代码字符串
    :return: 行数（int）
    """
    if not code:
        return 0
    # 若不存在真实换行，但存在转义的 \n / \r\n，则先把转义还原为真实换行
    if '\n' not in code and '\\n' in code:
        code = (code
                .replace('\\r\\n', '\n')  # 先处理转义的 \r\n
                .replace('\\n', '\n')     # 再处理转义的 \n
                .replace('\\r', '\n'))    # 以及转义的 \r

    # 统一换行符到 \n
    code = code.replace('\r\n', '\n').replace('\r', '\n')

    # 按行拆分统计
    return len(code.split('\n')) if code else 0

def difficult_id():
    dataset_path = "dataset/sysml/dataset.json"
    with open(dataset_path,"r",encoding="utf-8") as f:
        data = json.load(f)
    result = {
        "1":[],
        "2":[],
        "3":[],
        "4":[],
        "5":[]
    }
    for idx, sample in enumerate(data):
        sysm = sample["design"]
        line = count_lines(sysm)
        if line < 30:
            result["1"].append(idx)
        elif line<60:
            result["2"].append(idx)
        elif line<90:
            result["3"].append(idx)
        elif line<120:
            result["4"].append(idx)
        else:
            result["5"].append(idx)
    return result

def get_distribution(result):
    for key,value in result.items():
        print(f"{key}:{len(value)}")

def get_metrics_various_difficulty(difficult2id):
    traditional_metrics_path = "result/qwen3/direct.json"
    sysm_eval_metrics_path = "result/qwen3/sysm_eval/metrics_direct.json"
    with open(traditional_metrics_path,"r",encoding="utf-8") as f:
        traditional_metrics = json.load(f)
    with open(sysm_eval_metrics_path,"r",encoding="utf-8") as f:
        sysm_eval_metrics = json.load(f)
    difficult_result = {}
    for difficult, sample_id_list in difficult2id.items():
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
        difficult_item = {
            "blue": total_blue / len(sample_id_list),
            "rouge": total_rouge / len(sample_id_list),
            "bertscore": total_bertscore / len(sample_id_list),
            "sysm_eval_p": total_sysm_eval_p / len(sample_id_list),
            "sysm_eval_r": total_sysm_eval_r / len(sample_id_list)
        }
        difficult_result[difficult] = difficult_item
    result_path = "difficult_result.json"
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(difficult_result, f, ensure_ascii=False, indent=4)
    print(f"Qwen Difficult Result saved to {result_path}")


if __name__=="__main__":
    difficult2id = difficult_id()
    get_distribution(difficult2id)
    # get_metrics_various_difficulty(difficult2id)
