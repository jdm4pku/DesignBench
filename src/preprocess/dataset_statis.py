import os
import json

def get_word_length(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    words = text.split()
    return len(words)

def get_line_length(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return len(lines)

def dataset_statis(root_path="dataset/sysml/samples",output_path="dataset/sysml/dataset_statis.json"):
    dataset_result = {}
    for dirpath, dirnames, filenames in os.walk(root_path):
       dirnames.sort()
       if all(name in filenames for name in ['nl.txt', "nl_zh.txt", 'design.sysml', 'grammar.txt','domain.txt']):
           nl_en_path = os.path.join(dirpath,"nl.txt")
           design_path = os.path.join(dirpath, 'design.sysml')
           domain_path = os.path.join(dirpath,"domain.txt")
           nl_length = get_word_length(nl_en_path)
           design_length = get_line_length(design_path)
           with open(domain_path,'r',encoding='utf-8') as f:
                domain = f.read().strip()
           if domain not in dataset_result:
                dataset_result[domain] = {
                    "count": 1,
                    "nl_length": [nl_length],
                    "design_length": [design_length],
                }
           else:
                dataset_result[domain]["count"] += 1
                dataset_result[domain]["nl_length"].append(nl_length)
                dataset_result[domain]["design_length"].append(design_length)
    for item in dataset_result.keys():
        "计算nl_length和design_length的avg,max,min"
        dataset_result[item]["nl_length_avg"] = sum(dataset_result[item]["nl_length"]) / len(dataset_result[item]["nl_length"])
        dataset_result[item]["nl_length_max"] = max(dataset_result[item]["nl_length"])
        dataset_result[item]["nl_length_min"] = min(dataset_result[item]["nl_length"])
        dataset_result[item]["design_length_avg"] = sum(dataset_result[item]["design_length"]) / len(dataset_result[item]["design_length"])
        dataset_result[item]["design_length_max"] = max(dataset_result[item]["design_length"])
        dataset_result[item]["design_length_min"] = min(dataset_result[item]["design_length"])
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_result, f, ensure_ascii=False, indent=2)
    "remove the nl_length和design_length的list"
    for item in dataset_result.keys():
        del dataset_result[item]["nl_length"]
        del dataset_result[item]["design_length"]
    with open("table6.json", 'w', encoding='utf-8') as f:
        json.dump(dataset_result, f, ensure_ascii=False, indent=2)

if __name__=="__main__":
    dataset_statis()