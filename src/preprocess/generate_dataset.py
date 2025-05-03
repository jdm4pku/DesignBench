import os
import json

def generate_dataset_json(root_path="dataset/sysml/samples",output_path="dataset/sysml/grammar/dataset.json"):
    dataset = []
    count = 0
    for dirpath, dirnames, filenames in os.walk(root_path):
       if all(name in filenames for name in ['nl.txt', "nl_zh.txt", 'design.sysml', 'label.txt']):
           nl_en_path = os.path.join(dirpath,"nl.txt")
          #  nl_zh_path = os.path.join(dirpath,"nl_zh.txt")
           design_path = os.path.join(dirpath, 'design.sysml')
           label_path = os.path.join(dirpath, 'label.txt')
           with open(nl_en_path, 'r', encoding='utf-8') as f:
                nl_en = f.read().strip()
          #  with open(nl_zh_path, 'r', encoding='utf-8') as f:
          #       nl_zh = f.read().strip()
           with open(design_path, 'r', encoding='utf-8') as f:
                design_sysml = f.read().strip()
           with open(label_path, 'r', encoding='utf-8') as f:
                label = f.read().strip()
           dataset.append(
               {
                   "nl":nl_en,
                   "design":design_sysml,
                   "label":label
               }
           )
           count +=1
    print(f"add {count} items for the dataset")
    with open(output_path,'w',encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

if __name__=="__main__":
    generate_dataset_json()