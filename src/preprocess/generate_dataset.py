import os
import json

def generate_dataset_json(root_path="dataset/sysml/samples",output_path="dataset/sysml/dataset.json"):
    dataset = []
    count = 0
    for dirpath, dirnames, filenames in os.walk(root_path):
       dirnames.sort()
       if all(name in filenames for name in ['nl.txt', "nl_zh.txt", 'design.sysml', 'grammar.txt','domain.txt']):
           nl_en_path = os.path.join(dirpath,"nl.txt")
          #  nl_zh_path = os.path.join(dirpath,"nl_zh.txt")
           design_path = os.path.join(dirpath, 'design.sysml')
           domain_path = os.path.join(dirpath,"domain.txt")
           grammar_path = os.path.join(dirpath, 'grammar.txt')
           diagram_path = os.path.join(dirpath,'design.png')
           with open(nl_en_path, 'r', encoding='utf-8') as f:
                nl_en = f.read().strip()
          #  with open(nl_zh_path, 'r', encoding='utf-8') as f:
          #       nl_zh = f.read().strip()
           with open(design_path, 'r', encoding='utf-8') as f:
                design_sysml = f.read().strip()
           with open(domain_path,'r',encoding='utf-8') as f:
                domain = f.read().strip()
           with open(grammar_path, 'r', encoding='utf-8') as f:
                grammar = f.read().strip()
           dataset.append(
               {
                   "nl":nl_en,
                   "design":design_sysml,
                   "domain":domain,
                   "grammar":grammar,
                   "diagram": diagram_path
               }
           )
           count +=1
    print(f"add {count} items for the dataset")
    with open(output_path,'w',encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

if __name__=="__main__":
    generate_dataset_json()