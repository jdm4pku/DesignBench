import os
import time
import json
import openai
from openai import OpenAI
from tqdm import tqdm

def remame_grammar_file(root_path="dataset/sysml/samples"):
    count = 0
    for dirpath,dirnames, filenames in os.walk(root_path):
        if all(name in filenames for name in ['nl.txt', "nl_zh.txt", 'design.sysml', 'label.txt','domain.txt']):
            grammar_old_path = f"{dirpath}/label.txt"
            grammar_new_path = f"{dirpath}/grammar.txt"
            os.rename(grammar_old_path,grammar_new_path)
            count+=1
    print(f"{count}个已完成重命名！")

def generate_grammar_list(root_path="dataset/sysml/samples",output_path="dataset/sysml/grammar.json"):
    grammar_result = []
    grammar_count = {}
    count = 0
    for dirpath,dirnames, filenames in os.walk(root_path):
        if all(name in filenames for name in ['nl.txt', "nl_zh.txt", 'design.sysml', 'grammar.txt','domain.txt']):
            domain_path = os.path.join(dirpath,'grammar.txt')
            with open(domain_path, 'r', encoding='utf-8') as f:
                domain = f.read().strip()
            if domain not in grammar_count:
                grammar_count[domain] = 1
            else:
                grammar_count[domain] +=1
            count +=1
    print(f"add {count} items for the dataset")
    for key,values in grammar_count.items():
        grammar_result.append(
            {
                'domain':key,
                'count':values
            }
        )
    grammar_result = sorted(grammar_result, key=lambda x: x['count'], reverse=True)
    with open(output_path,'w',encoding='utf-8') as f:
        json.dump(grammar_result, f, ensure_ascii=False, indent=2)
    pass

if __name__=="__main__":
    # remame_grammar_file()
    generate_grammar_list()