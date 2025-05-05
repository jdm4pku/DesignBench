import os
import time
import json
import openai
from openai import OpenAI
from tqdm import tqdm

def get_completion(prompt):
    client = OpenAI(
        api_key="sk-sR8RiK6YYrtk8Rss1b29047069804d108211285c7a25356c",  # 填写上api-key
        base_url="https://api.yesapikey.com/v1"
    )
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

def generate_domain(root_path="dataset/sysml/samples"):
    file_list = ["01","02","03","04","05","06","07","08","09","10"]
    for id in file_list:
    # for id in tqdm(range(11,152)):
        data_dir = f"{root_path}/{id}"
        nl_path = f"{data_dir}/nl.txt"
        domain_path = f"{data_dir}/domain.txt"
        with open(nl_path,'r',encoding='utf-8') as f:
            nl_requirements = f.read()
        prompt = f"{nl_requirements}\n 帮我写一下这个系统属于什么领域，如航空航天、车辆交通、系统工程、摄影技术、计算仿真、能源材料、保密安全、网络通信、医疗健康、信息管理等? 注意只输出这个领域的名字！"
        domain_desc = get_completion(prompt)
        with open(domain_path,'w',encoding='utf-8') as f:
            f.write(domain_desc)
        print(f"{id}已生成领域")

def generate_domain_list(root_path="dataset/sysml/samples",output_path="dataset/sysml/domain.json"):
    domain_result = []
    domain_count = {}
    count = 0
    for dirpath,dirnames, filenames in os.walk(root_path):
        if all(name in filenames for name in ['nl.txt', "nl_zh.txt", 'design.sysml', 'grammar.txt','domain.txt']):
            domain_path = os.path.join(dirpath,'domain.txt')
            with open(domain_path, 'r', encoding='utf-8') as f:
                domain = f.read().strip()
            if domain not in domain_count:
                domain_count[domain] = 1
            else:
                domain_count[domain] +=1
            count +=1
    print(f"add {count} items for the dataset")
    for key,values in domain_count.items():
        domain_result.append(
            {
                'domain':key,
                'count':values
            }
        )
    domain_result = sorted(domain_result, key=lambda x: x['count'], reverse=True)
    with open(output_path,'w',encoding='utf-8') as f:
        json.dump(domain_result, f, ensure_ascii=False, indent=2)
    pass

if __name__=="__main__":
    generate_domain_list()