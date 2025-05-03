import os
import time
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


def generate_nl_requirements(root_path="dataset/sysml/samples"):
    file_list = ["01","02","03","04","05","06","07","08","09","10"]
    for id in file_list:
    # for id in tqdm(range(11,152)):
        data_dir = f"{root_path}/{id}"
        sysml_path = f"{data_dir}/design.sysml"
        nl_path = f"{data_dir}/nl.txt"
        with open(sysml_path,'r',encoding='utf-8') as f:
            sysml_model = f.read()
        prompt = f"{sysml_model}\n 帮我写一下这个sysml描述的系统模型对应的中文自然语言需求，要遵循人类终端用户描述需求的语言风格。 写一段话不要分点。"
        nl_requirements = get_completion(prompt)
        with open(nl_path,'w',encoding='utf-8') as f:
            f.write(nl_requirements)
        print(f"{id}已生成需求")
if __name__=="__main__":
    generate_nl_requirements()