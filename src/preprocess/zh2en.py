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

def translate_nl_requirements(root_path="dataset/sysml/samples"):
    # file_list = ["01","02","03","04","05","06","07","08","09","10"]
    # for id in file_list:
    for id in tqdm(range(11,152)):
        data_dir = f"{root_path}/{id}"
        nl_path = f"{data_dir}/nl.txt"
        nl_zh_path = f"{data_dir}/nl_zh.txt"
        with open(nl_path,'r',encoding='utf-8') as f:
            nl_zh = f.read()
        with open(nl_zh_path,'w',encoding='utf-8') as f:
            f.write(nl_zh)
        print(f"{id}需求：nl.txt --> nl_zh.txt")
        prompt = f"{nl_zh} \n 请精确将上面的中文描述翻译成英文。"
        nl_en = get_completion(prompt)
        with open(nl_path,'w',encoding='utf-8') as f:
            f.write(nl_en)
        print(f"{id}需求已经翻译。")

if __name__=="__main__":
    translate_nl_requirements()