import os
import json
from argparse import ArgumentParser
from vllm import LLM, SamplingParams
from tqdm import tqdm

def llm_parser_args():
    parser = ArgumentParser()
    parser.add_argument("--data",type=str,default="dataset/sysml/dataset.json")
    parser.add_argument("--reason",type=str,default="direct",choices=["direct","few-shot","cot","rag","grammar"])
    parser.add_argument("--prompt_dir",type=str,default="prompts")
    parser.add_argument("--model_type",type=str,default="general",choices=["general","code"])
    parser.add_argument("--model",type=str,required=True)
    parser.add_argument("--moda",type=str,default="greedy")
    parser.add_argument("--output_dir",type=str,default="result")
    return parser.parse_args()

def get_prompt(args):
    prompt_list = []
    with open(args.data, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    prompt_template_path = os.path.join(args.prompt_dir, f"{args.reason}.txt")
    with open(prompt_template_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    for item in dataset:
        nl = item["nl"]
        prompt = prompt_template.format(requirement=nl)
        prompt_list.append(prompt)
    return prompt_list

def load_general_llms(model_name,moda,max_tokens=1024,max_model_len=4096):
    model_dir = ""
    stop_token_id = []
    if model_name == "mistral-small-3.1-24B-instruct": 
        print("Loading Mistral-Small-3.1-24B-instruct")
        model_dir = "../../LLMs/Mistral-Small-3.1-24B_instruct"
        stop_token_id = [2]
    elif model_name == "Qwen3-32B":
        print("Loading Qwen3-32B")
        model_dir = "../../LLMs/Qwen3-32B"
        stop_token_id = [151645]
    elif model_name == "gemma-3-27b-it":
        print("Loading Gemma-3-27b-it")
        model_dir = "../../LLMs/Gemma-3-27b-it"
        stop_token_id = [1]
    elif model_name == "Llama-3.1-8B-Instruct":
        print("Loading Llama-3.1-8B-Instruct")
        model_dir = "../../LLMs/Meta-Llama-3.1-8B-Instruct"
        stop_token_id = []
    elif model_name == "internlm3-8b-instruct":
        print("Loading InternLM3-8b-instruct")
        model_dir = "../../LLMs/internlm3-8b-instruct"
        stop_token_id = [2]
    elif model_name == "Baichuan2-13B-Chat":
        print("Loading Baichuan2-13B-Chat")
        model_dir = "../../LLMs/Baichuan2-13B-Chat"
        stop_token_id = []
    elif model_name == "ChatGLM3-6B":
        print("Loading ChatGLM3-6B")
        model_dir = "../../LLMs/chatglm3-6b"
        stop_token_id = []
    if moda == "greedy":
        sample_params = SamplingParams(temperature=0.0,max_tokens = max_tokens, stop_token_ids = stop_token_id, n=1)
    else:
        sample_params = SamplingParams(temperature=0.4, top_p=0.95, max_tokens=max_tokens, n=20)
    # 如果model_dir不存在，则打印并结束程序
    if not os.path.exists(model_dir):
        print(f"Model directory {model_dir} does not exist.")
        exit(1)
    model = LLM(model=model_dir, max_model_len=max_model_len,trust_remote_code=True,gpu_memory_utilization=0.9,tensor_parallel_size=4)
    return model, sample_params
    # elif model_name == "DeepSeek-Coder-V2-Instruct": #vllm不支持这个模型
    #     print("Loading DeepSeek-Coder-V2-Instruct")
    #     model_dir = "../../LLMs/DeepSeek-Coder-V2-Instruct"
    # elif model_name == "Codestral-22B-v0.1":  #vllm不支持这个模型
    #     print("Loading Codestral-22B-v0.1")
    #     model_dir = "../../LLMs/Codestral-22B-v0.1"
    # elif model_name == "Phind-CodeLlama-34B-v2": #vllm不支持这个模型
    #     print("Loading Phind-CodeLlama-34B-v2")
    #     model_dir = "../../LLMs/Phind-CodeLlama-34B-v2-AWQ"
    # elif model_name == "Magicoder-S-CL-7B": #vllm不支持这个模型
    #     print("Loading Magicoder-S-CL-7B")
    #     model_dir = "../../LLMs/Magicoder-S-CL-7B"
    # elif model_name == "WizardCoder-33B-V1.1": #vllm不支持这个模型
    #     print("Loading WizardCoder-33B-V1.1")
    #     model_dir = "../../LLMs/WizardCoder-33B-V1.1-AWQ" 
    # elif model_name == "CodeLlama-34b-Instruct-hf":
    #     print("Loading CodeLlama-34b-Instruct-hf")
    #     model_dir = "../../LLMs/CodeLlama-34b-Instruct-hf"
    # elif model_name == "starcoder2-15b-instruct-v0.1":
    #     print("Loading starcoder2-15b-instruct-v0.1")
    #     model_dir = "../../LLMs/starcoder2-15b-instruct-v0.1"
    # elif model_name == "OctoCoder":
    #     print("Loading OctoCoder")
    #     model_dir = "../../LLMs/OctoCoder"
    # elif model_name == "CodeGen2.5-7B-Instruct":
    #     print("Loading CodeGen2.5-7B-Instruct")
    #     model_dir = "../../LLMs/CodeGen2.5-7B-Instruct"

def general_llm_inference(args,model,sample_params):
    prompt_list = get_prompt(args)
    predict_list = []
    for prompt in tqdm(prompt_list, desc="generating answer"):
        predict = model.generate([prompt], sample_params)
        predict_text = predict[0].outputs[0].text
        predict_dict = {
            "predict": predict_text
        }
        predict_list.append(predict_dict)
    output_dir = os.path.join(args.output_dir, args.model)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f"{args.reason}.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(predict_list, f, ensure_ascii=False, indent=4)
    
def evaluate_llm():
    args = llm_parser_args()
    if args.model_type == "general": # 使用vllm库加载通用模型
        model, sample_params = load_general_llms(args.model,args.moda)
        general_llm_inference(args,model,sample_params)
    elif args.model_type == "code": # 使用vllm库加载代码模型
        pass

if __name__=="__main__":
    evaluate_llm()