import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def print_eos_token_id(model_name: str, gpu_id: int = 7):
    """
    将模型加载到指定 GPU（默认为 cuda:7），并打印 eos_token_id。
    
    参数
    ----
    model_name : str
        Hugging Face 模型标识或本地路径。
    gpu_id : int
        GPU 序号（默认为 7，对应 cuda:7）。
    """
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() and gpu_id < torch.cuda.device_count()
                          else "cpu")
    print(f"[INFO] 使用设备: {device}")

    # 1. 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)

    # 2. 可选：加载模型本体到指定设备
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16         # 半精度省显存；可按需修改
    ).to(device)
    model.eval()

    # 3. 打印 eos_token 与其 ID
    eos_id = getattr(tokenizer, "eos_token_id", None)
    eos_tok = getattr(tokenizer, "eos_token", None)
    
    if eos_id is None:
        print(f"[WARN] 模型 '{model_name}' 未定义 eos_token。")
    else:
        print(f"模型: {model_name}")
        print(f"eos_token: {repr(eos_tok)}")
        print(f"eos_token_id: {eos_id}")

if __name__ == "__main__":
    # 示例：替换为你想查看的模型
    model_dir = "../../LLMs/Baichuan2-13B-Chat"
    print_eos_token_id(model_dir)