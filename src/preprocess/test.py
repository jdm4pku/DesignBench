
import json

with open("example.sysml","r",encoding="utf-8") as f:
    design_sysml = f.read().strip()

with open("example.txt","r",encoding="utf-8") as f:
    nl_req = f.read().strip()

result = {
    "req":nl_req,
    "design": design_sysml
}

with open("example.json",'w',encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

    