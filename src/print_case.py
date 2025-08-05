import os
import json


with open("predict/claude3/cot.json","r",encoding="utf-8") as f:
    data = json.load(f)

print(data[2])