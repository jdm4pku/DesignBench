import os
import time
from openai import OpenAI
os.environ["OPENAI_API_KEY"] = "sk-BwTI1iSg83soUQ6u2d1096B8A27848E5B3E4141154Dc592b"
os.environ["OPENAI_BASE_URL"] = "https://api.yesapikey.com/v1"
client = OpenAI()
while True:
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-2024-05-13",
            # model = "gpt-4o-mini-2024-07-18",
            #max_tokens=10000,
            #temperature=1,
            #top_p=1,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"}
            ]
        )
        #判断有没有补全内容
        if getattr(completion.choices[0].message, 'content', None):
            content = completion.choices[0].message.content
            print(completion)#完整返回值
            print(content)#提取补全内容
            break
        else:
            #如果没有内容，打印错误信息或提示
            #print(completion)
            print('error_wait_2s')
    except:
        pass
    time.sleep(2)