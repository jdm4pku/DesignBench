"""
本文件用于将中文的nl.txt翻译成英文
1. 读取dataset/sysml/grammar/xx/nl.txt文件
2. 调用大语言模型将nl.txt文件的内容翻译成英文
3. 将翻译后的英文保存到dataset/sysml/grammar/xx/nl_en.txt文件。
Note:有些文件还没check完成自然语言，这个可以先不用考虑，直接按照上面的逻辑实现就行
"""
import os
import openai
import time