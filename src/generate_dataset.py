"""
本文件用于生成dataset.json
1. 对于dataset/sysml/grammar/xx中的每个数据进行读取
2. 读取英文版的nl.txt，design.sysml,label.txt的内容放给三个键nl,design, label
3. 保存一下dataset.json文件，格式如下：
   [
     {
       "nl":
       "design":
       "label":
     },
     {
       "nl":
       "design":
       "label":
     }
   ]
"""