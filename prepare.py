import os
import numpy as np
import tiktoken

input_file = os.path.join(os.path.dirname(__file__), 'raw.txt')
with open(input_file, 'r', encoding='utf-8') as f:
    data = f.read()

print(f"原始文本长度: {len(data):,} 字符")

# 划分训练集和验证集（90/10）
n = len(data)
train_data = data[:int(n * 0.9)]
val_data = data[int(n * 0.9):]

# GPT-2的BPE tokenizer，词表大小50257
enc = tiktoken.get_encoding("gpt2")

print("正在tokenize训练集...")
train_ids = enc.encode_ordinary(train_data)
print("正在tokenize验证集...")
val_ids = enc.encode_ordinary(val_data)

print(f"训练集token数: {len(train_ids):,}")
print(f"验证集token数: {len(val_ids):,}")

# 存成uint16节省一半存储（词表50257 < 65535，uint16够用）
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

print("预处理完成！")
print(f"train.bin: {train_ids.nbytes / 1024 / 1024:.1f} MB")
print(f"val.bin:   {val_ids.nbytes / 1024 / 1024:.1f} MB")