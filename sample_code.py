import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import torch
from model import GPT, GPTConfig
import tiktoken

device = 'cuda'
checkpoint = torch.load('out-python-code/ckpt.pt', map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)

# 去掉compile产生的_orig_mod.前缀
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

model.load_state_dict(state_dict)
model.eval()
model.to(device)

enc = tiktoken.get_encoding("gpt2")

prompts = [
    "def binary_search(arr, target):\n    ",
    "def fibonacci(n):\n    ",
    "class Stack:\n    def __init__(self):\n        ",
    "import numpy as np\n\ndef softmax(x):\n    ",
]

for prompt in prompts:
    print(f"\n{'='*60}")
    print(f"输入:\n{prompt}")
    print(f"模型续写:")
    input_ids = enc.encode(prompt)
    x = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        y = model.generate(x, max_new_tokens=200, temperature=0.8, top_k=50)
    output = enc.decode(y[0].tolist())
    print(output)  