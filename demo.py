import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import torch
import gradio as gr
from model import GPT, GPTConfig
import tiktoken

device = 'cuda'
checkpoint = torch.load('out-python-code/ckpt.pt', map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)

state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

model.load_state_dict(state_dict)
model.eval()
model.to(device)
enc = tiktoken.get_encoding("gpt2")
print("模型加载完成！")

def generate_code(prompt, max_new_tokens=200, temperature=0.8, top_k=50):
    if not prompt.strip():
        return "请输入代码提示..."
    input_ids = enc.encode(prompt)
    x = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        y = model.generate(x, max_new_tokens=int(max_new_tokens), temperature=temperature, top_k=int(top_k))
    return enc.decode(y[0].tolist())

with gr.Blocks(title="Python Code GPT") as demo:
    gr.Markdown("""
    # 🌟 Python Code GPT
    基于GPT-2（124M）在50K条Python代码上fine-tune的代码补全模型
    **训练信息：** RTX 5090 | 3000步 | val loss: 1.15
    """)
    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(label="输入代码开头", lines=5, value="def binary_search(arr, target):\n    ")
            temperature = gr.Slider(0.1, 2.0, value=0.8, step=0.1, label="Temperature")
            top_k = gr.Slider(1, 100, value=50, step=1, label="Top-k")
            max_tokens = gr.Slider(50, 500, value=200, step=50, label="最大生成token数")
            btn = gr.Button("生成代码 🚀", variant="primary")
        with gr.Column():
            output = gr.Textbox(label="生成结果", lines=20)
    gr.Examples(
        examples=[
            ["def fibonacci(n):\n    "],
            ["class Stack:\n    def __init__(self):\n        self.items = []\n    \n    def push(self, item):\n        "],
            ["import numpy as np\n\ndef softmax(x):\n    "],
            ["def quicksort(arr):\n    if len(arr) <= 1:\n        "],
        ],
        inputs=prompt_input
    )
    btn.click(fn=generate_code, inputs=[prompt_input, max_tokens, temperature, top_k], outputs=output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
