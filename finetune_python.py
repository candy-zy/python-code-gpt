# ===== 数据 =====
dataset = 'python_code'
gradient_accumulation_steps = 4   # 模拟更大batch

# ===== 模型 =====
# 关键：从GPT-2预训练权重开始，不是随机初始化
# 这是fine-tune和从头训练的本质区别
init_from = 'gpt2'                 # 加载GPT-2 124M权重

# ===== 训练 =====
batch_size = 32                    # 5090有32GB显存，可以开大
block_size = 1024                  # GPT-2原版上下文长度
max_iters = 3000                   # fine-tune不需要太多步
lr_decay_iters = 3000
warmup_iters = 100

# ===== 学习率 =====
# fine-tune用比预训练小很多的学习率
# 原因：预训练权重已经很好了，大学习率会导致"灾难性遗忘"
# GPT-2预训练时用6e-4，这里缩小20倍
learning_rate = 3e-5
min_lr = 3e-6

# ===== 评估 =====
eval_interval = 200
eval_iters = 40
log_interval = 20

# ===== 5090 GPU设置 =====
device = 'cuda'
dtype = 'bfloat16'                 # 5090支持bfloat16，比float32快2倍
compile = True                     # torch.compile加速，5090上效果很好

# ===== 保存 =====
out_dir = 'out-python-code'
always_save_checkpoint = True

# ===== wandb日志（可选）=====
wandb_log = False
wandb_project = 'nanogpt-python'
wandb_run_name = 'finetune-gpt2-python-code'