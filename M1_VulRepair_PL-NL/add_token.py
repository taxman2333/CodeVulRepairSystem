from transformers import AutoTokenizer

# 加载原始分词器
model_path = "/data/share/models/vulrepair-7B"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 添加自定义特殊token
new_tokens = [
    "<S2SV_StartBug>", 
    "<S2SV_EndBug>", 
    "<S2SV_blank>", 
    "<S2SV_ModStart>", 
    "<S2SV_ModEnd>"
]

# 检查token是否已存在，避免重复添加
new_tokens = [token for token in new_tokens if token not in tokenizer.get_vocab()]
num_added = tokenizer.add_tokens(new_tokens)

print(f"Added {num_added} new tokens")

# 保存修改后的分词器
tokenizer.save_pretrained(model_path)

print("Tokenizer with new tokens has been saved successfully!")