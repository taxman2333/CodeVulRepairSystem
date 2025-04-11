from flask import Flask, request, render_template
import os
import torch
from transformers import RobertaTokenizer, T5ForConditionalGeneration, RobertaConfig, RobertaModel
from codebert_model import Seq2Seq

# 设置 Hugging Face 镜像地址
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

app = Flask(__name__)

# 全局变量存储模型和分词器
model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.route("/", methods=["GET", "POST"])
def index():
    global model, tokenizer

    if request.method == "POST":
        # 获取用户选择的模型类型
        model_type = request.form.get("model_type")
        tokenizer_path = request.form.get("tokenizer_path")
        model_weights_path = request.form.get("model_weights_path")
        input_code = request.form.get("input_code")

        if "unload_model" in request.form:
            # 卸载模型
            model = None
            tokenizer = None
            return render_template("index.html", message="模型已卸载！", tokenizer_path=tokenizer_path, model_weights_path=model_weights_path)

        # 根据模型类型设置默认的 Hugging Face 模型名称
        if model_type == "codet5":
            model_name_or_path = "MickyMike/VulRepair"
        elif model_type == "codebert":
            model_name_or_path = "microsoft/codebert-base"
        else:
            return render_template("index.html", message="请选择有效的模型类型！")

        if "load_model" in request.form:
            # 加载模型
            try:
                # 加载分词器
                if tokenizer_path and os.path.exists(tokenizer_path):
                    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
                else:
                    tokenizer = RobertaTokenizer.from_pretrained(model_name_or_path)

                # 仅 CodeBERT 添加特殊 Token
                if model_type == "codebert":
                    tokenizer.add_tokens([
                        "<S2SV_StartBug>", 
                        "<S2SV_EndBug>", 
                        "<S2SV_blank>", 
                        "<S2SV_ModStart>", 
                        "<S2SV_ModEnd>"
                    ])

                # 加载模型
                if model_type == "codet5":
                    if model_weights_path and os.path.exists(model_weights_path):
                        config = T5ForConditionalGeneration.from_pretrained(model_name_or_path).config
                        model = T5ForConditionalGeneration(config)
                        model.load_state_dict(torch.load(model_weights_path, map_location=device))
                    else:
                        model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)

                    # 调整词表大小
                    model.resize_token_embeddings(len(tokenizer))
                    model.to(device)
                    return render_template("index.html", message="CodeT5 模型加载成功！", model_type=model_type)

                elif model_type == "codebert":
                    config = RobertaConfig.from_pretrained(model_name_or_path)
                    encoder = RobertaModel.from_pretrained(model_name_or_path, config=config)
                    encoder.resize_token_embeddings(len(tokenizer))
                    decoder_layer = torch.nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
                    decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=6)
                    model = Seq2Seq(
                        encoder=encoder,
                        decoder=decoder,
                        config=config,
                        beam_size=10,
                        max_length=256,
                        sos_id=tokenizer.cls_token_id,
                        eos_id=tokenizer.sep_token_id,
                    )

                    if model_weights_path and os.path.exists(model_weights_path):
                        model.load_state_dict(torch.load(model_weights_path, map_location=device))

                    model.to(device)
                    return render_template("index.html", message="CodeBERT 模型加载成功！", model_type=model_type)

            except Exception as e:
                return render_template("index.html", message=f"模型加载失败：{str(e)}", tokenizer_path=tokenizer_path, model_weights_path=model_weights_path)

        elif "generate_code" in request.form:
            # 修复漏洞代码
            if model is None or tokenizer is None:
                return render_template("index.html", message="请先加载模型！")

            if not input_code.strip():
                return render_template("index.html", message="请输入需要修复的漏洞代码！")

            try:
                if model_type == "codet5":
                    input_ids = tokenizer.encode(input_code, return_tensors="pt", truncation=True, max_length=512).to(device)
                    outputs = model.generate(input_ids, max_length=256, num_beams=10, early_stopping=True)
                    fixed_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
                elif model_type == "codebert":
                    input_ids = tokenizer.encode(input_code, return_tensors="pt", truncation=True, max_length=256).to(device)
                    source_mask = input_ids.ne(1).to(device)
                    outputs = model(source_ids=input_ids, source_mask=source_mask)
                    fixed_code = tokenizer.decode(outputs[0][0].tolist(), skip_special_tokens=True)
                else:
                    fixed_code = "无效的模型类型！"

                return render_template("index.html", fixed_code=fixed_code, input_code=input_code, model_type=model_type)
            except Exception as e:
                return render_template("index.html", message=f"代码修复失败：{str(e)}")

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)