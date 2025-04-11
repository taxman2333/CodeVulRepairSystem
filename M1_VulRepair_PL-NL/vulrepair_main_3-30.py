# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division, print_function
import argparse
import logging
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from transformers import ( 
    get_linear_schedule_with_warmup, 
    T5ForConditionalGeneration, 
    RobertaTokenizer,
    T5Config
)
from torch.optim import AdamW
from tqdm import tqdm
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import datasets
from sklearn.model_selection import train_test_split
import datetime
import re
from difflib import SequenceMatcher


cpu_cont = 16
logger = logging.getLogger(__name__)

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_ids,
                 label,
                 decoder_input_ids):
        self.input_ids = input_ids
        self.label=label
        self.decoder_input_ids = decoder_input_ids
        

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, train_data=None, val_data=None, file_type="train"):
        if file_type == "train":
            sources = train_data["source"].tolist()
            labels = train_data["target"].tolist()
        elif file_type == "eval":
            sources = val_data["source"].tolist()
            labels = val_data["target"].tolist()
        elif file_type == "test":
            data = datasets.load_dataset("MickyMike/cvefixes_bigvul", split="test")
            sources = data["source"]
            labels = data["target"]
        self.examples = []
        for i in tqdm(range(len(sources))):
            self.examples.append(convert_examples_to_features(sources[i], labels[i], tokenizer, args))
        if file_type == "train":
            for example in self.examples[:3]:
                    logger.info("*** Example ***")
                    logger.info("label: {}".format(example.label))
                    logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))
                    logger.info("decoder_input_ids: {}".format(' '.join(map(str, example.decoder_input_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):       
        return self.examples[i].input_ids, self.examples[i].input_ids.ne(0), self.examples[i].label, self.examples[i].decoder_input_ids


def convert_examples_to_features(source, label, tokenizer, args):
    # encode - subword tokenize
    source_ids = tokenizer.encode(source, truncation=True, max_length=args.encoder_block_size, padding='max_length', return_tensors='pt')
    decoder_input_ids = tokenizer.encode(label, truncation=True, max_length=args.decoder_block_size, padding='max_length', return_tensors='pt')
    label = tokenizer.encode(label, truncation=True, max_length=args.decoder_block_size, padding='max_length', return_tensors='pt')
    return InputFeatures(source_ids, label, decoder_input_ids)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def train(args, train_dataset, model, tokenizer, eval_dataset):
    """ Train the model """
    # build dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=0)
    
    args.max_steps = args.epochs * len(train_dataloader)

    # evaluate model per epoch
    args.save_steps = len(train_dataloader) * 1
   
    args.warmup_steps = args.max_steps // 5
    model.to(args.device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)
    
    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//max(args.n_gpu, 1))
    logger.info("  Total train batch size = %d",args.train_batch_size*args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)
    
    global_step = 0
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_loss = float('inf')  # 初始化最佳损失
    patience = 5  # 早停的容忍次数
    no_improve_epochs = 0  # 连续未改进的 epoch 计数

    writer_path = "tb/codet5_training_loss"
    writer = SummaryWriter(writer_path)

    model.zero_grad()

    for idx in range(args.epochs): 
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            (input_ids, attention_mask, labels, decoder_input_ids) = [x.squeeze(1).to(args.device) for x in batch]
            model.train()
            # the forward function automatically creates the correct decoder_input_ids
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss
            avg_loss = round(train_loss/tr_num,5)
            bar.set_description("epoch {} loss {}".format(idx,avg_loss))
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  
                global_step += 1
                output_flag = True
                avg_loss = round(np.exp((tr_loss - logging_loss) /(global_step- tr_nb)),4)
                if global_step % args.save_steps == 0:
                    # placeholder of evaluation
                    eval_loss = evaluate(args, model, tokenizer, eval_dataset, eval_when_training=True)    
                    # Save model checkpoint
                    if eval_loss < best_loss:
                        best_loss = eval_loss
                        no_improve_epochs = 0  # 重置未改进计数
                        logger.info("  "+"*"*20)  
                        logger.info("  Best Loss:%s",round(best_loss,4))
                        logger.info("  "+"*"*20)                          
                        checkpoint_prefix = 'checkpoint-best-loss'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)                        
                        model_to_save = model.module if hasattr(model,'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format(args.model_name)) 
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)
                    else:
                        no_improve_epochs += 1
                        logger.info(f"No improvement for {no_improve_epochs} epoch(s).")

        # 检查是否需要早停
        if no_improve_epochs >= patience:
            logger.info("Early stopping triggered.")
            break

    # 保存分词器
    tokenizer_save_path = os.path.join(args.output_dir, "tokenizer")
    if not os.path.exists(tokenizer_save_path):
        os.makedirs(tokenizer_save_path)
    tokenizer.save_pretrained(tokenizer_save_path)
    logger.info(f"Tokenizer saved to {tokenizer_save_path}")

def clean_tokens(tokens):
    tokens = tokens.replace("<pad>", "")
    tokens = tokens.replace("<s>", "")
    tokens = tokens.replace("</s>", "")
    tokens = tokens.strip("\n")
    tokens = tokens.strip()
    return tokens

def normalize_code(code):
    """Normalize code by:
    1. Removing comments
    2. Standardizing whitespace
    3. Removing extra blank lines
    """
    # Remove single-line comments
    code = re.sub(r'//.*?\n', '\n', code)
    # Remove multi-line comments
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    # Standardize whitespace
    code = re.sub(r'\s+', ' ', code)
    # Remove trailing whitespace
    code = code.strip()
    return code

def evaluate(args, model, tokenizer, eval_dataset, eval_when_training=False):
    #build dataloader
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=0)
    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)
    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()
    
    eval_loss, num = 0, 0
    bar = tqdm(eval_dataloader, total=len(eval_dataloader))
    for batch in bar:
        (input_ids, attention_mask, labels, decoder_input_ids) = [x.squeeze(1).to(args.device) for x in batch]
        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
        if args.n_gpu > 1:
            loss = loss.mean()
        eval_loss += loss.item()
        num += 1
    eval_loss = round(eval_loss/num,5)
    model.train()
    logger.info("***** Eval results *****")
    logger.info(f"Evaluation Loss: {str(eval_loss)}")
    return eval_loss

def test(args, model, tokenizer, test_dataset, best_threshold=0.5):
    # build dataloader
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size, num_workers=0)
    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # Test!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    nb_eval_steps = 0
    model.eval()
    
    # Initialize metrics
    exact_match_scores = []
    edit_distance_scores = []
    combined_scores = []
    
    bar = tqdm(test_dataloader, total=len(test_dataloader))
    for batch in bar:
        (input_ids, attention_mask, labels, decoder_input_ids) = [x.squeeze(1).to(args.device) for x in batch]
        with torch.no_grad():
            beam_outputs = model.generate(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        do_sample=False,
                                        num_beams=args.num_beams,
                                        num_return_sequences=args.num_beams,
                                        max_length=args.decoder_block_size)
        
        beam_outputs = beam_outputs.detach().cpu().tolist()
        decoder_input_ids = decoder_input_ids.detach().cpu().tolist()
        
        # Get ground truth
        ground_truth = tokenizer.decode(decoder_input_ids[0], skip_special_tokens=False)
        ground_truth = clean_tokens(ground_truth)
        ground_truth_norm = normalize_code(ground_truth)
        
        # Process predictions
        best_pred = None
        best_score = -1
        
        for single_output in beam_outputs:
            prediction = tokenizer.decode(single_output, skip_special_tokens=False)
            prediction = clean_tokens(prediction)
            prediction_norm = normalize_code(prediction)
            
            # Calculate individual scores
            exact_match = 1.0 if prediction == ground_truth else 0.0
            edit_distance = SequenceMatcher(None, prediction_norm, ground_truth_norm).ratio()
            
            # Combined score (weighted average)
            combined = 0.6 * exact_match + 0.4 * edit_distance
            
            if combined > best_score:
                best_score = combined
                best_pred = prediction
                best_exact = exact_match
                best_edit = edit_distance
        
        # Record scores for this sample
        exact_match_scores.append(best_exact)
        edit_distance_scores.append(best_edit)
        combined_scores.append(best_score)
        
        nb_eval_steps += 1
    
    # Calculate final metrics
    exact_match_accuracy = np.mean(exact_match_scores)
    edit_distance_accuracy = np.mean(edit_distance_scores)
    combined_accuracy = np.mean(combined_scores)
    
    logger.info("***** Test results *****")
    logger.info(f"Exact Match Accuracy: {exact_match_accuracy:.4f}")
    logger.info(f"Edit Distance Accuracy: {edit_distance_accuracy:.4f}")
    logger.info(f"Combined Accuracy: {combined_accuracy:.4f}")
    
    # Save results
    output_dir = "./test_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    results_df = pd.DataFrame({
        "ExactMatch": exact_match_scores,
        "EditDistance": edit_distance_scores,
        "CombinedScore": combined_scores
    })
    results_df.to_csv(os.path.join(output_dir, "evaluation_metrics.csv"), index=False)
    
    return combined_accuracy

def setup_logging_with_command(args, mode="train"):
    """
    设置日志文件路径和格式，并在日志文件开头记录完整的命令。
    mode: "train" 或 "test"，用于区分训练和测试日志。
    """
    # 获取当前时间（包含分钟信息）
    current_time = datetime.datetime.now().strftime("%m-%d-%Hh%Mm")
    log_dir = f"./{mode}_log"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        if not os.path.exists(log_dir):  # 再次检查文件夹是否成功创建
            raise RuntimeError(f"Failed to create log directory: {log_dir}")
    
    # 根据模式设置日志文件名
    log_file = os.path.join(log_dir, f"{current_time}-{mode}.log")
    
    # 构建命令行参数字符串（包含所有参数及其值）
    command_args = f"python vulrepair_main.py \\\n"
    for arg, value in vars(args).items():
        if value is not None and value is not False:  # 忽略 None 和 False 的参数
            command_args += f"    --{arg}={value} \\\n"
    command_args = command_args.strip("\\\n")  # 去掉最后的换行符和反斜杠

    # 配置日志格式
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO
    )

    # 创建日志记录器
    logger = logging.getLogger()
    logger.handlers = []  # 清空之前的日志处理器

    # 添加文件日志
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s -   %(message)s'))
    logger.addHandler(file_handler)

    # 添加终端日志
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s -   %(message)s'))
    logger.addHandler(console_handler)

    # 在日志开头记录命令参数
    logger.info(command_args)
    logger.info(f"Log file: {log_file}")

    return log_file

def main():
    parser = argparse.ArgumentParser()
    # Params
    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--model_type", default="t5", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--encoder_block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--decoder_block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--num_beams", default=50, type=int,
                        help="Beam size to use when decoding.")                          
    parser.add_argument("--model_name", default="model.bin", type=str,
                        help="Saved model name.")
    parser.add_argument("--checkpoint_model_name", default="non_domain_model.bin", type=str,
                            help="Checkpoint model name.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--model_bin_path", type=str, default=None,
                    help="Optional path to local model weights. If provided, load local weights instead of HF pretrained.")  # ✅ 明确参数作用


    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epochs', type=int, default=1,
                        help="training epochs")

    args = parser.parse_args()

    # 设置日志
    if args.do_train:
        train_log_file = setup_logging_with_command(args, mode="train")
        logger.info(f"Training log saved to {train_log_file}")

    if args.do_test:
        test_log_file = setup_logging_with_command(args, mode="test")
        logger.info(f"Testing log saved to {test_log_file}")

        # Setup CUDA, GPU
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    args.n_gpu = 1
    args.device = device

    # 配置日志（保留原有部分）
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO
    )
    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu)
    
    # 设置随机种子（保留原有部分）
    set_seed(args)

    # 加载 Tokenizer（支持本地/HF双模式）
    try:
        # 优先检查本地路径是否存在
        if os.path.exists(args.tokenizer_name):
            tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
            logger.info(f"Loaded local tokenizer from {args.tokenizer_name}")
        else:
            tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
            logger.info(f"Loaded HF tokenizer: {args.tokenizer_name}")
    except Exception as e:
        logger.error(f"Tokenizer loading failed: {str(e)}")
        raise

    # 添加自定义特殊token（保留原有部分）
    tokenizer.add_tokens([
        "<S2SV_StartBug>", 
        "<S2SV_EndBug>", 
        "<S2SV_blank>", 
        "<S2SV_ModStart>", 
        "<S2SV_ModEnd>"
    ])

    # 核心修改：模型加载逻辑
    try:
        config = T5Config.from_pretrained(args.model_name_or_path)
        
        # 条件加载模型权重
        if args.model_bin_path:  # 本地权重优先
            if not os.path.exists(args.model_bin_path):
                raise FileNotFoundError(f"Model weights not found at {args.model_bin_path}")
            
            model = T5ForConditionalGeneration(config)  # 初始化空模型
            model.load_state_dict(
                torch.load(args.model_bin_path, map_location=args.device)
            )
            logger.info(f"Successfully loaded local weights from {args.model_bin_path}")
        else:  # 使用HF预训练权重
            model = T5ForConditionalGeneration.from_pretrained(
                args.model_name_or_path, 
                config=config
            )
            logger.info(f"Loaded pretrained weights from HF: {args.model_name_or_path}")
        
        # 调整词表大小（保留原有部分）
        model.resize_token_embeddings(len(tokenizer))
    except Exception as e:
        logger.error(f"Model initialization failed: {str(e)}")
        raise

    # 打印参数（保留原有部分）
    logger.info("Training/evaluation parameters %s", args)

    # 训练逻辑（保留原有部分）
    if args.do_train:
        train_data_whole = datasets.load_dataset("MickyMike/cvefixes_bigvul", split="train")
        df = pd.DataFrame({
            "source": train_data_whole["source"], 
            "target": train_data_whole["target"]
        })
        train_data, val_data = train_test_split(df, test_size=0.1238, random_state=42)
        train_dataset = TextDataset(tokenizer, args, train_data, val_data, file_type='train')
        eval_dataset = TextDataset(tokenizer, args, train_data, val_data, file_type='eval')
        train(args, train_dataset, model, tokenizer, eval_dataset)
    
    results = {}
    # 测试逻辑（移除原有权重加载代码）
    if args.do_test:
        model.to(args.device)
        test_dataset = TextDataset(tokenizer, args, file_type='test')
        test_accuracy = test(args, model, tokenizer, test_dataset, best_threshold=0.5)
        results["test_accuracy"] = test_accuracy

    return results

if __name__ == "__main__":
    main()