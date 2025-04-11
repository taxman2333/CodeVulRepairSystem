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
    AdamW, 
    get_linear_schedule_with_warmup, 
    T5ForConditionalGeneration, 
    RobertaTokenizer,
    T5Config,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,  # 新增导入
    Qwen2Tokenizer  # 手动导入 Qwen2Tokenizer
)
from tqdm import tqdm
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast
import datasets
from sklearn.model_selection import train_test_split
import safetensors.torch
from datetime import datetime


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

def save_model(model, output_dir, model_name):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    model_path = os.path.join(output_dir, model_name)
    if model_name.endswith('.safetensors'):
        safetensors.torch.save_file(model_to_save.state_dict(), model_path)
    else:
        torch.save(model_to_save.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")

def load_model(model, model_weights_dir, device):
    if os.path.isdir(model_weights_dir):
        model = T5ForConditionalGeneration.from_pretrained(model_weights_dir)
    else:
        raise FileNotFoundError(f"Model weights directory not found at {model_weights_dir}")
    model.to(device)
    logger.info(f"Model loaded from {model_weights_dir}")

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
    best_loss = 100

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
                        logger.info("  "+"*"*20)  
                        logger.info("  Best Loss:%s",round(best_loss,4))
                        logger.info("  "+"*"*20)                          
                        timestamp = datetime.now().strftime("%Y-%m-%d-%Hh")
                        output_dir = os.path.join(args.output_dir, f'saved_model_weight/{timestamp}')
                        save_model(model, output_dir, args.model_name)

def clean_tokens(tokens):
    tokens = tokens.replace("<pad>", "")
    tokens = tokens.replace("<s>", "")
    tokens = tokens.replace("</s>", "")
    tokens = tokens.strip("\n")
    tokens = tokens.strip()
    return tokens

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
    accuracy = []
    raw_predictions = []
    correct_prediction = ""
    bar = tqdm(test_dataloader, total=len(test_dataloader))
    for batch in bar:
        correct_pred = False
        (input_ids, attention_mask, labels, decoder_input_ids)=[x.squeeze(1).to(args.device) for x in batch]
        # 确保从 DataParallel 中提取实际模型
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        # 设置分词器的填充方式
        tokenizer.padding_side = "left"

        with autocast():
            beam_outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,  # 禁用采样模式
                num_beams=args.num_beams,
                num_return_sequences=args.num_beams,
                max_new_tokens=args.decoder_block_size
            )

        beam_outputs = beam_outputs.detach().cpu().tolist()
        decoder_input_ids = decoder_input_ids.detach().cpu().tolist()
        for single_output in beam_outputs:
            # pred
            prediction = tokenizer.decode(single_output, skip_special_tokens=False)
            prediction = clean_tokens(prediction)
            # truth
            ground_truth = tokenizer.decode(decoder_input_ids[0], skip_special_tokens=False)
            ground_truth = clean_tokens(ground_truth)
            if prediction == ground_truth:
                correct_prediction = prediction
                correct_pred = True
                break
        if correct_pred:
            raw_predictions.append(correct_prediction)
            accuracy.append(1)
        else:
            # if not correct, use the first output in the beam as the raw prediction
            raw_pred = tokenizer.decode(beam_outputs[0], skip_special_tokens=False)
            raw_pred = clean_tokens(raw_pred)
            raw_predictions.append(raw_pred)
            accuracy.append(0)
        nb_eval_steps += 1
    # calculate accuracy
    test_result = round(sum(accuracy) / len(accuracy), 4)
    logger.info("***** Test results *****")
    logger.info(f"Test Accuracy: {str(test_result)}")

    # write prediction to file
    df = pd.DataFrame({"raw_predictions": [], "correctly_predicted": []})
    df["raw_predictions"] = raw_predictions
    df["correctly_predicted"] = accuracy
    timestamp = datetime.now().strftime("%Y-%m-%d-%Hh")
    result_path = os.path.join(args.output_dir, f'test_results/{timestamp}')
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    df.to_csv(os.path.join(result_path, "VulRepair_raw_preds.csv"))
    logger.info(f"Test results saved to {result_path}")

def load_local_dataset(file_path):
    return pd.read_csv(file_path)

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
    parser.add_argument("--model_path", default=None, type=str,
                        help="The path to the local model directory containing config, tokenizer, and weights.")
    parser.add_argument("--model_weights_dir", type=str, default=None,
                        help="Directory containing local model weights in .bin or .safetensors format.")
    parser.add_argument("--train_data_path", default="/home/scq/VulRepair/dataset/train.csv", type=str,
                        help="Path to the local train dataset.")
    parser.add_argument("--test_data_path", default="/home/scq/VulRepair/dataset/test.csv", type=str,
                        help="Path to the local test dataset.")

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

    # Setup CUDA, GPU
    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
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

    # 加载 Tokenizer（根据模型类型加载）
    try:
        if args.model_type.lower() == "qwen2":
            tokenizer = Qwen2Tokenizer.from_pretrained(args.model_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        logger.info(f"Loaded tokenizer from {args.model_path}")
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
    # 核心修改：模型加载逻辑
    try:
        config = AutoConfig.from_pretrained(args.model_path)
        
        # 根据模型结构自动选择模型类
        if config.is_encoder_decoder:
            model_class = AutoModelForSeq2SeqLM
        else:
            model_class = AutoModelForCausalLM
        
        # 条件加载模型权重
        if args.model_weights_dir:  # 本地权重优先
            if not os.path.exists(args.model_weights_dir):
                raise FileNotFoundError(f"Model weights directory not found at {args.model_weights_dir}")
            
            model = model_class.from_pretrained(
                args.model_weights_dir,
                config=config,
                trust_remote_code=True  # 对于自定义模型如Qwen2需要此参数
            )
        else:
            raise ValueError("No model weights directory provided.")
        
        # 调整词表大小（保留原有部分）
        model.resize_token_embeddings(len(tokenizer))
    except Exception as e:
        logger.error(f"Model initialization failed: {str(e)}")
        raise

    # try:
    #     config = AutoConfig.from_pretrained(args.model_path)
    #     if args.model_type.lower() == "t5":
    #         model_class = AutoModelForSeq2SeqLM
    #     elif args.model_type.lower() == "qwen2":
    #         model_class = AutoModelForCausalLM  # 使用 AutoModelForCausalLM 加载 Qwen2 模型
    #     else:
    #         model_class = AutoModel  # 默认使用 AutoModel
        
    #     # 条件加载模型权重
    #     if args.model_weights_dir:  # 本地权重优先
    #         if not os.path.exists(args.model_weights_dir):
    #             raise FileNotFoundError(f"Model weights directory not found at {args.model_weights_dir}")
            
    #         model = model_class.from_pretrained(args.model_weights_dir, config=config)
    #     else:
    #         raise ValueError("No model weights directory provided.")
        
    #     # 调整词表大小（保留原有部分）
    #     model.resize_token_embeddings(len(tokenizer))
    # except Exception as e:
    #     logger.error(f"Model initialization failed: {str(e)}")
    #     raise

    # 打印参数（保留原有部分）
    logger.info("Training/evaluation parameters %s", args)

    # 训练逻辑（保留原有部分）
    if args.do_train:
        train_data_whole = load_local_dataset(args.train_data_path)
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
        test_data = load_local_dataset(args.test_data_path)
        test_dataset = TextDataset(tokenizer, args, test_data, file_type='test')
        test(args, model, tokenizer, test_dataset, best_threshold=0.5)

    return results

if __name__ == "__main__":
    main()
