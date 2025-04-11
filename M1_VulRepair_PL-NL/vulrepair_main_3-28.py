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
from torch.cuda.amp import GradScaler
from torch.amp import autocast
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import datasets
from sklearn.model_selection import train_test_split
import datetime


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

            cache_name = f"{file_type}_cache_bs{args.train_batch_size}_enc{args.encoder_block_size}_dec{args.decoder_block_size}.pt"
            cache_path = os.path.join(args.output_dir, cache_name) if args.output_dir else f"./cache/{cache_name}"
            
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            
            if os.path.exists(cache_path):
                logger.info(f"Loading cached features from {cache_path}")
                # 安全加载修改
                torch.serialization.add_safe_globals([InputFeatures])
                self.examples = torch.load(cache_path)
            else:
                self.examples = []
                for i in tqdm(range(len(sources)), desc=f"Processing {file_type} data"):
                    if file_type == "train" and random.random() < 0.5:
                        source = self.simple_augment(sources[i])
                    else:
                        source = sources[i]
                    self.examples.append(convert_examples_to_features(source, labels[i], tokenizer, args))
                
                torch.save(self.examples, cache_path)
                logger.info(f"Cached features saved to {cache_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return (
            self.examples[idx].input_ids,
            self.examples[idx].input_ids.ne(0),
            self.examples[idx].label,
            self.examples[idx].decoder_input_ids
        )

    def simple_augment(self, code):
        lines = code.split('\n')
        return '\n'.join(
            ' ' + line if random.random() < 0.3 and line.strip() else line.lstrip()
            for line in lines
        )


def convert_examples_to_features(source, label, tokenizer, args):
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
    """ Train the model with mixed precision support """
    # Initialize mixed precision tools
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
    
    # Build dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        num_workers=4,
        pin_memory=True
    )
    
    # Calculate steps
    args.max_steps = args.epochs * len(train_dataloader)
    args.warmup_steps = args.max_steps // 5
    model.to(args.device)

    # Prepare optimizer
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        eps=args.adam_epsilon
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps
    )

    # Multi-GPU training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Log training info
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.train_batch_size}")
    logger.info(f"  Total optimization steps = {args.max_steps}")
    logger.info(f"  Using FP16 mixed precision = {args.fp16}")

    # Training variables
    global_step = 0
    best_loss = float('inf')
    patience = 5
    no_improve_epochs = 0

    for epoch in range(args.epochs):
        epoch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch}")
        model.train()
        total_loss = 0
        
        for step, batch in enumerate(epoch_iterator):
            inputs = [x.squeeze(1).to(args.device) for x in batch]
            
            # Mixed precision forward pass
            with torch.amp.autocast(device_type='cuda', enabled=args.fp16):
                outputs = model(
                    input_ids=inputs[0],
                    attention_mask=inputs[1],
                    labels=inputs[2]
                )
                loss = outputs.loss
                if args.n_gpu > 1:
                    loss = loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
            
            # Backward pass with scaling
            scaler.scale(loss).backward()
            
            # Gradient clipping and optimizer step
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
            
            total_loss += loss.item()
            epoch_iterator.set_postfix(loss=loss.item())
        
        # 每轮结束后评估
        logger.info(f"Epoch {epoch} average loss: {total_loss / len(train_dataloader):.4f}")
        eval_loss = evaluate(args, model, tokenizer, eval_dataset, eval_when_training=True)
        if eval_loss < best_loss:
            best_loss = eval_loss
            no_improve_epochs = 0
            output_dir = os.path.join(args.output_dir, "checkpoint-best")
            os.makedirs(output_dir, exist_ok=True)
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), os.path.join(output_dir, args.model_name))
            
            # 保存分词器（新增部分）
            tokenizer_save_path = os.path.join(args.output_dir, "tokenizer")
            if not os.path.exists(tokenizer_save_path):
                os.makedirs(tokenizer_save_path)
            tokenizer.save_pretrained(tokenizer_save_path)
            logger.info(f"Saved best model and tokenizer to {output_dir}")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                logger.info("Early stopping triggered")
                
                # 早停时也保存分词器（新增部分）
                tokenizer_save_path = os.path.join(args.output_dir, "tokenizer")
                if not os.path.exists(tokenizer_save_path):
                    os.makedirs(tokenizer_save_path)
                tokenizer.save_pretrained(tokenizer_save_path)
                logger.info(f"Tokenizer saved to {tokenizer_save_path} during early stopping")
                return

def clean_tokens(tokens):
    tokens = tokens.replace("<pad>", "")
    tokens = tokens.replace("<s>", "")
    tokens = tokens.replace("</s>", "")
    tokens = tokens.strip("\n")
    tokens = tokens.strip()
    tokens = ' '.join(tokens.split())
    tokens = tokens.replace('\t', ' ')
    return tokens

def evaluate(args, model, tokenizer, eval_dataset, eval_when_training=False):
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size * 2,
        num_workers=0,
        pin_memory=True
    )

    if args.n_gpu > 1 and not eval_when_training:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running evaluation *****")
    logger.info(f"  Num examples = {len(eval_dataset)}")
    logger.info(f"  Batch size = {args.eval_batch_size * 2}")
    
    model.eval()
    eval_loss = 0.0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            inputs = [x.squeeze(1).to(args.device) for x in batch]
            
            outputs = model(
                input_ids=inputs[0],
                attention_mask=inputs[1],
                labels=inputs[2]
            )
            eval_loss += outputs.loss.item()
            total += 1
    
    eval_loss = eval_loss / total
    logger.info("***** Evaluation Results *****")
    logger.info(f"  Evaluation Loss = {eval_loss:.4f}")
    
    return eval_loss

def full_evaluate(args, model, tokenizer, eval_dataset):
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        num_workers=0,
        pin_memory=True
    )

    model.eval()
    eval_loss = 0.0
    exact_match = 0
    partial_match = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Full Evaluating"):
            inputs = [x.squeeze(1).to(args.device) for x in batch]
            
            outputs = model(
                input_ids=inputs[0],
                attention_mask=inputs[1],
                labels=inputs[2]
            )
            eval_loss += outputs.loss.item()
            
            generated_ids = model.generate(
                input_ids=inputs[0],
                attention_mask=inputs[1],
                max_length=args.decoder_block_size,
                num_beams=3,
                early_stopping=True
            )
            
            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            truths = tokenizer.batch_decode(inputs[2], skip_special_tokens=True)
            
            for pred, truth in zip(preds, truths):
                total += 1
                pred = clean_tokens(pred)
                truth = clean_tokens(truth)
                if pred == truth:
                    exact_match += 1
                    partial_match += 1
                elif truth in pred or pred in truth:
                    partial_match += 1
    
    eval_loss = eval_loss / len(eval_dataloader)
    exact_match_rate = exact_match / total if total > 0 else 0
    partial_match_rate = partial_match / total if total > 0 else 0
    
    logger.info("***** Full Evaluation Results *****")
    logger.info(f"  Evaluation Loss = {eval_loss:.4f}")
    logger.info(f"  Exact Match Rate = {exact_match_rate:.4f}")
    logger.info(f"  Partial Match Rate = {partial_match_rate:.4f}")
    
    return eval_loss

def test(args, model, tokenizer, test_dataset, best_threshold=0.5):
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(
        test_dataset, 
        sampler=test_sampler, 
        batch_size=args.eval_batch_size, 
        num_workers=0
    )
    
    if args.ensemble_models:
        models = [model]
        for model_path in args.ensemble_models:
            try:
                checkpoint = torch.load(model_path, map_location=args.device)
                ensemble_model = T5ForConditionalGeneration(model.config)
                ensemble_model.load_state_dict(checkpoint)
                ensemble_model.to(args.device)
                models.append(ensemble_model)
                logger.info(f"Loaded ensemble model from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load ensemble model {model_path}: {str(e)}")
                continue
    else:
        models = [model]
    
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
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
        (input_ids, attention_mask, labels, decoder_input_ids) = [x.squeeze(1).to(args.device) for x in batch]
        all_outputs = []
        for m in models:
            with torch.no_grad():
                outputs = m.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    do_sample=True,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    num_return_sequences=args.num_beams,
                    max_length=args.decoder_block_size,
                    temperature=args.temperature
                )
                all_outputs.append(outputs)
        
        beam_outputs = all_outputs[0] if len(all_outputs) == 1 else select_best_prediction(all_outputs)
        beam_outputs = beam_outputs.detach().cpu().tolist()
        decoder_input_ids = decoder_input_ids.detach().cpu().tolist()
        
        for single_output in beam_outputs:
            prediction = tokenizer.decode(single_output, skip_special_tokens=False)
            prediction = clean_tokens(prediction)
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
            raw_pred = tokenizer.decode(beam_outputs[0], skip_special_tokens=False)
            raw_pred = clean_tokens(raw_pred)
            raw_predictions.append(raw_pred)
            accuracy.append(0)
        nb_eval_steps += 1
    
    test_result = round(sum(accuracy) / len(accuracy), 4)
    logger.info("***** Test results *****")
    logger.info(f"Test Accuracy: {str(test_result)}")

    output_dir = "./raw_predictions"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame({"raw_predictions": raw_predictions, "correctly_predicted": accuracy})
    output_path = os.path.join(output_dir, "VulRepair_raw_preds.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"Saved raw predictions to {output_path}")

def select_best_prediction(all_outputs):
    return all_outputs[0]

def setup_logging_with_command(args, mode="train"):
    current_time = datetime.datetime.now().strftime("%m-%d-%Hh%Mm")
    log_dir = f"./{mode}_log"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"{current_time}-{mode}.log")
    
    command_args = f"python vulrepair_main.py \\\n"
    for arg, value in vars(args).items():
        if value is not None and value is not False:
            command_args += f"    --{arg}={value} \\\n"
    command_args = command_args.strip("\\\n")

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO
    )

    logger = logging.getLogger()
    logger.handlers = []

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s -   %(message)s'))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s -   %(message)s'))
    logger.addHandler(console_handler)

    logger.info(command_args)
    logger.info(f"Log file: {log_file}")

    return log_file

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--model_type", default="t5", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--encoder_block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization.")
    parser.add_argument("--decoder_block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization.")
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
                        help="Optional path to local model weights.")
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
    parser.add_argument("--fp16", action="store_true", 
                        help="Enable mixed precision training")
    parser.add_argument("--label_smoothing", default=0.1, type=float,
                        help="Label smoothing factor.")
    parser.add_argument("--top_p", default=0.95, type=float,
                        help="Nucleus sampling (top-p) probability.")
    parser.add_argument("--top_k", default=50, type=int,
                        help="Top-k sampling parameter.")
    parser.add_argument("--temperature", default=0.7, type=float,
                        help="Sampling temperature.")
    parser.add_argument("--ensemble_models", nargs='+', default=None,
                        help="Paths to additional models for ensemble prediction.")

    args = parser.parse_args()

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.do_train:
        train_log_file = setup_logging_with_command(args, mode="train")
        logger.info(f"Training log saved to {train_log_file}")

    if args.do_test:
        test_log_file = setup_logging_with_command(args, mode="test")
        logger.info(f"Testing log saved to {test_log_file}")

    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    args.n_gpu = 1
    args.device = device

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO
    )
    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu)
    
    set_seed(args)

    try:
        if os.path.exists(args.tokenizer_name):
            tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
            logger.info(f"Loaded local tokenizer from {args.tokenizer_name}")
        else:
            tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
            logger.info(f"Loaded HF tokenizer: {args.tokenizer_name}")
    except Exception as e:
        logger.error(f"Tokenizer loading failed: {str(e)}")
        raise

    tokenizer.add_tokens([
        "<S2SV_StartBug>", 
        "<S2SV_EndBug>", 
        "<S2SV_blank>", 
        "<S2SV_ModStart>", 
        "<S2SV_ModEnd>"
    ])

    try:
        config = T5Config.from_pretrained(args.model_name_or_path)
        config.label_smoothing = args.label_smoothing
        
        if args.model_bin_path:
            if not os.path.exists(args.model_bin_path):
                raise FileNotFoundError(f"Model weights not found at {args.model_bin_path}")
            
            model = T5ForConditionalGeneration(config)
            model.load_state_dict(
                torch.load(args.model_bin_path, map_location=args.device)
            )
            logger.info(f"Successfully loaded local weights from {args.model_bin_path}")
        else:
            model = T5ForConditionalGeneration.from_pretrained(
                args.model_name_or_path, 
                config=config
            )
            logger.info(f"Loaded pretrained weights from HF: {args.model_name_or_path}")
        
        model.resize_token_embeddings(len(tokenizer))
    except Exception as e:
        logger.error(f"Model initialization failed: {str(e)}")
        raise

    logger.info("Training/evaluation parameters %s", args)

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
    if args.do_test:
        model.to(args.device)
        test_dataset = TextDataset(tokenizer, args, file_type='test')
        if args.evaluate_during_training:
            full_evaluate(args, model, tokenizer, test_dataset)
        else:
            test(args, model, tokenizer, test_dataset, best_threshold=0.5)

    return results

if __name__ == "__main__":
    main()