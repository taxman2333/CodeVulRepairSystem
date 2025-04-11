import argparse
import logging
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from tqdm import tqdm
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import datasets
from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast, GradScaler
from bitsandbytes.optim import AdamW as bnb_AdamW
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)

logger = logging.getLogger(__name__)

class InputFeatures(object):
    """A single training/test features for an example."""
    def __init__(self, input_ids, label, decoder_input_ids):
        self.input_ids = input_ids
        self.label = label
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
        
        # 打印前 3 个样本的简化信息
        # if file_type == "train":
        #     for example in self.examples[:3]:
        #         logger.info("*** Example ***")
        #         logger.info(f"Decoded label: {tokenizer.decode(example.label, skip_special_tokens=True)}")
        #         logger.info(f"Decoded input_ids: {tokenizer.decode(example.input_ids, skip_special_tokens=True)}")
        #         logger.info(f"Decoded decoder_input_ids: {tokenizer.decode(example.decoder_input_ids, skip_special_tokens=True)}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        input_ids = self.examples[i].input_ids.squeeze(0)
        label = self.examples[i].label.squeeze(0)
        decoder_input_ids = self.examples[i].decoder_input_ids.squeeze(0)
        return input_ids, input_ids.ne(0), label, decoder_input_ids

def convert_examples_to_features(source, label, tokenizer, args):
    source_ids = tokenizer.encode(source, truncation=True, max_length=128, padding='max_length', return_tensors='pt').squeeze(0)
    decoder_input_ids = tokenizer.encode(label, truncation=True, max_length=128, padding='max_length', return_tensors='pt').squeeze(0)
    label = tokenizer.encode(label, truncation=True, max_length=128, padding='max_length', return_tensors='pt').squeeze(0)
    
    # 仅打印前 5 个样本的形状
    #if len(source_ids) <= 5:
       #print(f"source_ids shape: {source_ids.shape}, label shape: {label.shape}")
    
    return InputFeatures(source_ids, label, decoder_input_ids)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:  # 确保 n_gpu 已初始化
        torch.cuda.manual_seed_all(args.seed)

def train(args, train_dataset, model, tokenizer, eval_dataset):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=0)
    args.max_steps = args.epochs * len(train_dataloader)
    args.warmup_steps = args.max_steps // 5
    model.to(args.device)

    # 优化器和学习率调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.warmup_steps, gamma=0.1)

    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler()

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Instantaneous batch size per GPU = {args.train_batch_size // max(args.n_gpu, 1)}")
    logger.info(f"  Total train batch size = {args.train_batch_size * args.gradient_accumulation_steps}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_steps}")

    global_step = 0
    best_loss = float('inf')
    error_logged = False  # 用于控制错误日志的输出频率

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        tr_num = 0  # 累计处理的样本数
        train_loss = 0  # 累计的训练损失
        progress_bar = tqdm(train_dataloader, total=len(train_dataloader), desc=f"epoch {epoch} loss {epoch_loss:.5f}", leave=True)

        for step, batch in enumerate(progress_bar):
            input_ids, attention_mask, labels, decoder_input_ids = [x.to(args.device) for x in batch]
            with torch.amp.autocast(device_type='cuda'):  # 启用混合精度
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                if args.n_gpu > 1:
                    loss = loss.mean()
                loss = loss / args.gradient_accumulation_steps  # 平均分配梯度

            # 检查 loss 是否为非有限值
            if not torch.isfinite(loss):
                logger.warning(f"Non-finite loss detected at step {step}. Skipping step.")
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()  # 缩放梯度
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                # 检查梯度是否存在
                if any(p.grad is None for p in model.parameters()):
                    logger.warning("Some gradients are None. Skipping unscale and optimizer step.")
                    optimizer.zero_grad()
                    continue

                scaler.unscale_(optimizer)  # 确保梯度被正确缩放
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)  # 梯度裁剪

                try:
                    scaler.step(optimizer)  # 缩放优化器步长
                    scaler.update()  # 更新缩放器
                except ValueError as e:
                    logger.error(f"Scaler step failed: {e}")
                    optimizer.zero_grad()
                    continue

                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            avg_loss = train_loss / tr_num  # 计算当前平均损失

            # 动态更新进度条的描述信息
            progress_bar.set_description(f"epoch {epoch} loss {avg_loss:.5f}")

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch + 1}/{args.epochs} completed. Loss: {avg_epoch_loss:.4f}")

        # 每轮训练结束后进行评估
        if args.evaluate_during_training:
            eval_loss = evaluate(args, model, tokenizer, eval_dataset)
            logger.info(f"Evaluation Loss: {eval_loss:.4f}")

            # 如果当前评估 Loss 是最低的，则保存模型
            if eval_loss < best_loss:
                best_loss = eval_loss
                output_dir = os.path.join(args.output_dir, 'checkpoint-best-loss')
                os.makedirs(output_dir, exist_ok=True)
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                logger.info(f"Best Loss: {best_loss:.4f}. Model saved to {output_dir}")

        # 清理显存
        torch.cuda.empty_cache()

def evaluate(args, model, tokenizer, eval_dataset):
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=0)

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    eval_loss, num = 0, 0
    for batch in tqdm(eval_dataloader):
        input_ids, attention_mask, labels, decoder_input_ids = [x.to(args.device) for x in batch]
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            if args.n_gpu > 1:
                loss = loss.mean()
            eval_loss += loss.item()
            num += 1

    eval_loss = round(eval_loss / num, 5)
    logger.info("***** Eval results *****")
    logger.info(f"Evaluation Loss: {str(eval_loss)}")
    model.train()
    return eval_loss

def test(args, model, tokenizer, test_dataset):
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size, num_workers=0)

    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    accuracy = []
    for batch in tqdm(test_dataloader):
        input_ids, attention_mask, labels, decoder_input_ids = [x.to(args.device) for x in batch]
        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=args.decoder_block_size)
            predictions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            ground_truths = [tokenizer.decode(label, skip_special_tokens=True) for label in labels]
            accuracy.extend([1 if pred == gt else 0 for pred, gt in zip(predictions, ground_truths)])

    test_result = round(sum(accuracy) / len(accuracy), 4)
    logger.info("***** Test results *****")
    logger.info(f"Test Accuracy: {str(test_result)}")

def main():
    parser = argparse.ArgumentParser()
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()  # 确保在调用 set_seed 之前赋值
    args.device = device

    logger.info(f"Using device: {device}, Number of GPUs: {args.n_gpu}")

    set_seed(args)  # 确保此时 args.n_gpu 已初始化

    try:
        # 加载模型和分词器
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

        config = AutoConfig.from_pretrained(args.model_name_or_path)
        
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config)

        model = load_checkpoint_and_dispatch(
            model,
            args.model_name_or_path,
            device_map="auto",
            dtype=torch.float16  # 或 torch.bfloat16
        )
        
        # 打印模型配置，确保与权重匹配
        logger.info(f"Model configuration: {model.config}")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    logger.info(f"Model hidden size: {model.config.hidden_size}")
    logger.info(f"Number of layers: {model.config.num_hidden_layers}")

    model.gradient_checkpointing_disable()

    print(f"Model vocab size: {model.config.vocab_size}")

    # print(model.config)
    # print(config)

    if args.do_train:
        train_data_whole = datasets.load_dataset("MickyMike/cvefixes_bigvul", split="train")
        df = pd.DataFrame({"source": train_data_whole["source"], "target": train_data_whole["target"]})
        train_data, val_data = train_test_split(df, test_size=0.1238, random_state=42)
        
        # 获取所有标签值
        all_labels = train_data["target"].tolist() + val_data["target"].tolist()
        max_label_value = max([max(tokenizer.encode(label)) for label in all_labels])
        
        print(f"Max label value in dataset: {max_label_value}")
        
        train_dataset = TextDataset(tokenizer, args, train_data, val_data, file_type="train")
        eval_dataset = TextDataset(tokenizer, args, train_data, val_data, file_type="eval")
        train(args, train_dataset, model, tokenizer, eval_dataset)

if __name__ == "__main__":
    main()