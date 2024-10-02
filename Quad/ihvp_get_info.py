import random
import argparse
import os
from typing import Any
import trak
import torch
import sys
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn
from K-FAC import get_ihvp
from get_training_dataset import get_training_dataset
from get_validation_dataset import (get_dataloader, get_dataset)

import glob
'''
这段代码是一个用于获取预训练模型、LoRA模型或PEFT初始化模型的梯度或表示的脚本。它支持从给定的任务中获取这些信息，包括梯度、表示和损失。以下是代码的详细解释：

功能概述
加载模型：根据给定的模型名称或路径加载模型。
处理参数：通过命令行参数指定任务、训练文件、信息类型、模型路径等。
数据集和加载器：根据任务或训练文件获取数据集和加载器。
收集信息：根据指定的信息类型（梯度、表示或损失）收集信息并保存到指定路径。
'''

print(torch.cuda.is_available())
print(torch.version.cuda)
print(trak.test_install(use_fast_jl=True))
current_path = os.path.dirname(os.path.realpath(__file__))
print(os.path.join(current_path))
file_path = 'dataset/slimpajama-10'
print(os.path.exists(file_path))


'''
如果模型是PEFT模型，则加载PEFT配置和基础模型，然后加载PEFT模型。
否则，直接加载基础模型。
'''
def load_model(model_name_or_path: str,
               torch_dtype: Any = torch.bfloat16) -> Any:
    """
    Load a model from a given model name or path.

    Args:
        model_name_or_path (str): The name or path of the model.
        torch_dtype (Any, optional): The torch data type. Defaults to torch.bfloat16.

    Returns:
        Any: The loaded model.
    """

    is_peft = os.path.exists(os.path.join(
        model_name_or_path, "adapter_config.json"))
    print(f"is_peft:{is_peft}")
    if is_peft:
        # load this way to make sure that optimizer states match the model structure
        config = LoraConfig.from_pretrained(model_name_or_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path, torch_dtype=torch_dtype, device_map="auto")
        model = PeftModel.from_pretrained(
            base_model, model_name_or_path, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=True)

    for name, param in model.named_parameters():
        if 'lora' in name or 'Lora' in name:
            param.requires_grad = True
    return model

def return_jsonl_list(path):
    jsonl_list = glob.glob(os.path.join(path, '*.jsonl'))
    return jsonl_list
'''
使用argparse库解析命令行参数。
根据参数加载模型、数据集和加载器。
'''
parser = argparse.ArgumentParser(
    description='Script for getting validation gradients')
parser.add_argument('--task', type=str, default=None,
                    help='Specify the task from bbh, tydiqa or mmlu. One of variables of task and train_file must be specified')
parser.add_argument("--train_files", type=str, nargs='*',
                    default=["dataset/lambada-1.jsonl"], help="The path of the training data file we'd like to obtain the gradients/representations for. One of variables of task and train_file must be specified")
parser.add_argument(
    "--info_type", default="grads", choices=["grads", "reps", "loss"], help="The type of information")
parser.add_argument("--model_path", type=str,
                    default="5000_hf", help="The path to the model")
parser.add_argument("--max_samples", type=int,
                    default=None, help="The maximum number of samples")
parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                    choices=["float32", "bfloat16"], help="The torch data type")
parser.add_argument("--output_path", type=str,
                    default="output_op", help="The path to the output")
parser.add_argument("--data_dir", type=str,
                    default=None, help="The path to the data")
parser.add_argument("--gradient_projection_dimension", nargs='+',
                    help="The dimension of the projection, can be a list", type=int, default=[8192])
parser.add_argument("--gradient_type", type=str, default="sgd",
                    choices=["adam", "sign", "sgd"], help="The type of gradient")
parser.add_argument("--chat_format", type=str,
                    default="tulu", help="The chat format")
parser.add_argument("--use_chat_format", type=bool,
                    default=True, help="Whether to use chat format")
parser.add_argument("--max_length", type=int, default=2048,
                    help="The maximum length")
parser.add_argument("--zh", default=False, action="store_true",
                    help="Whether we are loading a translated chinese version of tydiqa dev data (Only applicable to tydiqa)")
parser.add_argument("--initialize_lora", default=False, action="store_true",
                    help="Whether to initialize the base model with lora, only works when is_peft is False")
parser.add_argument("--lora_r", type=int, default=8,
                    help="The value of lora_r hyperparameter")
parser.add_argument("--lora_alpha", type=float, default=32,
                    help="The value of lora_alpha hyperparameter")
parser.add_argument("--lora_dropout", type=float, default=0.1,
                    help="The value of lora_dropout hyperparameter")
parser.add_argument("--lora_target_modules", nargs='+', default=[
                    "q_proj", "k_proj", "v_proj", "o_proj"],  help="The list of lora_target_modules")
parser.add_argument("--processing_num_workers", type=int, default=1,
                    help="The list of lora_target_modules")



args = parser.parse_args()
print("*******************")
'''
print(os.path.exists(args.train_files))
items = os.listdir(args.train_files)
for item in items:
    print(item)
'''
print("*******************")

'''
如果指定了任务，则根据任务获取数据集和加载器。
如果指定了训练文件，则根据训练文件获取数据集和加载器。
'''
assert args.task is not None or args.train_files is not None
for train_file in args.train_files:
    train_file = ''.join(train_file)
tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
dtype = torch.float16 if args.torch_dtype == "float16" else torch.bfloat16
model = load_model(args.model_path, dtype)
print(model)
layer_count = 0

# pad token is not added by default for pretrained models
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<pad>"})

# resize embeddings if needed (e.g. for LlamaTokenizer)
embedding_size = model.get_input_embeddings().weight.shape[0]
if len(tokenizer) > embedding_size:
    model.resize_token_embeddings(len(tokenizer))

if args.initialize_lora:
    assert not isinstance(model, PeftModel)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
    )
    model = get_peft_model(model, lora_config)

if isinstance(model, PeftModel):
    model.print_trainable_parameters()

if args.task is not None:
    print("validation")
    dataset = get_dataset(args.task,
                          data_dir=args.train_files,
                          tokenizer=tokenizer,
                          chat_format=args.chat_format,
                          use_chat_format=args.use_chat_format,
                          max_length=args.max_length,
                          zh=args.zh)
    print(dataset)
    dataloader = get_dataloader(dataset, tokenizer=tokenizer)
else:
    print("training")
    assert args.train_files is not None
    dataset = get_training_dataset(all_file_paths=args.train_files, 
                               tokenizer=tokenizer, 
                               max_seq_length=args.max_length, 
                               sample_percentage=1.0, 
                               processing_num_workers=args.processing_num_workers)
    chunk_doc_dict = []
    chunk_id = 0
    for example in dataset:
        chunk_id = chunk_id + 1
        chunk_doc_dict.append({chunk_id:example['ids']})
    output_dir = args.output_path
    outfile = os.path.join(output_dir, f"chunk_doc_dict.pt")
    torch.save(chunk_doc_dict, outfile)
    print(dataset)


    dataset = dataset.remove_columns("ids")

    dataloader = get_dataloader(dataset, tokenizer=tokenizer, batch_size=1)

get_ihvp(dataloader, model, args.output_path, proj_dim=[8192])

