import contextlib
import numpy as np
import torch
from datasets import Dataset
from encode_dataset import encode_with_content_format
from tool_function import load_train_files
from unify_data_format import get_unify_dataset
from tqdm import tqdm
import json

import logging
logging.basicConfig(
    format='%(asctime)s %(filename)s:%(lineno)s [%(levelname)s] %(message)s', level=logging.INFO)


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

def get_training_dataset(all_file_paths,  tokenizer=None, max_seq_length=1024, sample_percentage=1.0, seed=0):
    lm_datasets = []
    train_file_paths = load_train_files(all_file_paths)
    count = 0#计算ids
    for filepath in train_file_paths:
        raw_datasets = load_raw_dataset(filepath, sample_percentage=sample_percentage, seed=seed)
        unified_datasets, count = get_unify_dataset(raw_datasets,count)
        temp_datasets = encode_data(unified_datasets, tokenizer, max_seq_length)
        lm_datasets = lm_datasets + temp_datasets
    result_datasets = Dataset.from_list(lm_datasets)
    result_datasets.set_format(type="pt")
    print(result_datasets)
    return result_datasets


def load_raw_dataset(file_path, sample_size=None, sample_percentage=1.0, seed=0):
    """ load raw dataset """
    processed_datasets = []
    with open(file_path, 'r', encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            processed_datasets.append(data)
    processed_datasets = Dataset.from_list(processed_datasets)
    
    if sample_size is None:
        sample_size = int(len(processed_datasets) * sample_percentage)

    if sample_size == len(processed_datasets):
        return processed_datasets  # not shuffle

    with temp_seed(seed):
        index = np.random.permutation(len(processed_datasets))[:sample_size]
    sampled_dataset = processed_datasets.select(index)

    return sampled_dataset


def encode_data(raw_datasets, tokenizer, max_seq_length):
    tokenized_list = []

    for example in tqdm(raw_datasets):
        gen = encode_with_content_format(example, tokenizer, max_seq_length)#chunk & tokenize
        for result in gen:
            tokenized_list.append(result)
    return tokenized_list

def get_dataloader(dataset, tokenizer, batch_size=1):
    data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, padding="longest") 
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,  # When getting gradients, we only do this single batch process
                            collate_fn=data_collator)
    print("There are {} examples in the dataset".format(len(dataset)))
    return dataloader
