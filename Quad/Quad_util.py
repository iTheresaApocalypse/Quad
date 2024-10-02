
from tqdm import tqdm
import json
import torch
from typing import Any
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import pandas as pd
import torch


def get_cluster_neighbour(cluster_size, threshold=0.2):
    print("getting cluster neighbour")
    cluster_neighbour = {}#retrieve the neighbors of each cluster.
    df = pd.read_csv('/mnt/hwfile/opendatalab/zhonghuaping.p/slimpajam-matrix.csv')
    cluster_distance = df.values

    for i in tqdm(range(cluster_size)):
        max_distance = 0
        cluster_neighbour[i] = []
        for j in range(cluster_size):
            max_distance = max(max_distance, cluster_distance[i][j])
            if cluster_distance[i][j] < threshold:
                cluster_neighbour[i].append(j)
    return cluster_neighbour, cluster_distance


def load_model(model_name_or_path: str,
               torch_dtype: Any = torch.bfloat16) -> Any:
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    return model, tokenizer


def sorted_highest_score(cluster_ucb, k):
    #return the top k keys with the highest values in the dictionary
    sorted_items = sorted(cluster_ucb.items(), key=lambda x: x[1], reverse=True)
    sorted_keys = [item[0] for item in sorted_items]
    return sorted_keys

def sample_minibatch(minibatch_index, batchsize):
    result = []
    for i in minibatch_index:
        for key, value in i.items():
            cluster_number = int(key)
            cluster_chose = int(value)
        all_datasets = []
        with open("/mnt/hwfile/opendatalab/zhangchi/slimpajama/clustering-" + str(cluster_number).zfill(5) + ".jsonl", "r", encoding="utf-8") as file:
            for line in file:
                data = json.loads(line)
                all_datasets.append(data)
        strlist = all_datasets[cluster_chose*batchsize:(cluster_chose+1)*batchsize]
        result = result + strlist
    return result



def get_sample_index(cluster_number, real_batchsize, cluster_sample):
    all_datasets = []
    # fetch data from the local source
    with open("/mnt/hwfile/opendatalab/zhangchi/slimpajama/clustering-" + str(cluster_number).zfill(5) + ".jsonl", "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)
            all_datasets.append(data)
    if((cluster_sample[cluster_number]+1)*real_batchsize<=len(all_datasets)):
        strlist = all_datasets[cluster_sample[cluster_number]*real_batchsize:(cluster_sample[cluster_number]+1)*real_batchsize]
        return True, strlist
    else:
        return False, None