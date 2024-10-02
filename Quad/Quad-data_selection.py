import math
import json
from tqdm import tqdm
import multiprocessing as mp
import random
from Quad_calculate_influence import calculate_influence
from Quad_util import get_cluster_neighbour,load_model, sorted_highest_score, sample_minibatch, get_sample_index

import os

alpha = 0.0001
cluster_size = 9873
batchsize = 10 #How many samples to draw from each cluster to estimate the overall influence score of that cluster.
threshold = 0.2
batch = 50 #How many clusters to sample from
num_gpu = 5 #How many GPUs to use for parallel computation.
real_batchsize = 20000 #The number of samples actually drawn from a certain cluster.

cluster_score = {i: 0 for i in range(0, cluster_size)}#The reward score for each cluster.
cluster_chose = {i: 0 for i in range(0, cluster_size)}#How many times each cluster has been selected (including selections of neighboring clusters)
cluster_ucb = {i: 0 for i in range(0, cluster_size)}#The Cluster Score(CS) for each cluster
cluster_sample = {i: 0 for i in range(0, cluster_size)}#The actual number of times each cluster has been sampled
final_chose = {i: 0 for i in range(0, cluster_size)}#Which clusters were ultimately selected
forbidden_cluster = []#All the data from a certain cluster has been taken
sum_chose = 0


def get_cluster_reward(topk):

    processes = []
    result = []
    chunk = int(batch/num_gpu)
    gpu_ids_str = os.environ.get('CUDA_VISIBLE_DEVICES')
    gpus = gpu_ids_str.split(',') if gpu_ids_str else []
    print(gpus)
    for index in range(num_gpu):
        minibatch_index = topk[index*chunk:(index+1)*chunk]
        minibatch = list(sample_minibatch(minibatch_index, batchsize))
        process = mp.Process(target=calculate_influence, args=(minibatch, gpus[index]))
        processes.append(process)
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    for index in range(num_gpu):
        with open("./output" + str(gpus[index]) + ".json", 'r') as f:
            data = json.load(f)
        result = result + data
    return result

def step1(k):
    merged_minibatch = []
    topk = []
    sorted_keys = sorted_highest_score(cluster_ucb, batch)
    loopindex = 0
    for i in range(batch):
        flag = False
        while flag == False:
            if k == 0:
                sample_index = loopindex
                loopindex = loopindex + 1
            else:
                sample_index = sorted_keys[loopindex]
                loopindex = loopindex + 1
            flag, minibatch = get_sample_index(sample_index, real_batchsize, cluster_sample)
            if flag == False:
                #If all samples in the cluster have been sampled
                cluster_score[sample_index] = -9999999
                cluster_ucb[sample_index] = -9999999
        merged_minibatch = merged_minibatch + minibatch
        topk.append({sample_index:cluster_sample[sample_index]})
    print("topk" + str(topk))
    print("len(merged_minibatch)" + str(len(merged_minibatch)))
    return topk


def step2(topk):
    reward = get_cluster_reward(topk)
    averages = []
    for i in range(0, len(reward), batchsize):
        part = reward[i:i+batchsize]
        if part:  # To prevent empty sections from occurring
            if len(part) == 0:
                averages.append(avg)
            else:
                avg = sum(part) / len(part)-0.0020
                averages.append(avg)
    print(averages)
    return averages

def step3(topk, cluster_neighbour, cluster_distance, averages, final_sum_chose, sum_average, sum_chose):
    idict_index = 0
    for i_dict in topk:
        for key, value in i_dict.items():
            i = int(key)
            cluster_sample[i] += 1
        if averages[idict_index] > 0:#Cluster has a sufficiently large positive impact on the model's performance.
            final_chose[i] += 1
            final_sum_chose += 1
            sum_average += averages[idict_index]
        for cluster in cluster_neighbour[i]:#Update the reward for each cluster
            sum_chose += 1
            cluster_score[cluster] += averages[idict_index]*(1-cluster_distance[(i,cluster)]/threshold)
            cluster_chose[cluster] += 1
        idict_index = idict_index + 1
    for key in cluster_ucb.keys():#Update the cluster score
        if cluster_chose[key] != 0:
            cluster_average_reward = cluster_score[key]/cluster_chose[key]
        else:
            cluster_average_reward = 0
        ucb = alpha*math.sqrt(2*(math.log(float(sum_chose)))/float(cluster_chose[key]+1))
        cluster_ucb[key] = cluster_average_reward + ucb
    return final_sum_chose, sum_average, sum_chose

def main():
    mp.set_start_method('spawn', force=True)
    final_sum_chose = 0
    iteration = 200
    sum_average = 0
    cluster_neighbour, cluster_distance = get_cluster_neighbour(cluster_size, threshold)
    sum_chose = 0
    for k in tqdm(range(iteration)):
        # step1 get mini-batch data
        topk = step1(k)
        # step2 calculate influence score
        averages = step2(topk)

        # step3 update ucb score
        final_sum_chose, sum_average, sum_chose = step3(topk, cluster_neighbour, cluster_distance, averages, final_sum_chose, sum_average, sum_chose)
        
        if final_sum_chose*real_batchsize >= 10000000:
            break
        print(float(sum_average/final_sum_chose))
        print(cluster_sample)
        print(final_chose)
        print(cluster_ucb)
        print(cluster_score)
        
    final_data = []
    for i in tqdm(range(cluster_size)):
        if final_chose[i]!=0:
            all_datasets = []
            # get local data
            with open("../data/cluster/clustering-" + str(i).zfill(5) + ".jsonl", "r", encoding="utf-8") as file:
                for line in file:
                    all_datasets.append(json.loads(line))
            final_data = final_data + all_datasets[0:real_batchsize*final_chose[i]]
    random.shuffle(final_data)
    with open('./Quad-selected_data.jsonl', 'w') as file:
        for item in final_data:
            json.dump(item, file)
            file.write('\n')

if __name__ == "__main__":
    main()
