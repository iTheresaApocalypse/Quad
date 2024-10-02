import os
import json

def calculate_influence(minibatch,index):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(index)
    import torch
    from Quad_info import get_info
    from Quad_matching import matching
    from torch.nn.functional import normalize
    from Quad_util import load_model
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    # load your model here
    model, tokenizer = load_model("deepseek-ai/deepseek-coder-1.3b-base",dtype) # please change to your own model
    model_gpu = model.to(device)
    print(model_gpu.device)
    grads, chunk_doc_dictionary = get_info(model_gpu, tokenizer, minibatch, device)
    batchsize = len(minibatch)

    for i in range(len(grads[8192])):
        if i == 0:
            result = grads[8192][i]
        else:
            result = torch.cat((result, grads[8192][i]), dim=0)
    result = normalize(result,dim=1)

    # step2: compute the dot product of the training set gradients and the validation set gradients
    cluster_reward =  matching(result, device).tolist()
    #map the chunk scores to document scores
    score = {i:0 for i in range(batchsize)}
    count = {i:0 for i in range(batchsize)}
    for i in range(len(cluster_reward)):
        #the index of chunk_doc_dictionary starts from 1
        score[chunk_doc_dictionary[i+1]-1] += cluster_reward[i]
        count[chunk_doc_dictionary[i+1]-1] += 1
    final_result = []
    for i in range(batchsize):
        if count[i] == 0:
            final_result.append(0)
        else:
            final_result.append(score[i]/count[i])
    
    print(final_result)
    with open("./output" + str(index) + ".json", 'w') as f:
        json.dump(final_result,f)
