import torch

def calculate_influence_score(training_info: torch.Tensor, validation_info: torch.Tensor):
    """Calculate the influence score"""
    influence_scores = torch.matmul(
        training_info, validation_info.transpose(0, 1))
    return influence_scores

def matching(result, device):
    validation_info = torch.load("/mnt/petrelfs/zhangchi/LESS-main/grads/deepseekcoder_>512-slimpajama-30000_old/gsm8k-ckpt30000-sgd-folderlambada-266/dim8192/all_orig.pt") # the path where ihvp is stored
    if not torch.is_tensor(validation_info):
        validation_info = torch.tensor(validation_info)
    validation_info = validation_info.to(device).float()
    training_info = result
    training_info = training_info.to(device).float()
    influence_score = calculate_influence_score(training_info=training_info, validation_info=validation_info)
    influence_score = influence_score.reshape(influence_score.shape[0], -1).mean(-1)
    return influence_score
