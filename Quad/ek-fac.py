import json
import os
from hashlib import md5
from typing import Iterable, List, Optional
import copy
import torch
import torch.nn.functional as F
import torch.nn as nn
from functorch import grad, make_functional_with_buffers, vmap
from peft import PeftModel
from torch import Tensor
from torch.nn.functional import normalize
from tqdm import tqdm
from trak import *
from trak.projectors import BasicProjector, CudaProjector, ProjectionType
from transformers import RobertaModel
import einops
import sys
import time


def prepare_batch(batch, device=torch.device("cuda:0")):
    """ Move the batch to the device. """
    for key in batch:
        batch[key] = batch[key].to(device)


def get_max_saved_index(output_dir: str, prefix="reps") -> int:
    """
    Retrieve the highest index for which the data (either representation or gradients) has been stored.

    Args:
        output_dir (str): The output directory.
        prefix (str, optional): The prefix of the files, [reps | grads]. Defaults to "reps".

    Returns:
        int: The maximum representation index, or -1 if no index is found.
    """
    files = [file for file in os.listdir(
        output_dir) if file.startswith(prefix)]
    index = [int(file.split(".")[0].split("-")[1])
             for file in files]  # e.g., output_dir/reps-100.pt
    return max(index) if len(index) > 0 else -1


def get_output(model,
               weights: Iterable[Tensor],
               buffers: Iterable[Tensor],
               input_ids=None,
               attention_mask=None,
               labels=None,
               ) -> Tensor:
    logits = model(weights, buffers, *(input_ids.unsqueeze(0),
                                       attention_mask.unsqueeze(0))).logits
    labels = labels.unsqueeze(0)
    loss_fct = F.cross_entropy
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1))
    return loss


def get_trak_projector(device: torch.device):
    """ Get trak projectors (see https://github.com/MadryLab/trak for details) """
    try:
        # num_sms = torch.cuda.get_device_properties(
        #     device.index).multi_processor_count
        # import fast_jl
        # # test run to catch at init time if projection goes through
        # fast_jl.project_rademacher_8(torch.zeros(
        #     8, 1_000, device=device), 512, 0, num_sms)
        projector = CudaProjector
        print(f"Using CudaProjector, device:{device}")
    except:
        projector = BasicProjector
        print("Using BasicProjector")
    return projector


def get_number_of_params(model):
    """ Make sure that only lora parameters require gradients in peft models. """
    if isinstance(model, PeftModel):
        names = [n for n, p in model.named_parameters(
        ) if p.requires_grad and "lora" not in n]
        assert len(names) == 0
    num_params = sum([p.numel()
                      for p in model.parameters() if p.requires_grad])
    print(f"Total number of parameters that require gradients: {num_params}")
    return num_params


def prepare_optimizer_state(model, optimizer_state, device):
    print(f'device about prepare_optimizer_state:{device}')
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    avg = torch.cat([optimizer_state[n]["exp_avg"].view(-1) for n in range(len(names))])
    avg_sq = torch.cat([optimizer_state[n]["exp_avg_sq"].view(-1)
                        for n in range(len(names))])
    avg = avg.to(device)
    avg_sq = avg_sq.to(device)
    return avg, avg_sq


# Create ekfac_factors and pseudo_grads
def create_efkac_factors_and_pseudo_grads(model, device):
    kfac_input_covs = []
    kfac_grad_covs = []
    grads = []
    mlp_blocks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.requires_grad == True and (
                name.endswith("wqkv") or name.endswith("wo")) != True:  # and name != 'output':
            kfac_input_covs.append(torch.zeros(module.in_features, module.in_features).to(device))
            kfac_grad_covs.append(torch.zeros(module.out_features, module.out_features).to(device))
            grads.append([])
            mlp_blocks.append([name, module])
    return kfac_input_covs, kfac_grad_covs, grads, mlp_blocks


# Create forward hook to record forward input
def forward_hook_fn(layer_info, name):
    def hook(module, input, output):
        layer_info[name]["input"] = input[0].detach().clone()

    return hook


# Create backward hook to record output gradient
def backward_hook_fn(layer_info, name):
    def hook(module, grad_input, grad_output):
        if name in layer_info:
            layer_info[name]["grad_output"] = grad_output[0].detach().clone()

    return hook


# Create ekfac_factors
def get_ekfac_factors(model, dataloader, device):
    tot = len(dataloader)

    layer_info = {}
    hook_handle = []
    kfac_input_covs, kfac_grad_covs, grads, mlp_blocks = create_efkac_factors_and_pseudo_grads(model, device)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.requires_grad == True and (
                name.endswith("wqkv") or name.endswith("wo")) != True:  # and name != 'output'
            layer_info[name] = {}
            hook_forward = module.register_forward_hook(forward_hook_fn(layer_info, name))
            hook_backward = module.register_full_backward_hook(backward_hook_fn(layer_info, name))
            hook_handle.append(hook_forward)
            hook_handle.append(hook_backward)
    print(f"length of kic:{len(kfac_input_covs)}")
    print(f"length of kgc:{len(kfac_grad_covs)}")
    print(f"length of grads:{len(grads)}")
    print(f"length of mlp_blocks:{len(mlp_blocks)}")
    get_GPUinfo(device)
    for batch in tqdm(dataloader, total=len(dataloader)):
        model.zero_grad()
        prepare_batch(batch)
        loss = model(**batch).loss

        # Compute input covariance matrix A_{l-1}
        for i, [name, module] in enumerate(mlp_blocks):
            input_batch = layer_info[name]["input"]

            input_cov = torch.einsum("...i,...j->ij", input_batch, input_batch)
            kfac_input_covs[i] += input_cov / tot
            del layer_info[name]["input"]
            del input_cov
            del input_batch
            torch.cuda.empty_cache()
        get_GPUinfo(device)
        print(f"kfac_input_covs[0] shape:{kfac_input_covs[0].shape}")
        loss.backward()

        # Compute output gradient covariance matrix S_l and weight gradient
        for i, [name, module] in enumerate(mlp_blocks):
            grad_output_batch = layer_info[name]["grad_output"]

            grad_cov = torch.einsum("...i,...j->ij", grad_output_batch, grad_output_batch)
            kfac_grad_covs[i] += grad_cov / tot
            del layer_info[name]["grad_output"]
            del grad_cov
            del grad_output_batch
            torch.cuda.empty_cache()

        get_GPUinfo(device)

        print(f"kfac_grads_covs[0] shape:{kfac_grad_covs[0].shape}")

    del layer_info

    torch.cuda.empty_cache()

    for hooks in hook_handle:
        hooks.remove()
    get_GPUinfo(device)
    return kfac_input_covs, kfac_grad_covs, mlp_blocks


# Get Q_list for each layer
def get_Q_list(kfac_input_covs, kfac_grad_covs, mlp_blocks):
    print("Entering Getting Q Lists")
    sys.stdout.flush()
    q_a_list = []
    q_s_list = []
    q_a_t_list = []
    q_s_t_list = []
    for i in range(len(mlp_blocks)):
        print(f"Getting Layer {i}'s Q_list")
        sys.stdout.flush()
        q_a, _, q_a_t = torch.svd(kfac_input_covs[i])
        q_s, _, q_s_t = torch.svd(kfac_grad_covs[i])
        q_a_list.append(q_a)
        q_s_list.append(q_s)
        q_a_t_list.append(q_a_t)
        q_s_t_list.append(q_s_t)

    print("Successfully Getting Q Lists!")
    sys.stdout.flush()
    return q_a_list, q_a_t_list, q_s_list, q_s_t_list


# Compute lambda_ii for each layer
def get_lambda_ii_list(model, device, dataloader, mlp_blocks, q_a_list, q_s_list):
    squared_projections_sum = [0.0] * len(mlp_blocks)
    print("Getting lambda ii for every layer!")
    get_GPUinfo(device)
    sys.stdout.flush()

    lambda_ii_avg_list = [0.0] * len(mlp_blocks)

    layer_info = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.requires_grad == True and (
                name.endswith("wqkv") or name.endswith("wo")) != True:  # and name != 'output':
            layer_info[name] = {}

    tot = 0
    for batch in tqdm(dataloader, total=len(dataloader)):
        batch_grads = [[] for _ in range(len(mlp_blocks))]
        model.zero_grad()
        prepare_batch(batch)

        loss = model(**batch).loss
        loss.backward()
        for name_p, p in model.named_parameters():
            if p.grad is not None:
                for name_m, module in model.named_modules():
                    if isinstance(module, nn.Linear) and module.weight is p and (
                            name_m.endswith("wqkv") or name_m.endswith("wo")) != True:  # and name_m != 'output':
                        layer_info[name_m]["weight_grad"] = p.grad.detach().clone()

        for i, [name, module] in enumerate(mlp_blocks):
            batch_grads[i].append(layer_info[name]["weight_grad"])

        tot += len(batch_grads[0])
        squared_projections_sum = accumulate_squared_projections_sum(batch_grads, squared_projections_sum, q_a_list,
                                                                     q_s_list)

    print(f"num of dataset:{tot}")
    lambda_ii_avg_list = [projections_sum_layer / tot for projections_sum_layer in squared_projections_sum]
    print("Successfully Get lambda_ii_avg_list!")
    sys.stdout.flush()
    return lambda_ii_avg_list


# Accumulate squared_projections_sum
def accumulate_squared_projections_sum(batch_grads, squared_projections_sum, q_a_list, q_s_list):
    for layer_num in range(len(batch_grads)):
        n_examples = len(batch_grads[0])
        for j in range(n_examples):
            dtheta = batch_grads[layer_num][j]
            dtheta = dtheta.to(torch.bfloat16)
            q_a = q_a_list[layer_num].to(torch.bfloat16)
            q_s = q_s_list[layer_num].to(torch.bfloat16)
            result = (q_s @ dtheta @ q_a.T).view(-1)
            squared_projections_sum[layer_num] += result ** 2

    return squared_projections_sum


def get_ihvp(dataloader, model, output_dir, proj_dim):
    print("Starting getting ihvp!")
    sys.stdout.flush()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    get_GPUinfo(device)

    # Get kfac_input_covs, kfac_grad_covs, mlp_blocks
    print("Starting Loading ekfac factors!")
    sys.stdout.flush()

    kfac_input_covs, kfac_grad_covs, mlp_blocks = get_ekfac_factors(model, dataloader, device)
    torch.save(kfac_input_covs, 'check/opt_op/kfac_input_covs.pt')
    torch.save(kfac_grad_covs, 'check/opt_op/kfac_grad_covs.pt')
    ''' 

    kfac_input_covs = torch.load("check/opt/kfac_input_covs.pt")
    kfac_grad_covs = torch.load("check/opt/kfac_grad_covs.pt")
    _, _, _, mlp_blocks = create_efkac_factors_and_pseudo_grads(model, device)
    get_GPUinfo(device)
    '''

    print(f"kic takes {get_list_Memory(kfac_input_covs)} GBs")
    print(f"kgc takes {get_list_Memory(kfac_grad_covs)} GBs")
    print("Finish Get EFPG!")
    sys.stdout.flush()

    # Get Q_list (length of MLP number)
    print("Loading Q List!")
    sys.stdout.flush()
    q_a_list, q_a_t_list, q_s_list, q_s_t_list = get_Q_list(kfac_input_covs, kfac_grad_covs, mlp_blocks)
    torch.save(q_a_list, "check/opt_op/q_a_list.pt")
    torch.save(q_s_list, "check/opt_op/q_s_list.pt")
    torch.save(q_s_t_list, "check/opt_op/q_s_t_list.pt")
    torch.save(q_a_t_list, "check/opt_op/q_a_t_list.pt")

    '''
    q_a_list = torch.load("check/opt/q_a_list.pt")
    q_s_list = torch.load("check/opt/q_s_list.pt")
    q_a_t_list = torch.load("check/opt/q_a_t_list.pt")
    q_s_t_list = torch.load("check/opt/q_s_t_list.pt")
    '''

    del kfac_input_covs, kfac_grad_covs
    torch.cuda.empty_cache()

    get_GPUinfo(device)
    print(
        f"q_lists takes {get_list_Memory(q_a_list) + get_list_Memory(q_a_t_list) + get_list_Memory(q_s_list) + get_list_Memory(q_s_t_list)} GBs")
    print("Finish Getting Q list")
    sys.stdout.flush()
    # Get lambda_ii list (length of MLP number)

    print("Start Loading Lambda!")
    sys.stdout.flush()
    lambda_ii_avg_list = get_lambda_ii_list(model, device, dataloader, mlp_blocks, q_a_list, q_s_list)

    torch.save(lambda_ii_avg_list, "check/opt_op/lambda_list.pt")
    '''

    lambda_ii_avg_list = torch.load("check/opt/lambda_list.pt")
    '''

    print("Getting and project ihvp!")
    sys.stdout.flush()

    # Start computing ihvp for each sample
    layer_info = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.requires_grad == True and (
                name.endswith("wqkv") or name.endswith("wo")) != True:  # and name != 'output':
            layer_info[name] = {}

    output_dirs = {}
    for dim in proj_dim:
        output_dir_per_dim = os.path.join(output_dir, f"dim{dim}")
        output_dirs[dim] = output_dir_per_dim
        os.makedirs(output_dir_per_dim, exist_ok=True)

    count = 0
    project_batch_size = 4
    ihvp_to_project_list = []
    for batch in tqdm(dataloader, total=len(dataloader)):
        print(f"number of batch:{count}")
        batch_grads = [[] for _ in range(len(mlp_blocks))]
        model.zero_grad()
        count += 1
        prepare_batch(batch)
        loss = model(**batch).loss
        loss.backward()
        for name_p, p in model.named_parameters():
            if p.grad is not None:
                for name_m, module in model.named_modules():
                    if isinstance(module, nn.Linear) and module.weight is p and (
                            name_m.endswith("wqkv") or name_m.endswith("wo")) != True:  # and name_m != 'output':
                        layer_info[name_m]["weight_grad"] = p.grad.detach().clone()

        for i, [name, module] in enumerate(mlp_blocks):
            batch_grads[i].append(layer_info[name]["weight_grad"])

        batch_ihvp = get_batch_ihvp(q_a_list, q_s_list, q_a_t_list, q_s_t_list, lambda_ii_avg_list, batch_grads,
                                    damping=0.001)
        batch_ihvp = batch_ihvp.view(-1)
        ihvp_to_project_list.append(batch_ihvp)

        if count % project_batch_size == 0:
            project_ihvp(count, ihvp_to_project_list, model, output_dir, [8192], None, "sgd", None)
            del ihvp_to_project_list
            torch.cuda.empty_cache()
            ihvp_to_project_list = []

        sys.stdout.flush()

    if len(ihvp_to_project_list) > 0:
        count += 1
        if count % project_batch_size == 0:
            project_ihvp(count, ihvp_to_project_list, model, output_dir, [8192], None, "sgd", None)
            del ihvp_to_project_list
            torch.cuda.empty_cache()
    for dim in proj_dim:
        output_dir = output_dirs[dim]
        print(f"output_dir:{output_dir}")
        merge_and_normalize_info(output_dir, prefix="grads")
        merge_info(output_dir, prefix="grads")

# Project an ihvp batch
def project_ihvp(count_batch, batch_ihvp, model, output_dir, proj_dim: List[int] = [8192],
                 adam_optimizer_state: Optional[dict] = None,
                 gradient_type: str = "sgd", max_samples: Optional[int] = None):
    print(get_GPUinfo(model.device))
    sys.stdout.flush()
    model_id = 0  # model_id is used to draft the random seed for the projectors
    block_size = 128  # fixed block size for the projectors
    projector_batch_size = 8  # batch size for the projectors
    torch.random.manual_seed(0)  # set the random seed for torch

    project_interval = len(batch_ihvp)  # project every 16 batches
    save_interval = len(batch_ihvp)  # save every len(batch_ihvp) batches

    def _project(current_full_grads, projected_grads):
        current_full_grads = torch.stack(current_full_grads).to(torch.float16)
        for i, projector in enumerate(projectors):
            current_projected_grads = projector.project(
                current_full_grads, model_id=model_id)
            projected_grads[proj_dim[i]].append(current_projected_grads.cpu())

    def _save(projected_grads, output_dirs):
        for dim in proj_dim:
            if len(projected_grads[dim]) == 0:
                continue
            projected_grads[dim] = torch.cat(projected_grads[dim])

            output_dir = output_dirs[dim]
            outfile = os.path.join(output_dir, f"grads-{count_batch}.pt")
            torch.save(projected_grads[dim], outfile)
            print(
                f"Saving {outfile}, {projected_grads[dim].shape}", flush=True)
            projected_grads[dim] = []

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    # prepare optimization states
    if gradient_type == "adam":
        assert adam_optimizer_state is not None
        # first and second moment estimates
        m, v = prepare_optimizer_state(model, adam_optimizer_state, device)
    projector = get_trak_projector(device)

    number_of_params = batch_ihvp[0].numel()
    # never made it work sadly
    # fmodel, params, buffers = make_functional_with_buffers(model)
    # grads_loss = torch.func.grad(get_output, has_aux=False, argnums=1)

    # initialize a project for each target projector dimension
    projectors = []

    # number_of_params.to('cuda')
    # dim.to('cuda')
    print(f"num of params: {number_of_params}")
    device = torch.device(device)

    print(f'proj_dim : {proj_dim}')

    for dim in proj_dim:
        print(f"dim : {dim}")
        proj = projector(grad_dim=number_of_params,
                         proj_dim=dim,
                         seed=0,
                         proj_type=ProjectionType.rademacher,
                         device=device,
                         dtype=dtype,
                         block_size=block_size,
                         max_batch_size=projector_batch_size)
        print(type(proj))
        projectors.append(proj)
    print(f"projectors : {projectors}")
    count = 0
    # set up a output directory for each dimension
    output_dirs = {}
    for dim in proj_dim:
        output_dir_per_dim = os.path.join(output_dir, f"dim{dim}")
        output_dirs[dim] = output_dir_per_dim
        os.makedirs(output_dir_per_dim, exist_ok=True)
    # max index for each dimension
    max_index = min(get_max_saved_index(
        output_dirs[dim], "grads") for dim in proj_dim)
    print(f"max_index:{max_index}")
    # projected_gradients

    projected_grads = {dim: [] for dim in proj_dim}  # projected gradients
    print(f"projected_grads : {projected_grads}")

    if len(batch_ihvp) > 0:
        _project(batch_ihvp, projected_grads)
        del batch_ihvp
    print(projected_grads)
    for dim in proj_dim:
        _save(projected_grads, output_dirs)

    torch.cuda.empty_cache()


# get a batch of whole ihvp
def get_batch_ihvp(q_a_list, q_s_list, q_a_t_list, q_s_t_list, lambda_ii_avg_list, batch_grads, damping=0.001):
    print(f"batch's volumn:{len(batch_grads[0])}, element[0][0] shape:{batch_grads[0][0].shape}")
    sys.stdout.flush()
    """Compute EK-FAC inverse Hessian-vector products."""
    # batch_grads is a list with a length equal to the number of Linear Layers,
    # where each element is a list with a length equal to the number of samples in
    # the batch, and each element is the weight gradient of the i-th layer.
    ihvp = []
    time_in = time.time()
    for i in range(len(batch_grads)):
        V = torch.stack(batch_grads[i])
        # Performing eigendecompositions on the input and gradient covariance matrices
        q_a, q_a_t, q_s, q_s_t = q_a_list[i], q_a_t_list[i], q_s_list[i], q_s_t_list[i]

        # Calculate the EK-FAC diagonal damping inverse matrix.”
        lambda_ii = lambda_ii_avg_list[i]
        ekfacDiag_damped_inv = 1.0 / (lambda_ii + damping)
        ekfacDiag_damped_inv = ekfacDiag_damped_inv.reshape((V.shape[-2], V.shape[-1]))

        # calculate middle result
        q_a_t = q_a_t.to(torch.bfloat16)
        q_s = q_s.to(torch.bfloat16)
        q_a = q_a.to(torch.bfloat16)
        q_s_t = q_s_t.to(torch.bfloat16)
        intermediate_result = torch.einsum("bij,jk->bik", V, q_a_t)
        intermediate_result = intermediate_result.to(torch.bfloat16)
        intermediate_result = torch.einsum("ji,bik->bjk", q_s, intermediate_result)
        result = intermediate_result / ekfacDiag_damped_inv.unsqueeze(0)
        result = result.to(torch.bfloat16)
        # calculate the ihvp component
        ihvp_component = torch.einsum("bij,jk->bik", result, q_a)
        ihvp_component = ihvp_component.to(torch.bfloat16)
        ihvp_component = torch.einsum("ji,bik->bjk", q_s_t, ihvp_component)
        # flattening the result except for the batch dimension
        ihvp_component = einops.rearrange(ihvp_component, "b j k -> b (j k)")
        ihvp.append(ihvp_component)

    time_out = time.time()
    print(f"total time: {time_out - time_in}")
    print(f"ihvp[0] shape before cat:{ihvp[0].shape}")
    # Concatenating the results across blocks to get the final ihvp
    return torch.cat(ihvp, dim=-1)


# This function represents saving the info with normalization.
def merge_and_normalize_info(output_dir: str, prefix="reps"):
    """ Merge and normalize the representations and gradients into a single file. """
    info = os.listdir(output_dir)
    info = [file for file in info if file.startswith(prefix)]
    # Sort the files in ascending order
    info.sort(key=lambda x: int(x.split(".")[0].split("-")[1]))
    merged_data = []
    for file in info:
        data = torch.load(os.path.join(output_dir, file))
        normalized_data = normalize(data, dim=1)
        merged_data.append(normalized_data)
    merged_data = torch.cat(merged_data, dim=0)

    output_file = os.path.join(output_dir, f"all_orig.pt")
    torch.save(merged_data, output_file)
    print(
        f"Saving the normalized {prefix} (Shape: {merged_data.shape}) to {output_file}.")


# This function represents saving the info but without normalization.
def merge_info(output_dir: str, prefix="reps"):
    """ Merge the representations and gradients into a single file without normalization. """
    info = os.listdir(output_dir)
    info = [file for file in info if file.startswith(prefix)]
    # Sort the files in ascending order
    info.sort(key=lambda x: int(x.split(".")[0].split("-")[1]))
    merged_data = []
    for file in info:
        data = torch.load(os.path.join(output_dir, file))
        merged_data.append(data)
    merged_data = torch.cat(merged_data, dim=0)

    output_file = os.path.join(output_dir, f"all_unormalized.pt")
    torch.save(merged_data, output_file)
    print(
        f"Saving the unnormalized {prefix} (Shape: {merged_data.shape}) to {output_file}.")


# get device's information
def get_GPUinfo(device):
    current_gpu_index = device
    props = torch.cuda.get_device_properties(current_gpu_index)

    # get total memory (unit: GB)
    total_memory = props.total_memory / (1024 ** 3)

    # get used memory (unit: GB)
    used_memory = torch.cuda.memory_allocated(current_gpu_index) / (1024 ** 3)

    # get free memory (unit: GB)
    free_memory = total_memory - used_memory

    print(f"GPU {current_gpu_index} total memory: {total_memory:.2f} GB")
    print(f"GPU {current_gpu_index} used memory: {used_memory:.2f} GB")
    print(f"GPU {current_gpu_index} free memory: {free_memory:.2f} GB")


def get_list_Memory(A):
    memory_size = sum(matrix.element_size() * matrix.nelement() for matrix in A)

    memory_size_gb = memory_size / (1024 ** 3)
    return memory_size_gb


def get_grads_Memory(grads):
    memory_size = 0
    for A in grads:
        memory_size += sum(matrix.element_size() * matrix.nelement() for matrix in A)

    memory_size_gb = memory_size / (1024 ** 3)
    return memory_size_gb

