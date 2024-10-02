import os
from typing import Iterable, List, Optional

import torch
import torch.nn.functional as F
from peft import PeftModel
from torch import Tensor
from torch.nn.functional import normalize
from tqdm import tqdm
from trak import *
from trak.projectors import BasicProjector, CudaProjector, ProjectionType


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
        projector = CudaProjector
        print("Using CudaProjector")
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


def obtain_gradients(model, batch):
    loss = model(**batch).loss
    loss.backward()
    vectorized_grads = torch.cat(
        [p.grad.view(-1) for p in model.parameters() if p.grad is not None])
    return vectorized_grads


def obtain_sign_gradients(model, batch):
    loss = model(**batch).loss
    loss.backward()

    # Instead of concatenating the gradients, concatenate their signs
    vectorized_grad_signs = torch.cat(
        [torch.sign(p.grad).view(-1) for p in model.parameters() if p.grad is not None])

    return vectorized_grad_signs



def prepare_optimizer_state(model, optimizer_state, device):
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    avg = torch.cat([optimizer_state[n]["exp_avg"].view(-1) for n in range(len(names))])
    avg_sq = torch.cat([optimizer_state[n]["exp_avg_sq"].view(-1)
                       for n in range(len(names))])
    avg = avg.to(device)
    avg_sq = avg_sq.to(device)
    return avg, avg_sq


def collect_grads(dataloader,
                  model,
                  device,
                  output_dir = None,
                  proj_dim: List[int] = [8192],
                  adam_optimizer_state: Optional[dict] = None,
                  gradient_type: str = "sgd",
                  max_samples: Optional[int] = None):
    """
    Collects gradients from the model during evaluation and saves them to disk.

    Args:
        dataloader (torch.utils.data.DataLoader): The data loader for evaluation dataset.
        model (torch.nn.Module): The model from which gradients will be collected.
        output_dir (str): The directory where the gradients will be saved.
        proj_dim List[int]: The dimensions of the target projectors. Each dimension will be saved in a separate folder.
        gradient_type (str): The type of gradients to collect. [adam | sign | sgd]
        adam_optimizer_state (dict): The optimizer state of adam optimizers. If None, the gradients will be collected without considering Adam optimization states. 
        max_samples (int, optional): The maximum number of samples to collect. Defaults to None.
    """

    model_id = 0  # model_id is used to draft the random seed for the projectors
    block_size = 128  # fixed block size for the projectors
    projector_batch_size = 8  # batch size for the projectors
    torch.random.manual_seed(0)  # set the random seed for torch

    project_interval = 5  # project every 16 batches
    save_interval = 160  # save every 160 batches

    def _project(current_full_grads, projected_grads):
        current_full_grads = torch.stack(current_full_grads).to(torch.float16)
        for i, projector in enumerate(projectors):
            current_projected_grads = projector.project(
                current_full_grads, model_id=model_id)
            projected_grads[proj_dim[i]].append(current_projected_grads.cpu())

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    # prepare optimization states
    if gradient_type == "adam":
        assert adam_optimizer_state is not None
        # first and second moment estimates
        m, v = prepare_optimizer_state(model, adam_optimizer_state, device)
    projector = get_trak_projector(device)
    number_of_params = get_number_of_params(model)
    # never made it work sadly
    # initialize a project for each target projector dimension
    projectors = []
    device = torch.device(device)
    for dim in proj_dim:
        proj = projector(grad_dim=number_of_params,
                         proj_dim=dim,
                         seed=0,
                         proj_type=ProjectionType.rademacher,
                         device=device,
                         dtype=dtype,
                         block_size=block_size,
                         max_batch_size=projector_batch_size)
        # print(type(proj))
        projectors.append(proj)

    count = 0
    full_grads = []  # full gradients
    projected_grads = {dim: [] for dim in proj_dim}  # projected gradients

    for batch in tqdm(dataloader, total=len(dataloader)):
        prepare_batch(batch,device)
        count += 1

        if count == 1:
            print("Using SGD gradients")
        vectorized_grads = obtain_gradients(model, batch)

        # add the gradients to the full_grads
        full_grads.append(vectorized_grads)
        model.zero_grad()

        if count % project_interval == 0:
            _project(full_grads, projected_grads)
            full_grads = []

        if max_samples is not None and count == max_samples:
            break

    if len(full_grads) > 0:
        _project(full_grads, projected_grads)
        full_grads = []

    return projected_grads



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
    return merged_data


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

