from Quad_collect_grad import collect_grads
from get_training_dataset import encode_data, get_dataloader
from unify_data_format import get_unify_dataset
from datasets import Dataset




def get_info(model, tokenizer, minibatch, device, max_seq_length=1024):
    # pad token is not added by default for pretrained models
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    # resize embeddings if needed (e.g. for LlamaTokenizer)
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    minibatch = Dataset.from_list(minibatch)
    print(minibatch)
    minibatch, _ = get_unify_dataset(minibatch, 0)
    minibatch = encode_data(minibatch, tokenizer, max_seq_length)
    chunk_doc_dictionary = {}
    chunk_id = 0
    new_minibatch = []
    for example in minibatch:
        chunk_id = chunk_id + 1
        chunk_doc_dictionary[chunk_id] = example['ids']
        del example["ids"]
        new_minibatch.append(example)
    new_minibatch = Dataset.from_list(new_minibatch)
    dataloader = get_dataloader(new_minibatch, tokenizer=tokenizer)

    gradient_projection_dimension = [8192]
    gradient_type = "sgd"
    max_samples = None
    return collect_grads(dataloader,
                model,
                device,
                proj_dim=gradient_projection_dimension,
                gradient_type=gradient_type,
                max_samples=max_samples), chunk_doc_dictionary
