from typing import  Sequence
from collections.abc import Iterable
import torch


def encode_with_content_format(example, tokenizer, max_seq_length=1024):
    max_seq_length = 1024
    tokenizer_chunk = int(max_seq_length)
    min_length = 512
    overlap_length = 1
    content = example['content']
    start = 0
    ids = tokenizer(content + tokenizer.eos_token, return_tensors='pt', truncation=False).input_ids.flatten()
    ids = ids[0:min(len(ids),20480)]
    while start < len(ids):
        if len(ids) - start < min_length:
            break
        input_ids = ids[start : min(start + tokenizer_chunk, len(ids))]
        labels = input_ids.clone()
        attention_mask = torch.ones_like(input_ids)
        yield {'input_ids':input_ids, 'labels':labels, "ids":example["ids"], 'attention_mask': attention_mask}
        # next start
        if start + tokenizer_chunk >= len(ids):
            break
        else:
            start = start + tokenizer_chunk - overlap_length


def flatten(xs: Sequence) -> list:
    """Flatten a nested list."""
    def _flatten(ys):
        for y in ys:
            if isinstance(y, Iterable) and not isinstance(y, (str, bytes)):
                yield from _flatten(y)
            else:
                yield y
    return list(_flatten(xs))
