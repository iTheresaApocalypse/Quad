def get_content_dataset(raw_datasets, count):
    result = []
    for example in raw_datasets:
        count = count + 1
        result.append({"ids":count, "content":example["content"]})
    return result, count

def get_text_dataset(raw_datasets, count):
    result = []
    for example in raw_datasets:
        count = count + 1
        result.append({"ids":count, "content":example["text"]})
    return result, count


def get_question_dataset(raw_datasets, count):
    result = []
    for example in raw_datasets:
        count = count + 1
        if not example['question'].endswith((' ', '\n', '\t')) and not example['answer'].startswith((' ', '\n', '\t')):
            example_text = example['question'] + ' ' + example['answer']
        else:
            example_text = example['question'] + example['answer']
        result.append({"ids":count, "content":example_text})
    return result, count

def get_unify_dataset(raw_datasets, count):
    if "content" in raw_datasets.column_names:
        return get_content_dataset(raw_datasets, count)
    elif "text" in raw_datasets.column_names:
        return get_text_dataset(raw_datasets, count)
    elif "question" in raw_datasets.column_names and "answer" in raw_datasets.column_names:
        return get_question_dataset(raw_datasets, count)
    else:
        raise ValueError("Invalid format")