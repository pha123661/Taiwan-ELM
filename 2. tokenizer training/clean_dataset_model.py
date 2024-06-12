import json
import os
from datetime import datetime

import jionlp as jio
import opencc
from datasets import concatenate_datasets, load_dataset, load_from_disk

converter = opencc.OpenCC('s2t.json')

if not os.path.exists("corpus_model"):
    dataset_names = [
        "liswei/news-collection-zhtw",
        "liswei/wikipedia-zhtw-dedup",
        "liswei/wikinews-zhtw-dedup"
    ]
    raw_datasets = [load_dataset(name, split='train')
                    for name in dataset_names]

    for idx in range(len(raw_datasets)):
        # remove all columns except "text", and make the removed columns as column "meta" as a json string
        def gen_meta(example, text_col='text'):
            example['meta'] = {k: example[k] for k in example if k != text_col}
            for k in example['meta']:
                del example[k]
            example['meta'] = json.dumps(
                example['meta'], ensure_ascii=False, sort_keys=True, default=lambda x: x.isoformat() if isinstance(x, datetime) else str(x))
            example['id'] = f"{dataset_names[idx]}-{hash(example[text_col])}"
            return example

        dataset = raw_datasets[idx]
        dataset = dataset.map(gen_meta, batched=False, num_proc=os.cpu_count())
        raw_datasets[idx] = dataset

    raw_datasets = concatenate_datasets(raw_datasets)
    raw_datasets.save_to_disk('corpus_model')
else:
    raw_datasets = load_from_disk('corpus_model')


def clean_text(example):
    text = example['text']
    text = jio.clean_text(text)
    text = converter.convert(text)

    example['text'] = text
    return example


raw_datasets = raw_datasets.map(
    clean_text, batched=False, num_proc=os.cpu_count())
raw_datasets.save_to_disk('corpus_cleaned_model')
raw_datasets.push_to_hub("zhtw-news-and-articles-2T")
