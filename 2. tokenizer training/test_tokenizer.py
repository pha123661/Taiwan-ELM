import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datasets import load_dataset
from matplotlib.ticker import MaxNLocator
from transformers import LlamaTokenizer

parser = argparse.ArgumentParser(description='Process tokenizer name.')

parser.add_argument('TokenizerName',
                    metavar='tokenizer',
                    type=str,
                    help='the name of the tokenizer to be tested')

args = parser.parse_args()

raw_dataset = load_dataset(
    "bigscience-data/roots_zh-tw_wikipedia", split="train")

print(raw_dataset)


def calculate_tokenization_length(tokenizer, dataset):
    def tokenize_function(examples):
        return {"length": [len(ids) for ids in tokenizer(examples['text']).input_ids]}

    return dataset.map(tokenize_function, batched=True, num_proc=16)['length']


vanilla_tokenizer = LlamaTokenizer.from_pretrained(
    "meta-llama/Llama-2-7b-hf", legacy=True)
merged_tokenizer = LlamaTokenizer.from_pretrained(args.TokenizerName)

vanilla_lengths = calculate_tokenization_length(
    vanilla_tokenizer, raw_dataset)
merged_lengths = calculate_tokenization_length(
    merged_tokenizer, raw_dataset)


df_old = pd.DataFrame(vanilla_lengths,
                      columns=['Tokenization Length'])
df_old['Tokenizer'] = 'Vanilla Tokenizer'

df_merged = pd.DataFrame(merged_lengths,
                         columns=['Tokenization Length'])
df_merged['Tokenizer'] = 'Merged Tokenizer'

df = pd.concat([df_old, df_merged])


plt.figure(figsize=(10, 6))
plot = sns.histplot(data=df, x='Tokenization Length',
                    hue='Tokenizer', multiple="layer", bins='auto', log_scale=True)
plt.xlabel('Tokenization Length')
plt.ylabel('Count')

# Help me plot two red lines at their mean
# then, calculate the saving percentage
# and plot a text box connecting the two red lines with the saving percentage
mean_old = df_old['Tokenization Length'].mean()
mean_merged = df_merged['Tokenization Length'].mean()

plt.title(
    f'Tokenization Length Comparison, from {mean_old:.2f} to {mean_merged:.2f}')

plt.savefig(os.path.join(args.TokenizerName,
            'tokenization_length_comparison.png'))
