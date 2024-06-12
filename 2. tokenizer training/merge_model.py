import argparse
import os
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import DatasetDict, concatenate_datasets, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments, set_seed)

if __name__ == "__main__":
    # Define arguments
    set_seed(11944004)
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged_tok", type=str,
                        default="Llama-2-7b-zhtw-tokenizer")
    parser.add_argument("--model", type=str, default="apple/OpenELM-270M")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--push_to_hub", action="store_true")
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, device_map='cpu')

    tokenizer = AutoTokenizer.from_pretrained(args.merged_tok)

    embeddings = model.get_input_embeddings()
    old_vocab_size = embeddings.weight.size(0)
    new_embeddings = nn.Embedding(
        len(tokenizer), embeddings.weight.size(1),
        dtype=embeddings.weight.dtype, device=embeddings.weight.device)
    new_embeddings.weight.data[:old_vocab_size] = embeddings.weight.data
    new_embeddings.weight.data[old_vocab_size:] = embeddings.weight.mean(
        dim=0, keepdim=True)
    model.set_input_embeddings(new_embeddings)
    if hasattr(model.config, "vocab_size"):
        model.config.vocab_size = len(tokenizer)

    print(
        f"Old vocab size: {old_vocab_size}, New vocab size: {len(tokenizer)}")

    lm_head = model.get_output_embeddings()
    if lm_head is not None:
        new_lm_head = nn.Linear(
            lm_head.in_features, len(tokenizer), bias=False,
            dtype=embeddings.weight.dtype, device=embeddings.weight.device)
        new_lm_head.weight.data[:old_vocab_size] = lm_head.weight.data
        new_lm_head.weight.data[old_vocab_size:] = lm_head.weight.mean(
            dim=0, keepdim=True)
        model.set_output_embeddings(new_lm_head)

        print(
            f"Old LM Head size: {lm_head.out_features}, New LM Head size: {len(tokenizer)}")

    else:
        print("No LM Head found in the model, possible model has shared weights with the input embeddings.")

    if args.push_to_hub:
        model.push_to_hub(args.output_dir, private=True)
        tokenizer.push_to_hub(args.output_dir, private=True)
    else:
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
