import os
import re
from itertools import chain

import sentencepiece as spm
from datasets import load_from_disk

if not os.path.exists("corpus_txt"):
    # Preparing dataset
    raw_datasets = load_from_disk('corpus_cleaned')['text']
    # save the dataset as txt file, move to new file every 50000 lines
    os.makedirs("corpus_txt", exist_ok=True)

    index = 0
    f = open(f"corpus_txt/corpus_{index}.txt", "w", encoding="utf-8")
    for i, text in enumerate(raw_datasets):
        if i > 0 and i % 50000 == 0:
            f.close()
            index += 1
            f = open(f"corpus_txt/corpus_{index}.txt", "w", encoding="utf-8")

        f.write(text)
        f.write("\n")
    f.close()
    del raw_datasets

output_dir = 'zhtw-bpe-tok'

spm.SentencePieceTrainer.train(
    input=[os.path.join("corpus_txt", f) for f in os.listdir(
        "corpus_txt") if re.match(r'corpus_\d+.txt', f)],
    model_prefix=output_dir,
    vocab_size=64_000,
    character_coverage=0.9995,
    accept_language="zh",
    model_type="bpe",
    split_digits=True,
    byte_fallback=True,
    max_sentence_length=128_000,
    input_sentence_size=30_000,
)

sp_bpe = spm.SentencePieceProcessor()
sp_bpe.load(f'{output_dir}.model')
print(sp_bpe.encode_as_pieces(
    'The excellence of a translation can only be judged by noting'))
print(len(sp_bpe.encode_as_pieces(
    'The excellence of a translation can only be judged by noting')))
print(sp_bpe.encode_as_pieces('你好嗎中國，我有老干媽'))
print(len(sp_bpe.encode_as_pieces('你好嗎中國，我有老干媽')))

# https://github.com/YuchuanTian/RethinkTinyLM/blob/cf1bd2ad3e4b9fc10bbf8f80624691182647e273/src/step3_generate_new_tokenizer.py
