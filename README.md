<center>
    <img src="https://huggingface.co/liswei/Taiwan-ELM/resolve/main/Taiwan%20ELM%20Logo.jpeg" alt="Efficient LLM for Taiwan">
</center>

> Efficient LLM for Taiwan with open weights/datasets/checkpoints and affordable sizes (270M/1.1B)

🤗 <a href="https://huggingface.co/collections/liswei/taiwan-elm-665c238424cc1676f9c8c3ee" target="_blank">Model Collection</a> • 📖 <a href="https://huggingface.co/collections/liswei/traditional-chinese-llm-corpus-6636f2dee48c957c340f23e6" target="_blank">Traditional Chinese Corpus</a> 

# Taiwan ELM

Taiwan ELM is a family of Efficient LLMs for Taiwan base on [apple/OpenELM](https://huggingface.co/apple/OpenELM).
The project aims to provide an efficient model for researchers without access to large-scale computing resources.

The model is trained using [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) on 2B Traditional Chinese tokens and 500K instruction samples.
We will extend the model to train on larger data sets and different base models if there is sufficient demand.

## What is being released?

We release both pre-trained **base models and instruction tuned variants** with 270M and 1.1B parameters.
Along with the model, **datasets used to train the models** are also released.

In an effort to improve transparency, training **checkpoints (including rng/optimizer state) and logs** are also released in the model page.

List of released models:
* [Taiwan-ELM-270M](https://huggingface.co/liswei/Taiwan-ELM-270M)
* [Taiwan-ELM-1_1B](https://huggingface.co/liswei/Taiwan-ELM-1_1B)
* [Taiwan-ELM-270M-Instruct](https://huggingface.co/liswei/Taiwan-ELM-270M-Instruct)
* [Taiwan-ELM-1_1B-Instruct](https://huggingface.co/liswei/Taiwan-ELM-1_1B-Instruct)

List of released datasets:
* [liswei/Taiwan-Text-Excellence-2B](https://huggingface.co/datasets/liswei/Taiwan-Text-Excellence-2B)
* [liswei/PromptPair-TW](https://huggingface.co/datasets/liswei/PromptPair-TW)
* [liswei/wikinews-zhtw-dedup](https://huggingface.co/datasets/liswei/wikinews-zhtw-dedup)
* [liswei/wikipedia-zhtw-dedup](https://huggingface.co/datasets/liswei/wikipedia-zhtw-dedup)
* [liswei/coct-en-zhtw-dedup](https://huggingface.co/datasets/liswei/coct-en-zhtw-dedup)

Some of the datasets are not used for training Taiwan ELM but also released:
* [liswei/common-crawl-zhtw](https://huggingface.co/datasets/liswei/common-crawl-zhtw)
* [liswei/c4-zhtw](https://huggingface.co/datasets/liswei/c4-zhtw)
* [liswei/rm-static-zhTW](https://huggingface.co/datasets/liswei/rm-static-zhTW)

Codebase:
* [Dataset Cleaning](./1.%20dataset%20cleaning)
* [Tokenizer Training](./2.%20tokenizer%20training)
* Training: [LLaMA-Factory - 5177f3b](https://github.com/hiyouga/LLaMA-Factory/tree/5177f3ba90f369ec55bb270d5fb868f1b94e3acf)

## Usage Examples

For instruction-tuned modesl, we adapt the [LLaMA2](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) template:
```jinja2
<s>[INST] <<SYS>>
{{ system_prompt }}
<</SYS>>

{{ user_message }} [/INST]
```

The model could be load via `AutoModelForCausalLM` or `text-generation-inference` with `trust_remote_code=True`:
```python
taiwan_elm_270m = AutoModelForCausalLM.from_pretrained("liswei/Taiwan-ELM-270M", trust_remote_code=True)
```

We also support additional generation methods and speculative generation, please find reference at [OpenELM#usage](https://huggingface.co/apple/OpenELM#usage).
