import os

from transformers import AutoTokenizer


def tokenize_and_sort_tokens(model_file):
    tokenizer = AutoTokenizer.from_pretrained(model_file)

    all_vocabs = tokenizer.get_vocab()
    sorted_vocabs = sorted(all_vocabs.items(), key=lambda x: x[1])

    for token, token_id in sorted_vocabs:
        token_id_tokenize = tokenizer.encode(token, add_special_tokens=False)
        print(f"Token: {token} -> ID: {token_id}, {token_id_tokenize}")


model_file_path = '/gs/bs/tga-bayes-crest/fujii/hf-checkpoints/Meta-Llama-3-8B'

tokenize_and_sort_tokens(model_file_path)
