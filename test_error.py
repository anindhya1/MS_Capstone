import os
os.environ["TOKENIZERS_PARALLELISM"]="false"
os.environ["OMP_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY","YES")

print("A) imports")
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

print("B) load tok/model")
tok = AutoTokenizer.from_pretrained("facebook/bart-large-cnn", use_fast=False)
mdl = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

print("C) tokenize")
x = tok("This is a short test input.", return_tensors="pt", truncation=True, max_length=64)

print("D) generate")
y = mdl.generate(**x, max_new_tokens=16)

print("E) decode")
print(tok.decode(y[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))
print("DONE")
