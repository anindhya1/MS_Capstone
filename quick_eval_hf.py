import os, argparse
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
from tqdm import tqdm
from datasets import load_dataset
import evaluate

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def summarize_batch(texts, model_name="facebook/bart-large-cnn", max_inp=1024, max_out=200):
    """Very simple baseline summarizer (truncates long docs)."""
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    preds = []
    for t in tqdm(texts, desc="Summarizing"):
        # naive truncation for long docs
        inputs = tok(t, truncation=True, max_length=max_inp, return_tensors="pt")
        out = mdl.generate(**inputs, max_new_tokens=max_out)
        preds.append(tok.decode(out[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))
    return preds

def compute_scores(preds, refs, bert_model="bert-base-uncased"):
    rouge = evaluate.load("rouge")
    r = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
    bsc = evaluate.load("bertscore").compute(predictions=preds, references=refs,
                                             model_type=bert_model, lang="en")
    return {
        "rouge1": r["rouge1"], "rouge2": r["rouge2"], "rougeL": r["rougeL"], "rougeLsum": r["rougeLsum"],
        "bertscore_p": sum(bsc["precision"])/len(bsc["precision"]),
        "bertscore_r": sum(bsc["recall"])/len(bsc["recall"]),
        "bertscore_f1": sum(bsc["f1"])/len(bsc["f1"]),
        "n": len(preds),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["govreport", "booksum"], required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--subset", type=int, default=100, help="Evaluate first N examples (None for full).")
    ap.add_argument("--model", default="facebook/bart-large-cnn", help="HF model for baseline summaries.")
    ap.add_argument("--bert-model", default="bert-base-uncased")
    ap.add_argument("--max-inp", type=int, default=1024)
    ap.add_argument("--max-out", type=int, default=200)
    args = ap.parse_args()

    if args.dataset == "govreport":
        # Fields: report (doc), summary (reference)
        ds = load_dataset("ccdv/govreport-summarization", split=args.split)  # auto-downloads & caches
        docs = ds["report"]
        refs = ds["summary"]
    else:
        # BookSum (chapter-level): chapter (doc), summary_text (reference)
        ds = load_dataset("kmfoda/booksum", "chapter", split=args.split)  # request chapter config
        docs = ds["chapter"]
        refs = ds["summary_text"]

    if args.subset:
        docs, refs = docs[:args.subset], refs[:args.subset]

    preds = summarize_batch(docs, model_name=args.model, max_inp=args.max_inp, max_out=args.max_out)
    scores = compute_scores(preds, refs, bert_model=args.bert_model)

    print("\n=== ROUGE ===")
    print(f"R1: {scores['rouge1']:.4f}  R2: {scores['rouge2']:.4f}  RL: {scores['rougeL']:.4f}  RLsum: {scores['rougeLsum']:.4f}")
    print("=== BERTScore ===")
    print(f"P:  {scores['bertscore_p']:.4f}  R: {scores['bertscore_r']:.4f}  F1: {scores['bertscore_f1']:.4f}")
    print(f"N: {scores['n']}")
    print("Done.")
if __name__ == "__main__":
    main()
