import json
import requests
from datasets import load_dataset
from rouge_score import rouge_scorer


def generate_summary_mistral(prompt):
    """Mistral seemed like the best choice for a local-logical reasoning model."""
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "mistral",
            "prompt": prompt,
            "stream": True,
            "num_predict": 400,  # Use this instead of max_tokens
            "temperature": 0.7
        },
        stream=True
    )

    output = ""
    try:
        for line in response.iter_lines():
            if line:
                line_data = json.loads(line.decode("utf-8"))
                if "response" in line_data:
                    output += line_data["response"]
    except Exception as e:
        output = f"Error: {e}"
    return output.strip()


# Load dataset
ds = load_dataset("kmfoda/booksum")
ds_train = load_dataset("kmfoda/booksum", split='train')

print(ds_train)

# Get the raw data (which are JSON strings)
raw_data = ds_train[:10]['summary']

# Parse each JSON string and extract the 'summary' field
references = []
for item in raw_data:
    parsed = json.loads(item)  # Parse the JSON string
    references.append(parsed['summary'])  # Extract the summary

# print(f"Number of summaries: {len(references)}")
# print(f"\nFirst summary (first 300 chars):\n{references[0][:300]}...")


prompt = ("You are a summary generator. I would like you to take the following long text and generate a concise summary "
          "out of it."
          "The ")
# predictions = generate_summary_mistral(prompt)
predictions = references

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

for i, (pred, ref) in enumerate(zip(predictions, references)):
    scores = scorer.score(ref, pred)
    print(f"\nSummary {i+1}:")
    print(f"  ROUGE-1: {scores['rouge1'].fmeasure:.4f}")
    print(f"  ROUGE-2: {scores['rouge2'].fmeasure:.4f}")
    print(f"  ROUGE-L: {scores['rougeL'].fmeasure:.4f}")