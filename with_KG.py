import json
from http.client import responses

import requests
from datasets import load_dataset
from rouge_score import rouge_scorer
from bert_score import BERTScorer

import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT


# Initialize models
model = SentenceTransformer('all-MiniLM-L6-v2')
keybert_model = KeyBERT(model='all-MiniLM-L6-v2')


# Helper function to extract key phrases
def extract_key_phrases(content, top_n=50):
    """Extract top-n key phrases using KeyBERT."""
    keywords = keybert_model.extract_keywords(content, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=top_n)
    return [kw[0] for kw in keywords]


def generate_knowledge_graph(data):
    """Generate and return the knowledge graph."""
    G = nx.Graph()
    key_phrases = []
    phrase_to_source = {}

    phrases = extract_key_phrases(data, top_n=10)
    key_phrases.extend(phrases)
    # phrase_to_source.update({phrase: row["Source"] for phrase in phrases})

    # for index, row in data.iterrows():
    #     phrases = extract_key_phrases(row["Content"], top_n=10)
    #     key_phrases.extend(phrases)
    #     phrase_to_source.update({phrase: row["Source"] for phrase in phrases})

    embeddings = model.encode(key_phrases)
    similarity_matrix = cosine_similarity(embeddings)

    # Thershold factor. Increase to increase noise. Decrease for vice-versa.
    threshold = 0.5

    for i, phrase_i in enumerate(key_phrases):
        G.add_node(phrase_i, label=phrase_i, color="#008080")
        similarity_scores = [
            (key_phrases[j], similarity_matrix[i][j])
            for j in range(len(key_phrases))
            if i != j and phrase_to_source[phrase_i] != phrase_to_source[key_phrases[j]]
        ]
        sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[:3]  # Top 3 connections
        for phrase_j, score in sorted_scores:
            if score > threshold:
                G.add_edge(phrase_i, phrase_j)

    return G


def generate_summaries_mistral(prompt):
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

# Extract the source content and summaries directly
chapters = ds_train[:3]['chapter']  # The original chapter text
summaries = ds_train[:3]['summary_text']  # The summary text


responses = []
for chapter in chapters:



    prompt = (
        "You are a summary generator. I would like you to generate a summary out of the following content: "
        f"{chapter}")
    responses.append(generate_summaries_mistral(prompt))


# # Get the raw data (which are JSON strings)
# raw_data = ds_train[:10]['summary']
# print(raw_data)
#
# # Parse each JSON string and extract the 'summary' field
# references = []
# for item in raw_data:
#     parsed = json.loads(item)  # Parse the JSON string
#     references.append(parsed['summary'])  # Extract the summary

# print(f"Number of summaries: {len(references)}")
# print(f"\nFirst summary (first 300 chars):\n{references[0][:300]}...")






# prompt = ("You are a summary generator. I would like you to take the following long text and generate a concise summary "
#           "out of it."
#           "The ")
#
predictions = responses

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
bert_scorer = BERTScorer(model_type='bert-base-uncased')

for i, (pred, ref) in enumerate(zip(predictions, summaries)):
    scores = scorer.score(ref, pred)
    P, R, F1 = bert_scorer.score([ref], [pred])
    print(f"\nSummary {i+1}:")
    print(f"  ROUGE-1: {scores['rouge1'].fmeasure:.4f}")
    print(f"  ROUGE-2: {scores['rouge2'].fmeasure:.4f}")
    print(f"  ROUGE-L: {scores['rougeL'].fmeasure:.4f}")
    print(f"  BERTScore Precision: {P.mean():.4f}, Recall: {R.mean():.4f}, F1: {F1.mean():.4f}")
