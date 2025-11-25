# import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"   # macOS Accelerate
# # LAST RESORT: uncomment only if the rest doesn’t fix it
# # os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ["PYTHONFAULTHANDLER"] = "1"       # helpful if it still crashes

# Optional last resort (see below): os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 1) Heavy ML stack first
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT

# 2) Then the rest
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

import json
from http.client import responses

import requests
from datasets import load_dataset
from rouge_score import rouge_scorer
from bert_score import BERTScorer
import networkx as nx
from pyvis.network import Network # to visualize the graph in HTML
import sacrebleu


print("Done importing!")
# Initialize models
sentence_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
keybert_model = KeyBERT(model=sentence_embedding_model)
print("Done loading models!")

def identify_crux_nodes(G):
    """
    Identify crux nodes in the knowledge graph based on centrality measures.
    :param G:
    :return: crux_nodes, neighbour_nodes
    """
    # Identify key nodes based on degree centrality
    degree_centrality = nx.degree_centrality(G)
    print("Printing degree centrality: ", degree_centrality)
    return [], []

# Helper function to extract key phrases
def extract_key_phrases(content, top_n=50):
    """Extract top-n key phrases using KeyBERT."""
    keywords = keybert_model.extract_keywords(content, keyphrase_ngram_range=(1, 2), use_mmr=True, diversity=0.7,
                                              stop_words="english", top_n=top_n)
    print("These are the keywords:", keywords)
    return [kw[0] for kw in keywords]


def generate_knowledge_graph(data):
    """Generate and return the knowledge graph."""
    print("I'm in the graph function")
    G = nx.Graph()
    key_phrases = []
    phrase_to_source = {}

    # Extract phrases from the string data
    key_phrases = extract_key_phrases(data, top_n=50)
    embeddings = sentence_embedding_model.encode(key_phrases)

    # Create a similarity matrix that records the semantic similarity of the extracted key phrases
    similarity_matrix = cosine_similarity(embeddings)

    # Threshold factor. Increase number to decrease noise. Decrease for vice-versa.
    threshold = 0.3

    for i, phrase_i in enumerate(key_phrases):
        G.add_node(phrase_i, label=phrase_i, color="#008080")
        similarity_scores = [
            (key_phrases[j], similarity_matrix[i][j])
            for j in range(len(key_phrases))
            if i != j  # Only avoid self-connections
        ]
        sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[:3]
        for phrase_j, score in sorted_scores:
            if score > threshold:
                G.add_edge(phrase_i, phrase_j)
    # # # --- Visualization code ---
    # plt.figure(figsize=(12, 12))  # Set the size of the figure for better readability
    #
    # # Use a layout algorithm to position the nodes
    # pos = nx.spring_layout(G, k=0.15, iterations=20)
    #
    # # Draw the nodes, edges, and labels
    # nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1500, edge_color='gray', font_size=8)
    #
    # # Add a title and display the plot
    # plt.title("Knowledge Graph Visualization")
    # plt.show()
    # # # --- End of visualization code ---

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


def main():

    # Load dataset
    ds = load_dataset("kmfoda/booksum")
    ds_train = load_dataset("kmfoda/booksum", split='train')

    # Extract the source content and summaries directly
    chapters = ds_train[:3]['chapter']  # The original chapter text
    summaries = ds_train[:3]['summary_text']  # The summary text


    responses = []
    for chapter in chapters:

        G = generate_knowledge_graph(chapter)

        crux_nodes, neighbour_nodes = identify_crux_nodes(G)

        # print(G.nodes)
        # print(G.edges)

        prompt = (
            "You are a summary generator. I would like you to generate a summary out of the following content: "
            f"{chapter}"
            f"Utilize this graph {G} to improve response. The graph nodes represent the topics you must pay attention to."
        )
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

    print("Key Phrase Graph Summary Evaluation Results:")
    for i, (pred, ref) in enumerate(zip(predictions, summaries)):
        scores = scorer.score(ref, pred)
        P, R, F1 = bert_scorer.score([ref], [pred])
        bleu = sacrebleu.corpus_bleu([pred], [[ref]])
        print(f"\nSummary {i+1}:")
        print(f"  ROUGE-1: {scores['rouge1'].fmeasure:.4f}")
        print(f"  ROUGE-2: {scores['rouge2'].fmeasure:.4f}")
        print(f"  ROUGE-L: {scores['rougeL'].fmeasure:.4f}")
        print(f"  BERTScore Precision: {P.mean():.4f}, Recall: {R.mean():.4f}, F1: {F1.mean():.4f}")
        print(f"Paragraph BLEU = {bleu.score:.2f}")

if __name__ == '__main__':
    main()