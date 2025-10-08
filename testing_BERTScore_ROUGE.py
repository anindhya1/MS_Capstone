from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertForMaskedLM, BertModel
from bert_score import BERTScorer
from rouge import Rouge
from rouge_score import rouge_scorer

text1 = "Stars shine, and the sky is covered with a blue blanket."
text2 = "Stars shine, and the sky is adorned with a bright color."

scorer = BERTScorer(model_type='bert-base-uncased')
P, R, F1 = scorer.score([text1], [text2])
print(f"BERTScore Precision: {P.mean():.4f}, Recall: {R.mean():.4f}, F1: {F1.mean():.4f}")


scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
scores = scorer.score(text1, text2)
print(f"ROUGE-1 Precision: {scores['rouge1'].precision:.4f}, Recall: {scores['rouge1'].recall:.4f}, F1: {scores['rouge1'].fmeasure:.4f}")
print(f"ROUGE-2 Precision: {scores['rouge2'].precision:.4f}, Recall: {scores['rouge2'].recall:.4f}, F1: {scores['rouge2'].fmeasure:.4f}")
print(f"ROUGE-L Precision: {scores['rougeL'].precision:.4f}, Recall: {scores['rougeL'].recall:.4f}, F1: {scores['rougeL'].fmeasure:.4f}")