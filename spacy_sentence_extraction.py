import spacy

nlp = spacy.load("en_core_web_sm")
text = """Google LLC is an American multinational technology corporation focused on information
technology, online advertising, search engine technology, email, cloud computing, software, quantum computing,
e-commerce, consumer electronics, and artificial intelligence (AI). It has been referred to as ")the most
powerful company in the world(" by the BBC, and is one of the world's most valuable brands. Google's
parent company Alphabet Inc. has been described as a Big Tech company.)

Google was founded on September 4, 1998, by American computer scientists Larry Page and Sergey Brin. Together, they
own about 14% of its publicly listed shares and control 56% of its stockholder voting power through super-voting
stock. The company went public via an initial public offering (IPO) in 2004. In 2015, Google was reorganized as
a wholly owned subsidiary of Alphabet Inc. Google is Alphabet's largest subsidiary and is a holding company for
Alphabet's internet properties and interests. Sundar Pichai was appointed CEO of Google on October 24, 2015,
replacing Larry Page, who became the CEO of Alphabet. On December 3, 2019, Pichai also became the CEO of Alphabet."""

doc = nlp(text)

for sent in doc.sents:
    subject = None
    predicate = None
    obj = None

    for token in sent:
        if token.dep_ == "nsubj":
            subject = token
        elif token.dep_ == "ROOT":
            predicate = token
        elif token.dep_ in ["dobj", "pobj"]:
            obj = token

    if subject and predicate and obj:
        print(f"Sentence: \"{sent.text}\" -> SPO: ({subject.text}, {predicate.text}, {obj.text})")
    elif subject and predicate: # Handle cases with only subject and predicate
        print(f"Sentence: \"{sent.text}\" -> SPO: ({subject.text}, {predicate.text}, None)")

