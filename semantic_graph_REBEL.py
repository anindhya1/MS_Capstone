# needed to load the REBEL model
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import math
import torch

# wrapper for wikipedia API
import wikipedia

# scraping of web articles
from newspaper import Article, ArticleException

# google news scraping
from GoogleNews import GoogleNews

# graph visualization
from pyvis.network import Network

# show HTML in notebook
import IPython

from lxml.html.clean import Cleaner
import requests
from datasets import load_dataset
from rouge_score import rouge_scorer
from bert_score import BERTScorer
import networkx as nx
from pyvis.network import Network # to visualize the graph in HTML
import sacrebleu
import json
import nltk
from summarizer import Summarizer


# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")

bert_extractive_model = Summarizer()  # default BERT model

def preprocess_with_bertsum(text: str, ratio: float = 0.5) -> str:
    """
    Use a BERT-based extractive summarizer to select the most important
    sentences from `text`, and return a compressed version consisting
    only of those sentences.

    `ratio` is the fraction of sentences to keep (e.g., 0.5 keeps ~50%).
    """
    text = text.strip()
    if not text:
        return text

    # Let the summarizer do the sentence selection.
    # return_as_list=True gives you a list of selected sentences in order.
    selected_sentences = bert_extractive_model(
        text,
        ratio=ratio,
        return_as_list=True
    )

    if not selected_sentences:
        # fallback to full text if summarizer returns nothing
        return text

    compressed_text = " ".join(selected_sentences)
    return compressed_text


# from https://huggingface.co/Babelscape/rebel-large
def extract_relations_from_model_output(text):
    relations = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    text_replaced = text.replace("<s>", "").replace("<pad>", "").replace("</s>", "")
    for token in text_replaced.split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                relations.append({
                    'head': subject.strip(),
                    'type': relation.strip(),
                    'tail': object_.strip()
                })
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                relations.append({
                    'head': subject.strip(),
                    'type': relation.strip(),
                    'tail': object_.strip()
                })
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        relations.append({
            'head': subject.strip(),
            'type': relation.strip(),
            'tail': object_.strip()
        })
    return relations

# knowledge base class
class KB():
    def __init__(self):
        self.entities = {}  # { entity_title: {...} }
        self.relations = []  # [ head: entity_title, type: ..., tail: entity_title,
        # meta: { article_url: { spans: [...] } } ]
        self.sources = {}  # { article_url: {...} }

    def are_relations_equal(self, r1, r2):
        return all(r1[attr] == r2[attr] for attr in ["head", "type", "tail"])

    def exists_relation(self, r1):
        return any(self.are_relations_equal(r1, r2) for r2 in self.relations)

    def merge_relations(self, r2):
        r1 = [r for r in self.relations
              if self.are_relations_equal(r2, r)][0]

        # if different article
        article_url = list(r2["meta"].keys())[0]
        if article_url not in r1["meta"]:
            r1["meta"][article_url] = r2["meta"][article_url]

        # if existing article
        else:
            spans_to_add = [span for span in r2["meta"][article_url]["spans"]
                            if span not in r1["meta"][article_url]["spans"]]
            r1["meta"][article_url]["spans"] += spans_to_add

    def merge_with_kb(self, kb2):
        for r in kb2.relations:
            article_url = list(r["meta"].keys())[0]
            source_data = kb2.sources[article_url]
            self.add_relation(r, source_data["article_title"],
                              source_data["article_publish_date"])

    def get_wikipedia_data(self, candidate_entity):
        try:
            page = wikipedia.page(candidate_entity, auto_suggest=False)
            entity_data = {
                "title": page.title,
                "url": page.url,
                "summary": page.summary
            }
            return entity_data
        except:
            return None

    def add_entity(self, e):
        self.entities[e["title"]] = {k: v for k, v in e.items() if k != "title"}

    def add_relation(self, r, article_title, article_publish_date):
        # check on wikipedia
        candidate_entities = [r["head"], r["tail"]]
        entities = [self.get_wikipedia_data(ent) for ent in candidate_entities]

        # if one entity does not exist, stop
        if any(ent is None for ent in entities):
            return

        # manage new entities
        for e in entities:
            self.add_entity(e)

        # rename relation entities with their wikipedia titles
        r["head"] = entities[0]["title"]
        r["tail"] = entities[1]["title"]

        # add source if not in kb
        article_url = list(r["meta"].keys())[0]
        if article_url not in self.sources:
            self.sources[article_url] = {
                "article_title": article_title,
                "article_publish_date": article_publish_date
            }

        # manage new relation
        if not self.exists_relation(r):
            self.relations.append(r)
        else:
            self.merge_relations(r)

    def print(self):
        print("Entities:")
        for e in self.entities.items():
            print(f"  {e}")
        print("Relations:")
        for r in self.relations:
            print(f"  {r}")
        print("Sources:")
        for s in self.sources.items():
            print(f"  {s}")

#
# # build a knowledge base from text
# def from_small_text_to_kb(text, verbose=False):
#     kb = KB()
#
#     # Tokenizer text
#     model_inputs = tokenizer(text, max_length=512, padding=True, truncation=True,
#                             return_tensors='pt')
#     if verbose:
#         print(f"Num tokens: {len(model_inputs['input_ids'][0])}")
#
#     # Generate
#     gen_kwargs = {
#         "max_length": 216,
#         "length_penalty": 0,
#         "num_beams": 3,
#         "num_return_sequences": 3
#     }
#     generated_tokens = model.generate(
#         **model_inputs,
#         **gen_kwargs,
#     )
#     decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
#
#     # create kb
#     for sentence_pred in decoded_preds:
#         relations = extract_relations_from_model_output(sentence_pred)
#         for r in relations:
#             kb.add_relation(r)
#
#     return kb
#
#
# # test the `from_small_text_to_kb` function
#
# text = "Napoleon Bonaparte (born Napoleone di Buonaparte; 15 August 1769 – 5 " \
# "May 1821), and later known by his regnal name Napoleon I, was a French military " \
# "and political leader who rose to prominence during the French Revolution and led " \
# "several successful campaigns during the Revolutionary Wars. He was the de facto " \
# "leader of the French Republic as First Consul from 1799 to 1804. As Napoleon I, " \
# "he was Emperor of the French from 1804 until 1814 and again in 1815. Napoleon's " \
# "political and cultural legacy has endured, and he has been one of the most " \
# "celebrated and controversial leaders in world history."
#
# kb = from_small_text_to_kb(text, verbose=True)
# kb.print()

# parse an article with newspaper3k
def get_article(url):
    article = Article(url)
    article.download()
    article.parse()
    return article

# extract the article from the url (along with metadata), extract relations and populate a KB
def from_url_to_kb(url):
    article = get_article(url)
    config = {
        "article_title": article.title,
        "article_publish_date": article.publish_date
    }
    # 🔹 NEW: BERTSum-style preprocessing
    compressed_text = preprocess_with_bertsum(article.text, ratio=0.5)
    kb = from_text_to_kb(compressed_text, article.url, **config)
    return kb

# get news links from google news
def get_news_links(query, lang="en", region="US", pages=1, max_links=100000):
    googlenews = GoogleNews(lang=lang, region=region)
    googlenews.search(query)
    all_urls = []
    for page in range(pages):
        googlenews.get_page(page)
        all_urls += googlenews.get_links()
    return list(set(all_urls))[:max_links]

# build a KB from multiple news links
def from_urls_to_kb(urls, verbose=False):
    kb = KB()
    if verbose:
        print(f"{len(urls)} links to visit")
    for url in urls:
        if verbose:
            print(f"Visiting {url}...")
        try:
            kb_url = from_url_to_kb(url)
            kb.merge_with_kb(kb_url)
        except ArticleException:
            if verbose:
                print(f"  Couldn't download article at url {url}")
    return kb

# from KB to HTML visualization
def save_network_html(kb, filename="network.html"):
    # create network
    net = Network(directed=True, width="auto", height="700px", bgcolor="#eeeeee")

    # nodes
    color_entity = "#00FF00"
    for e in kb.entities:
        net.add_node(e, shape="circle", color=color_entity)

    # edges
    for r in kb.relations:
        net.add_edge(r["head"], r["tail"],
                    title=r["type"], label=r["type"])

    # save network
    net.repulsion(
        node_distance=200,
        central_gravity=0.2,
        spring_length=200,
        spring_strength=0.05,
        damping=0.09
    )
    net.set_edge_smooth('dynamic')
    net.show(filename, notebook=False)
    print("Printing net", net)
    return  net

# extract text from url, extract relations and populate the KB
def from_text_to_kb(text, article_url, span_length=128, article_title=None,
                    article_publish_date=None, verbose=False):
    # tokenize whole text
    inputs = tokenizer([text], return_tensors="pt")

    # compute span boundaries
    num_tokens = len(inputs["input_ids"][0])
    if verbose:
        print(f"Input has {num_tokens} tokens")
    num_spans = math.ceil(num_tokens / span_length)
    if verbose:
        print(f"Input has {num_spans} spans")
    overlap = math.ceil((num_spans * span_length - num_tokens) /
                        max(num_spans - 1, 1))
    spans_boundaries = []
    start = 0
    for i in range(num_spans):
        spans_boundaries.append([start + span_length * i,
                                 start + span_length * (i + 1)])
        start -= overlap
    if verbose:
        print(f"Span boundaries are {spans_boundaries}")

    # transform input with spans
    tensor_ids = [inputs["input_ids"][0][boundary[0]:boundary[1]]
                  for boundary in spans_boundaries]
    tensor_masks = [inputs["attention_mask"][0][boundary[0]:boundary[1]]
                    for boundary in spans_boundaries]
    inputs = {
        "input_ids": torch.stack(tensor_ids),
        "attention_mask": torch.stack(tensor_masks)
    }

    # generate relations
    num_return_sequences = 3
    gen_kwargs = {
        "max_length": 256,
        "length_penalty": 0,
        "num_beams": 3,
        "num_return_sequences": num_return_sequences
    }
    generated_tokens = model.generate(
        **inputs,
        **gen_kwargs,
    )

    # decode relations
    decoded_preds = tokenizer.batch_decode(generated_tokens,
                                           skip_special_tokens=False)

    # create kb
    kb = KB()
    i = 0
    for sentence_pred in decoded_preds:
        current_span_index = i // num_return_sequences
        relations = extract_relations_from_model_output(sentence_pred)
        for relation in relations:
            relation["meta"] = {
                article_url: {
                    "spans": [spans_boundaries[current_span_index]]
                }
            }
            kb.add_relation(relation, article_title, article_publish_date)
        i += 1

    return kb

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


# text = """
# Manchester United Football Club, commonly referred to as Man United (often stylised as Man Utd) or simply United, is a professional football club based in Old Trafford, Greater Manchester, England. They compete in the Premier League, the top tier of English football. Nicknamed the Red Devils, they were founded as Newton Heath LYR Football Club in 1878, but changed their name to Manchester United in 1902. After a spell playing in Clayton, Manchester, the club moved to their current stadium, Old Trafford, in 1910.
#
# Domestically, Manchester United have won a joint-record twenty top-flight league titles, thirteen FA Cups, six League Cups and a record twenty-one FA Community Shields. Additionally, in international football, they have won the European Cup/UEFA Champions League three times, and the UEFA Europa League, the UEFA Cup Winners' Cup, the UEFA Super Cup, the Intercontinental Cup and the FIFA Club World Cup once each.[7][8] Appointed as manager in 1945, Matt Busby built a team with an average age of just 22 nicknamed the Busby Babes that won successive league titles in the 1950s and became the first English club to compete in the European Cup. Eight players were killed in the Munich air disaster, but Busby rebuilt the team around star players George Best, Denis Law and Bobby Charlton – known as the United Trinity. They won two more league titles before becoming the first English club to win the European Cup in 1968.
#
# After Busby's retirement, Manchester United were unable to produce sustained success until the arrival of Alex Ferguson, who became the club's longest-serving and most successful manager, winning 38 trophies including 13 league titles, five FA Cups and two Champions League titles between 1986 and 2013.[9] In the 1998–99 season, under Ferguson, the club became the first in the history of English football to achieve the continental treble of the Premier League, FA Cup and UEFA Champions League.[10] In winning the UEFA Europa League under José Mourinho in 2016–17, they became one of five clubs to have won the original three main UEFA club competitions (the Champions League, Europa League and Cup Winners' Cup).
#
# Manchester United is one of the most widely supported football clubs in the world[11][12] and have rivalries with Liverpool, Manchester City, Arsenal and Leeds United. Manchester United was the highest-earning football club in the world for 2016–17, with an annual revenue of €676.3 million,[13] and the world's second-most-valuable football club in 2024, valued at £6.55 billion ($5.22 billion).[14] After being floated on the London Stock Exchange in 1991, the club was taken private in 2005 after a purchase by American businessman Malcolm Glazer valued at almost £800 million, of which over £500 million of borrowed money became the club's debt.[15] From 2012, some shares of the club were listed on the New York Stock Exchange, although the Glazer family retains overall ownership and control of the club.
#
# History
# See also: List of Manchester United F.C. seasons
# refer to caption
# A chart showing the progress of Manchester United through the English football league system, from joining as Newton Heath in 1892–93 to the present
# Early years (1878–1945)
# Main article: History of Manchester United F.C. (1878–1945)
# Manchester United were formed in 1878 as Newton Heath LYR Football Club by the Carriage and Wagon department of the Lancashire and Yorkshire Railway (LYR) depot at Newton Heath.[16] The team initially played games against other departments and railway companies, but on 20 November 1880, they competed in their first recorded match; wearing the colours of the railway company – green and gold – they were defeated 6–0 by Bolton Wanderers' reserve team.[17] By 1888, the club had become a founding member of The Combination, a regional football league. Following the league's dissolution after only one season, Newton Heath joined the newly formed Football Alliance, which ran for three seasons before being merged with The Football League. This resulted in the club starting the 1892–93 season in the First Division, by which time it had become independent of the railway company and dropped the "LYR" from its name.[16] After two seasons, the club was relegated to the Second Division.[16]
#
# A black-and-white photograph of a football team lining up before a match. Four players, wearing dark shirts, light shorts and dark socks, are seated. Four more players are standing immediately behind them, and three more are standing on a higher level on the back row. Two men in suits are standing on either side of the players.
# The Manchester United team at the start of the 1905–06 season, in which they were runners-up in the Second Division
# In January 1902, with debts of £2,670 – equivalent to £370,000 in 2023[nb 1] – the club was served with a winding-up order.[18] Captain Harry Stafford found four local businessmen, including John Henry Davies (who became club president), each willing to invest £500 in return for a direct interest in running the club and who subsequently changed the name;[19] on 24 April 1902, Manchester United was officially born.[20][nb 2] Under Ernest Mangnall, who assumed managerial duties in 1903, Manchester United finished as Second Division runners-up in 1906 and secured promotion to the First Division, which they won in 1908 – the club's first league title. The following season began with victory in the first ever Charity Shield[21] and ended with the club's first FA Cup title. Mangnall was considered a significant influence behind the team's move to Old Trafford in 1910, and Manchester United won the First Division for the second time in 1911.[22] At the end of the following season, however, Mangnall left the club to join Manchester City.[23]
#
# In 1922, three years after the resumption of football following the First World War, the club was relegated to the Second Division, where it remained until regaining promotion in 1925. Relegated again in 1931, Manchester United became a yo-yo club, achieving its all-time lowest position of 20th place in the Second Division in 1934, under secretary-manager Scott Duncan, narrowly avoiding relegation to the Third Division. Two years later, Duncan led the club to promotion before another relegation followed in 1937, which led to his resignation in November of that year. Following the death of principal benefactor John Henry Davies in October 1927, the club's finances deteriorated to the extent that Manchester United would likely have gone bankrupt had it not been for James W. Gibson, who, in December 1931, invested £2,000 and assumed control of the club.[24] In the 1938–39 season, the last year of football before the Second World War, the club finished 14th in the First Division.[24]
#
# Busby years (1945–1969)
# Main article: History of Manchester United F.C. (1945–1969)
# A black-and-white photograph of several people in suits and overcoats on the steps of an aircraft.
# The Busby Babes in 1955. Manager Matt Busby is pictured front right.
# In October 1945, the impending resumption of football after the war led to the managerial appointment of Matt Busby, who demanded an unprecedented level of control over team selection, player transfers. and training sessions.[25] Busby led the team to second-place league finishes in 1947, 1948. and 1949, and to FA Cup victory in 1948. In 1952, the club won the First Division, its first league title for 41 years.[26] They then won back-to-back league titles in 1956 and 1957; the squad, which had an average age of 22, was nicknamed "the Busby Babes" by the media, a testament to Busby's faith in his youth players.[27] In 1957, Manchester United became the first English team to compete in the European Cup, despite objections from The Football League, who had denied Chelsea the same opportunity the previous season.[28] En route to the semi-final, which they lost to Real Madrid, the team recorded a 10–0 victory over Belgian champions Anderlecht, which remains the club's biggest victory on record.[29]
#
# A stone tablet, inscribed with the image of a football pitch and several names. It is surrounded by a stone border in the shape of a football stadium. Above the tablet is a wooden carving of two men holding a large wreath.
# A plaque at Old Trafford in memory of those who died in the Munich air disaster, including players' names
# The following season, on the way home from a European Cup quarter-final victory against Red Star Belgrade, the aircraft carrying the Manchester United players, officials, and journalists crashed while attempting to take off after refuelling in Munich, Germany. The Munich air disaster of 6 February 1958 claimed 23 lives, including those of eight players – Geoff Bent, Roger Byrne, Eddie Colman, Duncan Edwards, Mark Jones, David Pegg, Tommy Taylor and Billy Whelan – and injured several more.[30][31]
#
#
# The United Trinity statue of George Best (left), Denis Law (centre) and Bobby Charlton (right) outside Old Trafford
# Assistant manager Jimmy Murphy took over as manager while Busby recovered from his injuries and the club's makeshift side reached the FA Cup final, which they lost to Bolton Wanderers. In recognition of the team's tragedy, UEFA invited the club to compete in the 1958–59 European Cup alongside eventual League champions Wolverhampton Wanderers. Despite approval from The Football Association, The Football League determined that the club should not enter the competition, since it had not qualified.[32][33] Busby rebuilt the team through the 1960s by signing players such as Denis Law and Paddy Crerand, who combined with the next generation of youth players – including George Best – to win the FA Cup in 1963. Busby rested several key players for the League game before the Cup Final, which gave Dennis Walker the chance to make his debut against Nottingham Forest on 20 May. Walker thus became the first Black player to represent United.[34] The following season, they finished second in the league, then won the title in 1965 and 1967. In 1968, Manchester United became the first English club to win the European Cup, beating Benfica 4–1 in the final[35] with a team that contained three European Footballers of the Year: Bobby Charlton, Denis Law and George Best.[36] They then represented Europe in the 1968 Intercontinental Cup against Estudiantes of Argentina, but defeat in the first leg in Buenos Aires meant a 1–1 draw at Old Trafford three weeks later was not enough to claim the title. Busby resigned as manager in 1969 before being replaced by the reserve team coach, former Manchester United player Wilf McGuinness.[37]"""
#
# kb = from_text_to_kb(text, article_url="")
# # kb.print()
#
# # # test the `from_url_to_kb` function
# # url = "https://finance.yahoo.com/news/microstrategy-bitcoin-millions-142143795.html"
# # kb = from_url_to_kb(url)
# # kb.print()
#
# # # test the `from_urls_to_kb` function
# # news_links = get_news_links("Google", pages=1, max_links=3)
# # kb = from_urls_to_kb(news_links, verbose=True)
# # kb.print()
#
# # extract KB from news about Google and visualize it
# # news_links = get_news_links("Google", pages=5, max_links=20)
# # kb = from_urls_to_kb(news_links, verbose=True)
# #
# filename = "network_3_google.html"
# save_network_html(kb, filename=filename)

def main():

    # Load dataset
    ds = load_dataset("kmfoda/booksum")
    ds_train = load_dataset("kmfoda/booksum", split='train')

    # Extract the source content and summaries directly
    chapters = ds_train[:3]['chapter']  # The original chapter text
    summaries = ds_train[:3]['summary_text']  # The summary text


    responses = []
    for chapter in chapters:
        # 🔹 NEW: BERTSum-style preprocessing of the raw chapter text
        compressed_chapter = preprocess_with_bertsum(chapter, ratio=0.5)

        # create knowledge base
        kb = from_text_to_kb(compressed_chapter, article_url="")

        # G = generate_knowledge_graph(chapter)
        #
        # crux_nodes, neighbour_nodes = identify_crux_nodes(G)

        # print(G.nodes)
        # print(G.edges)
        filename = "network_3_google.html"
        G = save_network_html(kb, filename=filename)

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

    print("\n\nUsing REBEL NLPlanet knowledge base-enhanced summaries:")
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