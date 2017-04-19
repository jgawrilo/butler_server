# -*- coding: utf-8 -*-

#from google import search
import requests
import codecs
from bs4 import BeautifulSoup
import re
import nltk
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
# load nltk's SnowballStemmer as variabled 'stemmer'
from nltk.stem.snowball import SnowballStemmer
import glob, os
from readability.readability import Document
from selenium import webdriver
import spacy
import pyap
import json
from table import HTMLTableParser
from subprocess import Popen
from sys import stderr

from flask import Flask, request, Response

try:
    import Image
except ImportError:
    from PIL import Image

import pytesseract

from google import google

reload(sys)  
sys.setdefaultencoding('utf8')

stemmer = SnowballStemmer("english")

nlp = spacy.load('en')

page_dict = {}
entity_dict = {}
address_dict = {}

print "here"

app = Flask(__name__)

# Social Sites
social_mappings = [
    {"site":"LinkedIn","urls":["https://www.linkedin.com/"]},
    {"site":"Instagram","urls":["https://www.instagram.com/"],"left_split":"https://www.instagram.com/",
    "profile_class":"_79dar","image_class":"_iv4d5"},
    {"site":"Github","urls":["https://github.com/"],"left_split":"https://github.com/"},
    {"site":"Pinterest","urls":["https://www.pinterest.com/"],"left_split":"https://www.pinterest.com/"},
    {"site":"Facebook","urls":["https://www.facebook.com/"], "left_split":"https://www.facebook.com/"},
    {"site":"Twitter","urls":["https://twitter.com/"],"left_split":"https://twitter.com/"},
    {"site":"YouTube","urls":["https://www.youtube.com"],"left_split":"https://twitter.com/"}
    #{"site":"F6S","urls":["https://www.f6s.com/"]}
    ]

def visible(element):
    if element.parent.name in ['style', 'script', '[document]', 'head', 'title']:
        return False
    elif re.match('<!--.*-->', str(element.encode('utf-8'))):
        return False
    return True

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

def process_entity_relations(entity_relations_str):
    # format is ollie.
    entity_relations = list()
    for s in entity_relations_str:
        entity_relations.append(s[s.find("(") + 1:s.find(")")].split(';'))
    return entity_relations

def getEntities(text):
    ent_dict = {}
    for ent in nlp(unicode(text)).ents:
        ent_txt = ' '.join(ent.text.split()).upper().replace("\\n","").strip()
        ent_dict[(ent_txt,ent.label_)] = ent_dict.get((ent_txt,ent.label_),0)
        ent_dict[(ent_txt,ent.label_)] += 1
        entity_dict[ent_txt] = entity_dict.get(ent_txt,"e"+str(len(entity_dict)))
    return [{"value":x[0],"type":x[1],"count":ent_dict[x],"id":entity_dict[x[0]]} for x in ent_dict if x[1] not in 
    ["CARDINAL","DATE","MONEY","PERCENT","TIME","WORK_OF_ART"]]

def getPhoneNumbers(text):
    pass

def getEmails(text):
    pass

def getAddresses(text):
    addresses = pyap.parse(text, country='US')
    addresses = map(str,addresses)
    for address in addresses:
        address_dict[address] = address_dict.get(address,"a"+str(len(address_dict)))
    return map(lambda x: {"id":address_dict[x],"value":x},addresses)

def getRelationships(location):
    out = "/Users/jgawrilow/j/hiearchical_clustering/data/rel_"+location
    command = 'cd /Users/jgawrilow/Desktop/stanford-corenlp-full-2016-10-31; java -mx12g -cp "*" ' \
               'edu.stanford.nlp.naturalli.OpenIE {} -resolve_coref true -triple.strict true -format ollie > {}'. \
        format("/Users/jgawrilow/j/hiearchical_clustering/data/"+location, out)


    java_process = Popen(command, stdout=stderr, stderr=open(os.devnull, 'w'), shell=True)
    java_process.wait()
    assert not java_process.returncode, 'ERROR: Call to stanford_ie exited with a non-zero code status.'

    with open(out, 'r') as output_file:
        results_str = output_file.readlines()
    results = process_entity_relations(results_str)

def get_tables(url):
    tp = HTMLTableParser()
    tables = tp.parse_url(url)
    for table in tables:
        print table.head()

def get_html(url):
    headers = headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2172.95 Safari/537.36'}
    response = requests.get(url,headers=headers)
    html = response.content.encode("utf-8","ignore")
    return html


def get_screenshot_text(url,i):
    br = webdriver.PhantomJS()
    br.get(url)
    br.save_screenshot("data/" + str(i) + ".png")
    br.quit

    return pytesseract.image_to_string(Image.open("data/" + str(i) + ".png"))

def get_text_title(html):
    soup = BeautifulSoup(html, "lxml")
    data = soup.findAll(text=True)
    result = filter(visible, data)
    text = " ".join(result).encode("utf-8","ignore")
    title = ""
    if soup.title != None:
        title = soup.title.string
    return text.strip(), title.strip()


def get_readability_text(html):
    readable_article = Document(html).summary()
    readable_title = Document(html).short_title()
    return readable_article.strip()


def get_urls(term,num_pages=10):
    #return [url for url in search(term, stop=10)]
    num_page = 1
    search_results = google.search(term, num_page)
    return [x.link for x in search_results]

def build_social_json(url,ptype):
    page_dict[url] = page_dict.get(url,"p"+str(len(page_dict)))
    return {
        "url":url,
        "id":page_dict[url],
        "title":None,
        "profile":{
            "names":[],
            "emails":[],
            "phone_numbers":[],
            "addresses":[],
            "relationships":[],
            "usernames":[],
            "other":[],
            "images":[],
            "videos":[]
        },
        "entities":[],
        "type":ptype,
    }

def build_json(url,title,entities,addresses,ptype):
    page_dict[url] = page_dict.get(url,"p"+str(len(page_dict)))
    return {
        "url":url,
        "id":page_dict[url],
        "title":title,
        "profile":{
            "names":[],
            "emails":[],
            "phone_numbers":[],
            "addresses":addresses,
            "relationships":[],
            "usernames":[],
            "other":[],
            "images":[],
            "videos":[]
        },
        "entities":entities,
        "type":ptype, 
    }

def updateAllNodes (node):
    if len(node["children"]) == 0:
        return 1
    for child in node["children"]:
        node["count"] += updateAllNodes(child)
    return node["count"]


def add_node(node, parent, urls ):
    # First create the new node and append it to its parent's children
    url = None
    count = 0
    if node.id < len(urls):
        url = urls[node.id]
        count = 1
    newNode = dict( node_id=node.id, url=url, children=[], count=count)
    parent["children"].append( newNode )

    # Recursively add the current node's children
    if node.left: add_node( node.left, newNode, urls )
    if node.right: add_node( node.right, newNode, urls )


# Called when twitter is scraped...
@app.route('/search/', methods=['GET'])
def handle_search():
    q = request.args.get("q")
    print q
    
    filelist = glob.glob("data/*")
    for f in filelist:
        os.remove(f)

    idx_out = codecs.open("data/idx.txt","w",encoding="utf8")

    #define vectorizer parameters
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))
    

    urls = get_urls(q)
    texts = []
    good_urls = []
    entries = []
    for i,url in enumerate(urls):
        print url
        html = ""
        text = ""
        readable_text = ""

        if url.endswith(".pdf"):
            continue

        if any(map(url.startswith,map(lambda x: x["urls"][0],social_mappings))):
            data = build_social_json(url,"social")
        else:
            try:
                html = get_html(url)
                text,title = get_text_title(html)
                readable_text = get_readability_text(html)
                addresses = getAddresses(text)
                entities = getEntities(text)
                #ss_text = get_screenshot_text(url,i)
                data = build_json(url,title,entities,addresses,"page")
            except (UnicodeDecodeError,IOError,Exception):
                print "Error"
                continue
            if text == None or text.strip() == "":
                print "No Text"
                continue
        texts.append(text)

        #get_tables(url)
        
        good_urls.append(url)
        with codecs.open("data/"+str(i)+".html","w",encoding="utf8",errors="ignore") as out:
            out.write(html)
        with codecs.open("data/"+str(i)+".txt","w",encoding="utf8",errors="ignore") as out:
            out.write(text)
        with codecs.open("data/"+str(i)+".rtxt","w",encoding="utf8",errors="ignore") as out:
            out.write(readable_text)
        with codecs.open("data/"+str(i)+".json","w",encoding="utf8",errors="ignore") as out:
            out.write(json.dumps(data,indent=2))
            entries.append(data)
        #with codecs.open("data/"+str(i)+".sstxt","w",encoding="utf8",errors="ignore") as out:
        #    out.write(ss_text)
        #print getRelationships(str(i)+".txt")
        idx_out.write(str(i) + "\t" + url + "\n")
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts) #fit the vectorizer to synopses
    #print(tfidf_matrix.shape)
    from sklearn.metrics.pairwise import cosine_similarity
    dist = 1 - cosine_similarity(tfidf_matrix)
    idx_out.close()
    #print dist
    import matplotlib.pyplot as plt

    from scipy.cluster.hierarchy import ward, dendrogram,linkage, to_tree

    linkage_matrix = linkage(dist) #define the linkage_matrix using ward clustering pre-computed distances
    #print good_urls
    #print linkage_matrix

    tree = to_tree(linkage_matrix)
    #print tree
    d3Dendro = dict(children=[], name="Top",count=len(good_urls))
    add_node(tree, d3Dendro, good_urls)

    updateAllNodes(d3Dendro["children"][0])

    return_data = {"pages":entries,"treemap":d3Dendro["children"][0]}

    with codecs.open("data/data.json","w",encoding="utf8",errors="ignore") as out:
        out.write(json.dumps(return_data,indent=2))

    resp = Response(json.dumps(return_data))
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

if __name__ == "__main__":
    app.debug=True
    #context = ('server.crt', 'server.key')

    print "Running..."
    app.run(threaded=True,
        host="0.0.0.0",
        port=(5000)) #ssl_context=context)





    #print json.dumps(d3Dendro["children"][0],indent=2)



    '''

    fig, ax = plt.subplots(figsize=(30, 20)) # set size

    titles = ["0","1","2"]
    ax = dendrogram(linkage_matrix, orientation="right", labels=good_urls);

    plt.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')

    #plt.tight_layout() #show plot with tight layout

    #uncomment below to save figure
    plt.savefig('ward_clusters.png', dpi=200) #save figure as ward_clusters

    

    plt.close()
    '''

