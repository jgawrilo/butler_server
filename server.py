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
from nltk.corpus import stopwords

from flask import Flask, request, Response, session, escape
from gensim import corpora
from collections import defaultdict
from gensim.models.hdpmodel import HdpModel
import haul
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pickle
import random
from datetime import datetime

from elasticsearch import Elasticsearch
import hashlib

nes = Elasticsearch(["http://localhost:9200/"])

from scipy.cluster.hierarchy import ward, dendrogram,linkage, to_tree

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

app = Flask(__name__)
app.config['SECRET_KEY'] = 'nitsuj'

resp = Response(json.dumps({"success":True}))
resp.headers['Access-Control-Allow-Origin'] = '*'

regex = re.compile(("([a-z0-9!#$%&'*+\/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+\/=?^_`"
                    "{|}~-]+)*(@|\sat\s)(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(\.|"
                    "\sdot\s))+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)"))

# Social Sites
social_mappings = [
    {"site":"Google","urls":["https://plus.google.com/"]},
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

stop = [
    "https://en.wikipedia.org"
]


def LoadUserAgents(uafile="user_agents.txt"):
    """
    uafile : string
        path to text file of user agents, one per line
    """
    uas = []
    with open(uafile, 'rb') as uaf:
        for ua in uaf.readlines():
            if ua:
                uas.append(ua.strip()[1:-1-1])
    random.shuffle(uas)
    return uas

# load the user agents, in random order
user_agents = LoadUserAgents(uafile="user_agents.txt")

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
    return [{"a":x[0],"rel":x[1],"b":x[2]} for x in entity_relations]

def getEntities(text):
    ent_dict = {}
    for ent in nlp(unicode(text)).ents:
        ent_txt = ' '.join(ent.text.split()).upper().replace("\\n","").strip()
        ent_dict[(ent_txt,ent.label_)] = ent_dict.get((ent_txt,ent.label_),0)
        ent_dict[(ent_txt,ent.label_)] += 1
    return [{"value":x[0],"type":x[1],"count":ent_dict[x],"id":"entity"+hashlib.md5(x[0] + "->" + x[1]).hexdigest()} for x in ent_dict if x[1] not in 
    ["CARDINAL","DATE","MONEY","PERCENT","TIME","WORK_OF_ART"]]

def getPhoneNumbers(text):
    phones = []
    for match in re.finditer(r"\(?\b[2-9][0-9]{2}\)?[-. ]?[2-9][0-9]{2}[-. ]?[0-9]{4}\b", text):
        match = match.group()
        if match not in dislike_page_set:
            pid = "phone" + hashlib.md5(match).hexdigest()
            phones.append({"id":pid,"value":match})
    return phones


def get_emails(text):
    ret = []
    for email in (email[0] for email in re.findall(regex, text) if not email[0].startswith('//')):
        ret.append({"id":"email"+hashlib.md5(email).hexdigest(),"value":email})
    return ret

def doLDA(documents,query):
    # remove common words and tokenize
    stoplist = set(stopwords.words('english')).union(set([x.lower() for x in query.split()]))
    texts = [[word for word in document.lower().split() if word not in stoplist and not is_float(word)]
             for document in documents]

    # remove words that appear only once
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    texts = [[token for token in text if frequency[token] > 1]
             for text in texts]

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    hdp = HdpModel(corpus,dictionary)

    topic_answers = []
    for x in range(len(documents)):
        #print x, corpus[x]
        entries = sorted(hdp[corpus[x]],key=lambda x: x[1],reverse=True)
        if entries:
            d = hdp.show_topics(formatted=False)[entries[0][0]]
            ret = {"number":d[0],"scores":map(lambda x:{"value":x[0].replace("\"",""),"score":x[1]},d[1])}
            topic_answers.append(ret)
        else:
            topic_answers.append(None)
    return topic_answers


def getAddresses(text):
    addresses = pyap.parse(text, country='US')
    addresses = map(lambda x: " ".join(str(x).upper().split()),addresses)
    return map(lambda x: {"id":"address"+hashlib.md5(x).hexdigest(),"value":x},addresses)

def getRelationships(location):
    out = "/Users/jgawrilow/j/butler_server/data/rel_"+location
    command = 'cd /Users/jgawrilow/Desktop/stanford-corenlp-full-2016-10-31; java -mx12g -cp "*" ' \
               'edu.stanford.nlp.naturalli.OpenIE {} -resolve_coref true -triple.strict true -format ollie > {}'. \
        format("/Users/jgawrilow/j/butler_server/data/"+location, out)


    java_process = Popen(command, stdout=stderr, stderr=open(os.devnull, 'w'), shell=True)
    java_process.wait()
    assert not java_process.returncode, 'ERROR: Call to stanford_ie exited with a non-zero code status.'

    with open(out, 'r') as output_file:
        results_str = output_file.readlines()
    results = process_entity_relations(results_str)
    return results

def get_tables(url,i):
    tp = HTMLTableParser()
    tables = tp.parse_url(url)
    for j, table in enumerate(tables):
        if table is not None:
            table.applymap(lambda x:x.strip().replace("\t"," ") if type(x) == str else x)
            table.to_csv("data/" + str(i) + "_"+ str(j) + ".csv",header=True,sep="\t")

def get_html(url):
    ua = random.choice(user_agents)  # select a random user agent
    headers = {
    "Connection" : "close",  # another way to cover tracks
    "User-Agent" : ua
    }
    response = requests.get(url,headers=headers)
    html = response.content.encode("utf-8","ignore")
    return html

def get_images(url):
    result = haul.find_images(url)
    return map(lambda x: url + x if x.startswith("/") else x,result.image_urls)

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


def get_urls(term,num_pages=1):
    search_results = google.search(term, num_pages)
    return [x.link for x in search_results]

def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return  False

def build_social_json(name, url,ptype):
    pid = "page" + hashlib.md5(url).hexdigest()
    data = {
        "name":name,
        "url":url,
        "id":pid,
        "title":None,
        "profile":{
            "names":[{
            "value":"Unknown"
            }],
            "emails":[],
            "phone_numbers":[],
            "addresses":[],
            "relationships":[],
            "usernames":[{
            "value":"Unknown"
            }],
            "other":[],
            "images":[{
            "url":"http://simpleicon.com/wp-content/uploads/user1.png"
            }],
            "videos":[]
        },
        "entities":[],
        "type":ptype,
    }
    nes.index(index="butler", doc_type="pages",body=data,id=pid)
    return data

def build_json(name,url,title,entities,addresses,ptype,rels,emails,phones,images):
    pid = "page" + hashlib.md5(url).hexdigest()
    data = {
        "name":name,
        "url":url,
        "id":pid,
        "title":title,
        "profile":{
            "names":[],
            "emails":emails,
            "phone_numbers":phones,
            "addresses":addresses,
            "relationships":rels,
            "usernames":[],
            "other":[],
            "images":images,
            "videos":[]
        },
        "entities":entities,
        "type":ptype, 
    }
    nes.index(index="butler", doc_type="pages", body=data, id=pid)
    return data

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

def build_profile(entries):
    main_profile = {
            "names":[],
            "emails":[],
            "phone_numbers":[],
            "addresses":[],
            "relationships":[],
            "other":[],
            "images":[],
            "videos":[],
            "social_media":[]
        }
    phone_dict = {}
    address_dict = {}
    names_dict = {}
    email_dict = {}
    social_dict = {}

    for e in entries:
        if e["type"] == "social":
            social_dict[e["id"]] = social_dict.get(e["id"],[e["url"],0,e["profile"]["images"][0]["url"],e["profile"]["usernames"][0]["value"]])
            social_dict[e["id"]][1] += 1
            continue

        for n in e["profile"]["phone_numbers"]:
            phone_dict[n["id"]] = phone_dict.get(n["id"],[n["value"],0,set()])
            phone_dict[n["id"]][1] += 1
            phone_dict[n["id"]][2].add(json.dumps({"id":e["id"],"url":e["url"]}))
        for n in e["profile"]["emails"]:
            email_dict[n["id"]] = email_dict.get(n["id"],[n["value"],0,set()])
            email_dict[n["id"]][1] += 1
            email_dict[n["id"]][2].add(json.dumps({"id":e["id"],"url":e["url"]}))
        for n in e["profile"]["addresses"]:
            address_dict[n["id"]] = address_dict.get(n["id"],[n["value"],0,set()])
            address_dict[n["id"]][1] += 1
            address_dict[n["id"]][2].add(json.dumps({"id":e["id"],"url":e["url"]}))
        for n in e["entities"]:
            if n["type"] == "PERSON":
                names_dict[n["id"]] = names_dict.get(n["id"],[n["value"],0,set()])
                names_dict[n["id"]][1] += n["count"]
                names_dict[n["id"]][2].add(json.dumps({"id":e["id"],"url":e["url"]}))

    main_profile["phone_numbers"] = sorted([{"id":x,"value":phone_dict[x][0],"count":phone_dict[x][1],"from":list(map(json.loads,phone_dict[x][2]))} for x in phone_dict],key=lambda x: len(x["from"]),reverse=True)
    main_profile["addresses"] = sorted([{"id":x,"value":address_dict[x][0],"count":address_dict[x][1],"from":list(map(json.loads,address_dict[x][2]))} for x in address_dict],key=lambda x: len(x["from"]),reverse=True)
    main_profile["names"] = sorted([{"id":x,"value":names_dict[x][0],"count":names_dict[x][1],"from":list(map(json.loads, names_dict[x][2]))} for x in names_dict],key=lambda x: len(x["from"]),reverse=True)[:3]
    main_profile["emails"] = sorted([{"id":x,"value":email_dict[x][0],"count":email_dict[x][1],"from":list(map(json.loads, email_dict[x][2]))} for x in email_dict],key=lambda x: len(x["from"]),reverse=True)
    main_profile["relationships"] = sorted([{"id":x,"type":"connection","value":names_dict[x][0],"count":names_dict[x][1],"from":list(map(json.loads, names_dict[x][2]))} for x in names_dict],key=lambda x: len(x["from"]),reverse=True)[3:]
    main_profile["social_media"] = sorted([{"id":x,"url":social_dict[x][0],"count":social_dict[x][1],"profile_url":social_dict[x][2],"username":social_dict[x][3]} for x in social_dict],key=lambda x: x["count"],reverse=True)
    
    return main_profile


# Called when a name is given
@app.route('/name/', methods=['GET'])
def handle_name():
    name = request.args.get("name")
    print "GET: Name", name
    nes.index(index="butler", doc_type="searches",body={"name":name},id=name)
    return resp

# Called when something is unliked
@app.route('/unlike/', methods=['GET'])
def handle_unlike():
    name = request.args.get("name")
    uid = request.args.get("id")
    print "GET: Like", name, uid
    nes.index(index="butler", doc_type="unlikes",body={"name":name,"time":datetime.now().isoformat(),"id":uid})
    return resp
    '''
    name = request.args.get("name")
    sid = request.args.get("id")
    if sid.startswith("phone"):
        for phone in phone_dict:
            if phone_dict[phone] == sid:
                print "found match!!!"
                dislike_phone_set.add(phone)
                for page in page_dict:
                    marks = []
                    for i,p in enumerate(page_dict[page]["profile"]["phone_numbers"]):
                        if sid == p["id"]:
                            marks.append(i)
                            break
                    for mark in marks:
                        del page_dict[page]["profile"]["phone_numbers"][mark]
    elif sid.startswith("p"):
        for page in page_dict:
            if page_dict[page]["id"] == sid:
                mark = page
                break
        del page_dict[mark]
        del text_dict[mark]
        dislike_page_set.add(mark)
    
    crunch()
    return resp
    '''

# Called when something is liked
@app.route('/like/', methods=['GET'])
def handle_like():
    name = request.args.get("name")
    lid = request.args.get("id")
    print "GET: Like", name, lid
    nes.index(index="butler", doc_type="likes",body={"name":name,"time":datetime.now().isoformat(),"id":lid})
    '''
    if sid.startswith("phone"):
        for phone in phone_dict:
            if phone_dict[phone] == sid:
                new_search(phone)
    '''
    return resp

# Called when clear is clicked
@app.route('/clear/', methods=['GET'])
def handle_clear():
    name = request.args.get("name")
    return resp

@app.route('/get_searches/',methods=['GET'])
def handle_get_searches():
    query = {
        "size": 0,
        "aggs" : {
            "searches" : {
                "terms" : { "field" : "name" }
            }
    }}
    result = nes.search(index="butler", doc_type="searches", body=query)
    resp = Response(json.dumps(result["aggregations"]["searches"]["buckets"],indent=2))
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

# Called when reload is clicked
@app.route('/reload/', methods=['GET'])
def handle_reload():    

    name = request.args.get("name")

    print "GET: Reload", name
    query = {
    "query": {
        "term": {
           "name": {
              "value": name
                   }
                }
            }
    }

    results = nes.search(index="butler", doc_type="results", body=query)
    print json.dumps(results)
    if len(results["hits"]["hits"]) == 1:
        resp = Response(json.dumps(results["hits"]["hits"][0]["_source"]["data"],indent=2))
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp

# Called when save/export is clicked
@app.route('/save_export/', methods=['GET'])
def handle_save():
    # Re-run everything here to get likes and dislikes, etc.
    name = request.args.get("name")
    return resp

@app.route('/crunch/',methods=['GET'])
def handle_crunch():
    #name = request.args.get("name")
    #return crunch(name)
    name = request.args.get("name")
    print "GET: Crunch ->", name
    qs = getQueries(name)
    q, num_pages = qs[-1]
    num_pages += 1
    nes.index(index="butler", doc_type="queries",body={"name":name,"query":q,"time":datetime.now().isoformat(), "num_pages":num_pages})
    return_data = process_search(q,name,num_pages)
    resp = Response(json.dumps(return_data,indent=2))
    nes.index(index="butler", doc_type="results",body={"name":name,"query":q,"data":return_data},id=name)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

def crunch(name):

    qs = getQueries(name)
    entries, texts, good_urls = [],[],[]
    for p in page_dict:
        entries.append(page_dict[p])
        good_urls.append(p)
        texts.append(text_dict[p])

    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

    tfidf_matrix = tfidf_vectorizer.fit_transform(texts) #fit the vectorizer to synopses
    answers = doLDA(texts,q)

    for i,a in enumerate(answers):
        entries[i]["topic"] = a

    dist = 1 - cosine_similarity(tfidf_matrix)

    linkage_matrix = linkage(dist) #define the linkage_matrix using ward clustering pre-computed distances

    tree = to_tree(linkage_matrix)

    d3Dendro = dict(children=[], name="Top",count=len(good_urls))
    add_node(tree, d3Dendro, good_urls)

    updateAllNodes(d3Dendro["children"][0])

    profile = build_profile(entries)

    return_data = {"profile":profile,"pages":entries,"treemap":d3Dendro["children"][0]}

    data_state = return_data

    resp = Response(json.dumps(return_data,indent=2))
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

def new_search(q,num_pages=1):
    global data_state
    print q
    print num_pages, "pages"
    urls = get_urls(q,num_pages)
    texts = []
    good_urls = []
    entries = []

    for url in page_dict:
        entries.append(page_dict[url])
        texts.append(text_dict[url])
        good_urls.append(url)
    for i,url in enumerate(urls):
        print url
        html = ""
        text = ""
        readable_text = ""
        title = ""
        entities = []
        addresses = []
        rels = []
        emails  = []
        phones = []
        images = []
        social = False

        if url in page_dict:
            entries.append(page_dict[url])
            texts.append(text_dict[url])
            continue

        if url in dislike_page_set or url.endswith(".pdf") or any(map(url.startswith,stop)):
            print "Skipping..."
            continue

        if any(map(url.startswith,map(lambda x: x["urls"][0],social_mappings))):
            print "Social"
            data = build_social_json(name,url,"social")
            social = True
        else:
            try:
                html = get_html(url)
                text,title = get_text_title(html)
                readable_text = get_readability_text(html)
                addresses = getAddresses(text)
                entities = getEntities(text)
                emails = get_emails(text)
                phones = getPhoneNumbers(text)
                #images = get_images(url)
                #ss_text = get_screenshot_text(url,i)
            except (UnicodeDecodeError,IOError,haul.exceptions.RetrieveError,Exception):
                print "Error..."
                continue
            if text == None or text.strip() == "":
                print "No Text..."
                continue
        texts.append(text)
        text_dict[url] = text

        #get_tables(url,i)
        
        good_urls.append(url)

        if text != "":
            pass
            #rels = getRelationships(str(i)+".txt")
        else:
            rels = []
        if not social:
            data = build_json(name,url,title,entities,addresses,"page",rels,emails,phones,images)
        entries.append(data)
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts) #fit the vectorizer to synopses
    answers = doLDA(texts,q)

    for i,a in enumerate(answers):
        entries[i]["topic"] = a

    dist = 1 - cosine_similarity(tfidf_matrix)

    linkage_matrix = linkage(dist)

    tree = to_tree(linkage_matrix)

    d3Dendro = dict(children=[], name="Top",count=len(good_urls))
    add_node(tree, d3Dendro, good_urls)

    updateAllNodes(d3Dendro["children"][0])

    profile = build_profile(entries)

    return_data = {"profile":profile,"pages":entries,"treemap":d3Dendro["children"][0]}

    data_state = return_data

    resp = Response(json.dumps(return_data,indent=2))
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


def getByURL(url, dtype, name):
    query = {
    "query": {
        "term": {
           "url": {
              "value": url
                   }
                }
            }
    }
    results = nes.search(index="butler", doc_type=dtype, body=query)

    if len(results["hits"]["hits"]) == 1:
        return results["hits"]["hits"][0]["_source"]
    else:
        return None


def getQueries(name):
    query = {
    "query": {
        "term": {
           "name": {
              "value": name
                   }
                }
            }
    }
    results = nes.search(index="butler", doc_type="queries", body=query)

    if len(results["hits"]["hits"]) >= 1:
        return map(lambda x:(x["_source"]["query"],x["_source"]["num_pages"]),results["hits"]["hits"])
    else:
        return []

def getLikesUnlikes(name):
    query = {
    "query": {
        "term": {
           "name": {
              "value": name
                   }
                }
            }
    }
    likes = nes.search(index="butler", doc_type="likes", body=query)
    unlikes = nes.search(index="butler", doc_type="unlikes", body=query)

    return set(map(lambda x: x["_source"]["id"], likes["hits"]["hits"])), set(map(lambda x: x["_source"]["id"], unlikes["hits"]["hits"]))


# Called when twitter is scraped...
@app.route('/next/', methods=['GET'])
def handle_next():
    name = request.args.get("name")
    print "GET: Next ->", name
    qs = getQueries(name)
    q, num_pages = qs[-1]
    num_pages += 1
    nes.index(index="butler", doc_type="queries",body={"name":name,"query":q,"time":datetime.now().isoformat(), "num_pages":num_pages})
    return_data = process_search(q,name,num_pages)
    resp = Response(json.dumps(return_data,indent=2))
    nes.index(index="butler", doc_type="results",body={"name":name,"query":q,"data":return_data},id=name)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp



def process_search(q,name,num_pages=1):
    # Google results...
    urls = get_urls(q,num_pages)

    likes, unlikes = getLikesUnlikes(name)

    texts = []
    good_urls = []
    entries = []
    for i,url in enumerate(urls):
        print url
        if url.endswith(".pdf") or any(map(url.startswith,stop)):
            print "Skipping..."
            continue

        html = ""
        text = ""
        readable_text = ""
        title = ""
        entities = []
        addresses = []
        rels = []
        emails  = []
        phones = []
        images = []
        social = False

        page = getByURL(url,"pages",name)

        if page:
            if page["id"] in unlikes:
                continue
            entries.append(page)
            texts.append(getByURL(url,"texts",name)["text"])
            good_urls.append(url)
            continue

        if any(map(url.startswith,map(lambda x: x["urls"][0],social_mappings))):
            print "Social"
            data = build_social_json(name,url,"social")
            social = True

        else:
            try:
                html = get_html(url)
                text,title = get_text_title(html)
                readable_text = get_readability_text(html)
                addresses = getAddresses(text)
                entities = getEntities(text)
                emails = get_emails(text)
                phones = getPhoneNumbers(text)
                #images = get_images(url)
                #ss_text = get_screenshot_text(url,i)
            except (UnicodeDecodeError,IOError,haul.exceptions.RetrieveError,Exception):
                print "Error..."
                continue
            if text == None or text.strip() == "":
                print "No Text..."
                continue
        texts.append(text)
        nes.index(index="butler", doc_type="texts",body={"name":name,"query":q,"time":datetime.now().isoformat(),
            "url":url,"text":text})

        #get_tables(url,i)
        
        good_urls.append(url)

        if text != "":
            pass
            #rels = getRelationships(str(i)+".txt")
        else:
            rels = []
        if not social:
            data = build_json(name,url,title,entities,addresses,"page",rels,emails,phones,images)
        entries.append(data)

    print texts

    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts) #fit the vectorizer to synopses
    answers = doLDA(texts,q)

    for i,a in enumerate(answers):
        if a:
            a["scores"] = a["scores"][:5]
            a["string"] = ",".join(map(lambda x:x["value"], a["scores"]))
        entries[i]["topic"] = a

    dist = 1 - cosine_similarity(tfidf_matrix)

    linkage_matrix = linkage(dist) #define the linkage_matrix using ward clustering pre-computed distances

    tree = to_tree(linkage_matrix)
    #print tree
    d3Dendro = dict(children=[], name="Top",count=len(good_urls))
    add_node(tree, d3Dendro, good_urls)

    updateAllNodes(d3Dendro["children"][0])

    profile = build_profile(entries)

    return_data = {"profile":profile,"pages":entries,"treemap":d3Dendro["children"][0]}

    return return_data

# Called when twitter is scraped...
@app.route('/search/', methods=['GET'])
def handle_search():
    q = request.args.get("q")
    name = request.args.get("name")
    num_pages = int(request.args.get("n",1))

    print "GET: Search ->", name, q, num_pages

    nes.index(index="butler", doc_type="queries",body={"name":name,"query":q,"time":datetime.now().isoformat(), "num_pages":1})

    return_data = process_search(q,name,num_pages)

    resp = Response(json.dumps(return_data,indent=2))
    nes.index(index="butler", doc_type="results",body={"name":name,"query":q,"data":return_data},id=name)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

if __name__ == "__main__":
    app.debug=True
    #context = ('server.crt', 'server.key')

    print "Running..."
    app.run(threaded=True,
        host="0.0.0.0",
        port=(5000)) #ssl_context=context)
    

