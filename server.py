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
from fuzzywuzzy import fuzz

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

total_count = 0

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

def process_entity_relations(entity_relations_str,entities):
    entity_set = set()
    for e in entities:
        entity_set.add(e["value"])
    # format is ollie.
    return_list = []
    entity_relations = list()
    for s in entity_relations_str:
        a,b,c = s[s.find("(") + 1:s.find(")")].split(';')
        if a.strip() == "Justin Gawrilow":
            return_list.append([{"id":"other"+hashlib.md5(" ".join([x[0],x[1],x[2]])).hexdigest(),"value":" ".join([x[0],x[1],x[2]])} for x in entity_relations])
            entity_relations = []
            continue
        if a.upper() in entity_set or b.upper() in entity_set:
            entity_relations.append(s[s.find("(") + 1:s.find(")")].split(';'))
    return_list.append([{"id":"other"+hashlib.md5(" ".join([x[0],x[1],x[2]])).hexdigest(),"value":" ".join([x[0],x[1],x[2]])} for x in entity_relations])
    return return_list

def getEntities(text,likes,unlikes):
    ent_dict = {}
    for ent in nlp(unicode(text)).ents:
        # normaalizing text
        ent_txt = ' '.join(ent.text.split()).upper().replace("\\n","").strip()
        ent_txt = ent_txt.split("'S")[0]
        ent_txt = ''.join([i for i in ent_txt if not i.isdigit()])

        ent_dict[(ent_txt,ent.label_)] = ent_dict.get((ent_txt,ent.label_),0)
        ent_dict[(ent_txt,ent.label_)] += 1
    return [{"value":x[0],"type":x[1],"count":ent_dict[x],"id":"entity"+hashlib.md5(x[0] + "->" + x[1]).hexdigest()} for x in ent_dict if x[1] not in 
    ["CARDINAL","DATE","MONEY","PERCENT","TIME","WORK_OF_ART"] and "entity"+hashlib.md5(x[0] + "->" + x[1]).hexdigest() not in unlikes]

def getPhoneNumbers(text,likes,unlikes):
    phones = []
    for match in re.finditer(r"\(?\b[2-9][0-9]{2}\)?[-. ]?[2-9][0-9]{2}[-. ]?[0-9]{4}\b", text):
        match = match.group()
        pid = "phone" + hashlib.md5(match).hexdigest()
        if pid not in unlikes:
            phones.append({"id":pid,"value":match})
    return phones

def get_emails(text,likes,unlikes):
    ret = []
    for email in (email[0] for email in re.findall(regex, text) if not email[0].startswith('//')):
        eid = "email"+hashlib.md5(email).hexdigest()
        if eid not in unlikes:
            ret.append({"id":eid,"value":email})
    return ret

def doLDA(docu,level,last_count):
    global total_count
    total_count += 1
    # remove common words and tokenize
    slots, documents, urls = [],[],[]
    for x in docu:
        slots.append(x[1])
        documents.append(x[0])
        urls.append(x[2])

    stoplist = set(stopwords.words('english'))
    texts = [[word for word in document.lower().split() if word not in stoplist and not is_float(word)]
             for document in documents]

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    hdp = HdpModel(corpus,dictionary,random_state=7)
    lda = hdp.suggested_lda_model()

    topic_answers_dict = {"count":len(documents),"url":None,"level":level,"children":[],"node_id":total_count}
    result_dict = {}
    for i in range(len(documents)):
        entries = sorted(lda[corpus[i]],key=lambda x: x[1],reverse=True)
        if entries:
            d = lda.show_topics(-1,formatted=False)[entries[0][0]]
            ret = {"text":documents[i],"url":urls[i],"row":i, "slot":slots[i], "number":d[0],"scores":map(lambda x:{"value":x[0].replace("\"",""),"score":x[1]},d[1])[:5]}
            result_dict[d[0]] = result_dict.get(d[0],[])
            result_dict[d[0]].append(ret)
        else:
            print "issue..."
    for k in result_dict:
        if len(docu) == last_count:
            for res in result_dict[k]:
                topic_answers_dict["children"].append({"count":1,"url":res["url"],"node_id":slots[res["row"]],"level":level+1,"children":[],"scores":res["scores"]})
        elif len(result_dict[k]) > 1:
            docs = [(x["text"],x["slot"],x["url"]) for x in result_dict[k]]
            topic_answers_dict["children"].append(doLDA(docs,level+1,len(docu)))
        else:
            topic_answers_dict["children"].append({"count":1,"url":result_dict[k][0]["url"],"node_id":slots[result_dict[k][0]["row"]],"level":level+1,"children":[],"scores":result_dict[k][0]["scores"]})

    return topic_answers_dict

def getAddresses(text,likes,unlikes):
    addresses = pyap.parse(text, country='US')
    addresses = map(lambda x: " ".join(str(x).upper().split()),addresses)
    return filter(lambda x: x["id"] not in unlikes, map(lambda x: {"id":"address"+hashlib.md5(x).hexdigest(),"value":x},addresses))

def getRelationships(texts,entities):
    SPLIT_STRING = "\nJustin Gawrilow is great.\n"
    text = SPLIT_STRING.join(texts)
    file_name = hashlib.md5(text).hexdigest()
    out = "/Users/jgawrilow/j/butler_server/data/" + file_name + ".txt"
    myout = "/Users/jgawrilow/j/butler_server/data/my" + file_name + ".txt"
    with codecs.open(out,"w",encoding="utf8") as fout:
        fout.write(text)
    command = 'cd /Users/jgawrilow/Desktop/stanford-corenlp-full-2016-10-31; java -mx12g -cp "*" ' \
               'edu.stanford.nlp.naturalli.OpenIE {} -resolve_coref true -triple.strict true -format ollie > {}'. \
        format(out, myout)


    java_process = Popen(command, stdout=stderr, stderr=open(os.devnull, 'w'), shell=True)
    java_process.wait()
    assert not java_process.returncode, 'ERROR: Call to stanford_ie exited with a non-zero code status.'

    with open(myout, 'r') as output_file:
        results_str = output_file.readlines()
    results = process_entity_relations(results_str,entities)
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


def get_urls(terms,num_pages=1):
    results = []
    for term in terms:
        print "Searching for:" + term
        search_results = google.search(term, num_pages)
        results.extend([{"q":term,"url":x.link} for x in search_results])
    return results

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

def build_json(name,url,title,entities,addresses,ptype,rels,emails,phones,images,other):
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
            "other":other,
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

def prune_entities(entries):
    e_set = set()
    for e in entries:
        for n in e["entities"]:
            if n["type"] == "PERSON":
                # value, type, count, id
                e_set.add((n["value"]))

    for e in entries:
        for n in e["entities"]:
            if n["type"] == "PERSON":
                print "Testing", n["value"]
                for test in e_set:
                    if test != n["value"] and fuzz.ratio(test, n["value"]) >= 64:
                        print test, n["value"]



def build_profile(entries,likes,unlikes):
    #prune_entities(entries)
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
    other_dict = {}

    for e in entries:
        if e["type"] == "social":
            social_dict[e["id"]] = social_dict.get(e["id"],[e["url"],0,e["profile"]["images"][0]["url"],e["profile"]["usernames"][0]["value"]])
            social_dict[e["id"]][1] += 1
            continue
        for n in e["profile"]["other"]:
            other_dict[n["id"]] = other_dict.get(n["id"],[n["value"],0,set()])
            other_dict[n["id"]][1] += 1
            other_dict[n["id"]][2].add(json.dumps({"id":e["id"],"url":e["url"]}))
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

    main_profile["other"] = sorted([{"id":x,"value":other_dict[x][0],"count":other_dict[x][1],"from":list(map(json.loads,other_dict[x][2]))} for x in other_dict if x not in unlikes],key=lambda x: len(x["from"]),reverse=True)
    main_profile["phone_numbers"] = sorted([{"id":x,"value":phone_dict[x][0],"count":phone_dict[x][1],"from":list(map(json.loads,phone_dict[x][2]))} for x in phone_dict if x not in unlikes],key=lambda x: len(x["from"]),reverse=True)
    main_profile["addresses"] = sorted([{"id":x,"value":address_dict[x][0],"count":address_dict[x][1],"from":list(map(json.loads,address_dict[x][2]))} for x in address_dict if x not in unlikes],key=lambda x: len(x["from"]),reverse=True)
    main_profile["names"] = sorted([{"id":x,"value":names_dict[x][0],"count":names_dict[x][1],"from":list(map(json.loads, names_dict[x][2]))} for x in names_dict if x not in unlikes],key=lambda x: len(x["from"]),reverse=True)[:3]
    main_profile["emails"] = sorted([{"id":x,"value":email_dict[x][0],"count":email_dict[x][1],"from":list(map(json.loads, email_dict[x][2]))} for x in email_dict if x not in unlikes],key=lambda x: len(x["from"]),reverse=True)
    main_profile["relationships"] = sorted([{"id":x,"type":"connection","value":names_dict[x][0],"count":names_dict[x][1],"from":list(map(json.loads, names_dict[x][2]))} for x in names_dict if x not in unlikes],key=lambda x: len(x["from"]),reverse=True)[3:]
    main_profile["social_media"] = sorted([{"id":x,"url":social_dict[x][0],"count":social_dict[x][1],"profile_url":social_dict[x][2],"username":social_dict[x][3]} for x in social_dict if x not in unlikes],key=lambda x: x["count"],reverse=True)
    
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
    print "GET: Unlike", name, uid
    nes.index(index="butler", doc_type="unlikes",body={"name":name,"time":datetime.now().isoformat(),"id":uid})
    return resp

# Called when something is liked
@app.route('/like/', methods=['GET'])
def handle_like():
    name = request.args.get("name")
    lid = request.args.get("id")
    print "GET: Like", name, lid
    nes.index(index="butler", doc_type="likes",body={"name":name,"time":datetime.now().isoformat(),"id":lid})
    return resp

# Called when clear is clicked
@app.route('/clear/', methods=['GET'])
def handle_clear():
    name = request.args.get("name")
    print "GET: Clear", name
    qs = getQueries(name)
    q, num_pages = qs[-1]
    return_data = process_search([q],name,num_pages)
    nes.index(index="butler", doc_type="results",body={"name":name,"query":q,"data":return_data},id=name)
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

def do_reload(name):
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
    return results

# Called when reload is clicked
@app.route('/reload/', methods=['GET'])
def handle_reload():    
    name = request.args.get("name")
    print "GET: Reload", name
    results = do_reload(name)
    if len(results["hits"]["hits"]) >= 1:
        resp = Response(json.dumps(results["hits"]["hits"][-1]["_source"]["data"],indent=2))
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp

# Called when save/export is clicked
@app.route('/save_export/', methods=['GET'])
def handle_save():
    name = request.args.get("name")
    print "GET: Save/Export ->", name
    #qs = getQueries(name)
    #q, num_pages = qs[-1]
    #return_data = process_search([q],name,num_pages)
    #nes.index(index="butler", doc_type="results",body={"name":name,"query":q,"data":return_data},id=name)
    return resp

@app.route('/crunch/',methods=['GET'])
def handle_crunch():
    name = request.args.get("name")
    print "GET: Crunch ->", name
    qs = getQueries(name)
    q, num_pages = qs[-1]
    likes,unlikes = getLikesUnlikes(name)
    print likes
    new_searches = []
    results = do_reload(name)
    if len(results["hits"]["hits"]) >= 1:
        data = results["hits"]["hits"][-1]["_source"]["data"]
        for p in data["profile"]["names"] + data["profile"]["phone_numbers"] + data["profile"]["emails"] \
            + data["profile"]["addresses"] + data["profile"]["other"]:
            if p["id"] in likes:
                new_searches.append(p["value"])
        for p in data["pages"]:
            for e in p["entities"]:
                if e["id"] in likes:
                    new_searches.append(e["value"])

    queries = [q] + new_searches
    print queries
    return_data = process_search(queries,name,num_pages)
    resp = Response(json.dumps(return_data,indent=2))
    nes.index(index="butler", doc_type="results",body={"name":name,"query":q,"data":return_data},id=name)
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
    return_data = process_search([q],name,num_pages)
    resp = Response(json.dumps(return_data,indent=2))
    nes.index(index="butler", doc_type="results",body={"name":name,"query":q,"data":return_data},id=name)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

def populateEntries(entries,tree_stuff):
    if "scores" in tree_stuff:
        print json.dumps(tree_stuff,indent=2)
        entries[tree_stuff["node_id"]]["topic"] = { \
            "scores":tree_stuff["scores"],\
            "number":"",\
            "string":" ".join([x["value"] for x in tree_stuff["scores"]])\
        }
    else:
        for child in tree_stuff["children"]:
            populateEntries(entries,child) 


def process_search(q,name,num_pages=1):
    # Google results...
    urls = get_urls(q,num_pages)
    global total_count

    likes, unlikes = getLikesUnlikes(name)

    for query in q:
        nes.index(index="butler", doc_type="queries",body={"name":name,"query":query,"time":datetime.now().isoformat(), "num_pages":num_pages})

    texts = []
    good_urls = []
    entries = []
    all_entities = []
    for i,url_obj in enumerate(urls):
        url = url_obj["url"]
        query = url_obj["q"]
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
                addresses = getAddresses(text,likes,unlikes)
                entities = getEntities(text,likes,unlikes)
                emails = get_emails(text,likes,unlikes)
                phones = getPhoneNumbers(text,likes,unlikes)
                if len(addresses) > 5 or len(emails) > 5 or len(phones) > 5:
                    print "Too many of something..."
                    continue
                all_entities.extend(entities)
                #images = get_images(url)
                #ss_text = get_screenshot_text(url,i)
            except (UnicodeDecodeError,IOError,haul.exceptions.RetrieveError):
                print "Error..."
                continue
            if text == None or text.strip() == "":
                print "No Text..."
                continue
        texts.append(text)
        nes.index(index="butler", doc_type="texts",body={"name":name,"query":query,"time":datetime.now().isoformat(),
            "url":url,"text":text})

        #get_tables(url,i)
        
        good_urls.append(url)

        if not social:
            data = build_json(name,url,title,entities,addresses,"page",rels,emails,phones,images,[])
        entries.append(data)

    #print "Processing Relationships", len(texts), len(entries)
    #all_rels = getRelationships(texts,all_entities)
    #print len(all_rels)
    #print all_rels
    #for i,rel in enumerate(all_rels):
    #    entries[i]["profile"]["other"] = rel


    #tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem)

    doTexts = []
    for i, text in enumerate(texts):
        doTexts.append((text,i,good_urls[i]))

    print "Pre pickle..."
    pickle.dump( texts , open( "text.p", "wb" ) )
    pickle.dump( good_urls , open( "urls.p", "wb" ) )
    print "Pickled..."

    total_count = len(doTexts)

    tree_stuff = doLDA(doTexts,0,None)

    populateEntries(entries,tree_stuff)
    
    #ldaTexts,answers = doLDA(texts,q)

    #tfidf_matrix = tfidf_vectorizer.fit_transform(ldaTexts) #fit the vectorizer to synopses

    #for i,a in enumerate(answers):
    #    if a:
    #        a["scores"] = a["scores"][:5]
    #        a["string"] = ",".join(map(lambda x:x["value"], a["scores"]))
    #    entries[i]["topic"] = a

    #dist = 1 - cosine_similarity(tfidf_matrix)

    #linkage_matrix = linkage(dist) #define the linkage_matrix using ward clustering pre-computed distances

    #tree = to_tree(linkage_matrix)
    #print tree
    #d3Dendro = dict(children=[], name="Top",count=len(good_urls))
    #add_node(tree, d3Dendro, good_urls)

    #updateAllNodes(d3Dendro["children"][0])

    profile = build_profile(entries,likes,unlikes)

    return_data = {"profile":profile,"pages":entries,"treemap":tree_stuff}

    return return_data

# Called when twitter is scraped...
@app.route('/search/', methods=['GET'])
def handle_search():
    q = request.args.get("q")
    name = request.args.get("name")
    num_pages = int(request.args.get("n",1))
    print "GET: Search ->", name, q, num_pages

    return_data = process_search([q],name,num_pages)

    resp = Response(json.dumps(return_data,indent=2))
    nes.index(index="butler", doc_type="results",body={"name":name,"query":q,"data":return_data},id=name)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

if __name__ == "__main__":
    app.debug=True
    print "Running..."
    app.run(threaded=True,
        host="0.0.0.0",
        port=(5000))
    

