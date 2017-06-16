# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup
import re
import nltk
import sys
from nltk.stem.snowball import SnowballStemmer
from readability.readability import Document
from selenium import webdriver
import pyap
import json
from table import HTMLTableParser
from nltk.corpus import stopwords
from flask import Flask, request, Response, send_from_directory
from gensim import corpora
from gensim.models.hdpmodel import HdpModel
import haul
import random
from datetime import datetime
from elasticsearch import Elasticsearch
import hashlib
from fuzzywuzzy import fuzz
from google import google
import search2
import os

reload(sys)  
sys.setdefaultencoding('utf8')

config = json.load(open("config.json"))
nes = Elasticsearch([config["es"]])
stemmer = SnowballStemmer("english")

total_count = 0

app = Flask(__name__)
app.config['SECRET_KEY'] = 'kyc_butler'

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

def filterRels(texts,entities):
    ent_set = set([x["value"] for x in entities])
    new_texts = []
    for text in texts:
        new_text = []
        sents = nltk.tokenize.sent_tokenize(text)
        for sent in sents:
            words = set(map(lambda x: x.upper(), nltk.tokenize.wordpunct_tokenize(sent)))
            if len(ent_set.intersection(words)) > 3:
                new_text.append(sent)
        new_texts.append("\n".join(new_text))
        #new_texts.append(" ".join([]))
    return new_texts


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
    response = requests.get(url,headers=headers,timeout=5)
    html = response.content.encode("utf-8","ignore")
    return html

def get_images(url):
    result = haul.find_images(url)
    return map(lambda x: url + x if x.startswith("/") else x,result.image_urls)

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


def doNLP(text,likes,unlikes):
    print "doNLP"
    url = config["nlp_service"] + '/?properties={"annotators": "tokenize,ssplit,pos,ner,depparse,openie", "date": "2017-05-04T15:03:54"},"pipelineLanguage":"en""}'
    return_ents, best_return_rels = [],[]
    try:
        resp = requests.post(url,data=text)
        print "done."
        data = json.loads(resp.text)
        entities = []

        return_rels = []

        for sentence in data["sentences"]:
            last_ent = None
            for token in sentence["tokens"]:
                ner_type = token["ner"]
                if ner_type not in  ["O","NUMBER","DURATION","DATE","MONEY","ORDINAL","PERCENT","TIME"]:
                    entities.append((ner_type,token["word"],token["index"]))
            for rel in sentence["openie"]:
                return_rels.append({"id":"other"+hashlib.md5(" ".join([rel["subject"],rel["relation"],rel["object"]])).hexdigest(),"value":" ".join([rel["subject"],rel["relation"],rel["object"]])})
        print "done1"
        ent_list = []
        last_type = None
        last_index = 1
        for ent in entities:
            if last_type == ent[0] and last_index == ent[2]-1:
                ent_list[-1] = (" ".join([ent_list[-1][0],ent[1]]),ent[0])
            else:
                ent_list.append((ent[1],ent[0]))
            last_type = ent[0]
            last_index = ent[2]
        print "done2"
        ent_dict = {}
        entity_set = set()
        for ent in ent_list:
            # normaalizing text
            ent_txt = ent[0]
            label = ent[1]
            ent_txt = ' '.join(ent_txt.split()).upper().replace("\\n","").strip()
            ent_txt = ent_txt.split("'S")[0]
            ent_txt = ''.join([i for i in ent_txt if not i.isdigit()])

            ent_dict[(ent_txt,label)] = ent_dict.get((ent_txt,label),0)
            ent_dict[(ent_txt,label)] += 1
            entity_set.add(ent_txt)
        print "done3"

        best_return_rels = []
        for rel in return_rels:
            add = False
            for e in entity_set:
                if e in rel["value"].upper():
                    add = True
                    break
            if add:
                best_return_rels.append(rel)

        return_ents = [{"value":x[0],"type":x[1],"count":ent_dict[x],"id":"entity"+hashlib.md5(x[0] + "->" + x[1]).hexdigest()} for x in ent_dict \
        if "entity"+hashlib.md5(x[0] + "->" + x[1]).hexdigest() not in unlikes]
    except:
        print "Error"
    return return_ents, best_return_rels

def get_urls(terms,num_pages=1):
    results = []
    for term in terms:
        print "Searching for:" + term
        search_results = google.search(term, num_pages)
        results.extend([{"q":term,"url":x.link} for x in search_results])
    if not results:
        results = map(lambda x: {"q":terms[0],"url":x}, search2.do_search(terms[0],num_pages))
    return results

def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return  False

def build_social_json(name, url,ptype,screenshot_path):
    pid = "page" + hashlib.md5(url).hexdigest()
    data = {
        "name":name,
        "url":url,
        "screenshot_path":screenshot_path,
        "id":pid,
        "title":None,
        "profile":{
            "names":[],
            "emails":[],
            "phone_numbers":[],
            "addresses":[],
            "relationships":[],
            "social_media":[],
            "other":[],
            "images":[],
            "videos":[]
        },
        "entities":[],
        "type":ptype,
    }
    nes.index(index=config["butler_index"], doc_type="pages",body=data,id=pid)
    return data

def build_json(name,url,title,entities,addresses,ptype,rels,emails,phones,images,other,screenshot_path):
    pid = "page" + hashlib.md5(url).hexdigest()
    data = {
        "name":name,
        "url":url,
        "screenshot_path":screenshot_path,
        "id":pid,
        "title":title,
        "profile":{
            "names":[],
            "emails":emails,
            "phone_numbers":phones,
            "addresses":addresses,
            "relationships":rels,
            "social_media":[],
            "other":other,
            "images":images,
            "videos":[]
        },
        "entities":entities,
        "type":ptype, 
    }
    nes.index(index=config["butler_index"], doc_type="pages", body=data, id=pid)
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
            social_dict[e["id"]] = social_dict.get(e["id"],[e["url"],0,None,None])
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

    main_profile["other"] = sorted([{"id":x,"value":other_dict[x][0],"count":other_dict[x][1],"from":list(map(json.loads,other_dict[x][2])), "metadata":{"liked":x in likes, "unliked":x in unlikes}} for x in other_dict],key=lambda x: len(x["from"]),reverse=True)
    main_profile["phone_numbers"] = sorted([{"id":x,"value":phone_dict[x][0],"count":phone_dict[x][1],"from":list(map(json.loads,phone_dict[x][2])), "metadata":{"liked":x in likes, "unliked":x in unlikes}} for x in phone_dict],key=lambda x: len(x["from"]),reverse=True)
    main_profile["addresses"] = sorted([{"id":x,"value":address_dict[x][0],"count":address_dict[x][1],"from":list(map(json.loads,address_dict[x][2])), "metadata":{"liked":x in likes, "unliked":x in unlikes}} for x in address_dict],key=lambda x: len(x["from"]),reverse=True)
    main_profile["names"] = sorted([{"id":x,"value":names_dict[x][0],"count":names_dict[x][1],"from":list(map(json.loads, names_dict[x][2])), "metadata":{"liked":x in likes, "unliked":x in unlikes}} for x in names_dict],key=lambda x: len(x["from"]),reverse=True)[:3]
    main_profile["emails"] = sorted([{"id":x,"value":email_dict[x][0],"count":email_dict[x][1],"from":list(map(json.loads, email_dict[x][2])), "metadata":{"liked":x in likes, "unliked":x in unlikes}} for x in email_dict],key=lambda x: len(x["from"]),reverse=True)
    main_profile["relationships"] = sorted([{"id":x,"type":"connection","value":names_dict[x][0],"count":names_dict[x][1],"from":list(map(json.loads, names_dict[x][2])), "metadata":{"liked":x in likes, "unliked":x in unlikes}} for x in names_dict],key=lambda x: len(x["from"]),reverse=True)[3:]
    main_profile["social_media"] = sorted([{"id":x,"url":social_dict[x][0],"count":social_dict[x][1],"profile_url":social_dict[x][2],"username":social_dict[x][3], "metadata":{"liked":x in likes, "unliked":x in unlikes}} for x in social_dict],key=lambda x: x["count"],reverse=True)
    
    return main_profile


# Called when a name is given
@app.route('/name/', methods=['GET'])
def handle_name():
    name = request.args.get("name")
    print "GET: Name", name
    nes.index(index=config["butler_index"], doc_type="searches",body={"name":name},id=name)
    return resp

# Called when something is unliked
@app.route('/unlike/', methods=['GET'])
def handle_unlike():
    name = request.args.get("name")
    uid = request.args.get("id")
    print "GET: Unlike", name, uid
    nes.index(index=config["butler_index"], doc_type="unlikes",body={"name":name,"time":datetime.now().isoformat(),"id":uid})
    return resp

# Called when something is liked
@app.route('/like/', methods=['GET'])
def handle_like():
    name = request.args.get("name")
    lid = request.args.get("id")
    print "GET: Like", name, lid
    nes.index(index=config["butler_index"], doc_type="likes",body={"name":name,"time":datetime.now().isoformat(),"id":lid})
    return resp

# Called when clear is clicked
@app.route('/clear/', methods=['GET'])
def handle_clear():
    name = request.args.get("name")
    print "GET: Clear", name
    #qs = getQueries(name)
    #q, num_pages = qs[-1]
    #return_data = process_search([q],name,num_pages)
    #nes.index(index="butler", doc_type="results",body={"name":name,"query":q,"data":return_data},id=name)
    return resp

@app.route('/get_searches/',methods=['GET'])
def handle_get_searches():
    query = {
        "size": 0,
        "aggs" : {
            "searches" : {
                "terms" : { "field" : "name", "size":500 }
            }
    }}
    result = nes.search(index=config["butler_index"], doc_type="searches", body=query)
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

    results = nes.search(index=config["butler_index"], doc_type="results", body=query)
    return results

# Called when reload is clicked
@app.route('/reload/', methods=['GET'])
def handle_reload():    
    name = request.args.get("name")
    print "GET: Reload", name
    qs = getQueries(name)
    q, num_pages = qs[-1]
    likes,unlikes = getLikesUnlikes(name)
    new_searches = []
    results = do_reload(name)
    if len(results["hits"]["hits"]) >= 1:
        data = results["hits"]["hits"][-1]["_source"]["data"]
        for p in data["profile"]["names"] + data["profile"]["phone_numbers"] + data["profile"]["emails"] \
            + data["profile"]["addresses"] + data["profile"]["other"]:
            if p["id"] in likes:
                pass
                #new_searches.append(p["value"])
        for p in data["pages"]:
            for e in p["entities"]:
                if e["id"] in likes:
                    pass
                    #new_searches.append(e["value"])

    queries = [q] + new_searches
    print queries
    return_data = process_search(queries,name,num_pages)
    resp = Response(json.dumps(return_data,indent=2))
    nes.index(index=config["butler_index"], doc_type="results",body={"name":name,"query":q,"data":return_data},id=name)
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
                pass
                #new_searches.append(p["value"])
        for p in data["pages"]:
            for e in p["entities"]:
                if e["id"] in likes:
                    pass
                    #new_searches.append(e["value"])

    queries = [q] + new_searches
    print queries
    return_data = process_search(queries,name,num_pages)
    resp = Response(json.dumps(return_data,indent=2))
    nes.index(index=config["butler_index"], doc_type="results",body={"name":name,"query":q,"data":return_data},id=name)
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
    results = nes.search(index=config["butler_index"], doc_type=dtype, body=query)

    if len(results["hits"]["hits"]) == 1:
        return results["hits"]["hits"][0]["_source"]
    else:
        return None

def getScreenShot(url):
    ss_id = hashlib.md5(url).hexdigest() + ".png"
    os.system(config["chrome_loc"] + ' --headless --disable-gpu --no-sandbox --screenshot ' + url)
    os.system("mv screenshot.png ss/" + ss_id)
    return "/ss/" + ss_id

def getQueries(name):
    query = {
    "sort" : [
        { "time" : {"order" : "asc"}},
    ],
    "query": {
        "term": {
           "name": {
              "value": name
                   }
                }
            }
    }
    results = nes.search(index=config["butler_index"], doc_type="queries", body=query)

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
    likes = nes.search(index=config["butler_index"], doc_type="likes", body=query)
    unlikes = nes.search(index=config["butler_index"], doc_type="unlikes", body=query)

    return set(map(lambda x: x["_source"]["id"], likes["hits"]["hits"])), set(map(lambda x: x["_source"]["id"], unlikes["hits"]["hits"]))


# Called when twitter is scraped...
@app.route('/previous/', methods=['GET'])
def handle_previous():
    name = request.args.get("name")
    print "GET: Previous ->", name
    qs = getQueries(name)
    q, num_pages = qs[-1]
    print q, num_pages
    num_pages -= 1
    return_data = process_search([q],name,num_pages)
    resp = Response(json.dumps(return_data,indent=2))
    nes.index(index=config["butler_index"], doc_type="results",body={"name":name,"query":q,"data":return_data},id=name)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

# Called when twitter is scraped...
@app.route('/next/', methods=['GET'])
def handle_next():
    name = request.args.get("name")
    print "GET: Next ->", name
    qs = getQueries(name)
    q, num_pages = qs[-1]
    print q, num_pages
    num_pages += 1
    return_data = process_search([q],name,num_pages)
    resp = Response(json.dumps(return_data,indent=2))
    nes.index(index=config["butler_index"], doc_type="results",body={"name":name,"query":q,"data":return_data},id=name)
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


def mark_data(page,likes,unlikes):
    data_things = [
        "names",
        "emails",
        "phone_numbers",
        "addresses",
        "relationships",
        "other",
        "social_media"
    ]

    for entity in page["entities"]:
        entity["metadata"] = {"unliked":False, "liked":False}
        if entity["id"] in unlikes:
            entity["metadata"]["unliked"] = True
        if entity["id"] in likes:
            entity["metadata"]["liked"] = True

    for thing in data_things:
        for p in page["profile"][thing]:
            p["metadata"] = {"unliked":False, "liked":False}
            if p["id"] in unlikes:
                p["metadata"]["unliked"] = True
            if p["id"] in likes:
                p["metadata"]["liked"] = True

    return page

def process_search(q,name,num_pages=1):
    # Google results...
    has_text_results = False
    while not has_text_results:
        print "Getting %d more pages" % num_pages
        urls = get_urls(q,num_pages)
        global total_count

        likes, unlikes = getLikesUnlikes(name)

        for query in q:
            nes.index(index=config["butler_index"], doc_type="queries",body={"name":name,"query":query,"time":datetime.now().isoformat(), "num_pages":num_pages})

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
            all_text = ""
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
                print "Page already mined."
                if page["id"] in unlikes:
                    print "Page unliked."
                    continue
                page = mark_data(page,likes,unlikes)
                entries.append(page)
                text = getByURL(url,"texts",name)["text"]
                if text.strip() != "":
                    has_text_results = True
                texts.append(text)
                good_urls.append(url)
                all_entities.extend(page["entities"])
                continue

            if any(map(url.startswith,map(lambda x: x["urls"][0],social_mappings))):
                print "Social"
                screenshot_path = getScreenShot(url)
                data = build_social_json(name,url,"social",screenshot_path)
                all_entities.extend(data["entities"])
                social = True

            else:
                try:
                    html = get_html(url)
                    all_text,title = get_text_title(html)
                    text = get_readability_text(html)
                    text,title = get_text_title(text)
                    addresses = getAddresses(all_text,likes,unlikes)
                    screenshot_path = getScreenShot(url)
                    entities,other = doNLP(text,likes,unlikes)
                    print 'full done'
                    emails = get_emails(all_text,likes,unlikes)
                    phones = getPhoneNumbers(all_text,likes,unlikes)
                    if len(addresses) > 5 or len(emails) > 5 or len(phones) > 5:
                        print "Too many of something..."
                        continue
                    all_entities.extend(entities)
                    #images = get_images(url)
                except (UnicodeDecodeError,IOError,haul.exceptions.RetrieveError):
                    print "Error..."
                    continue
                if text == None or text.strip() == "":
                    print "No Text..."
                    continue
            texts.append(all_text)
            if all_text.strip() != "":
                has_text_results = True
            nes.index(index=config["butler_index"], doc_type="texts",body={"name":name,"query":query,"time":datetime.now().isoformat(),
                "url":url,"text":all_text,"main_text":text})

            #get_tables(url,i)
            
            good_urls.append(url)

            if not social:
                data = build_json(name,url,title,entities,addresses,"page",rels,emails,phones,images,other,screenshot_path)
                data = mark_data(data,likes,unlikes)
            entries.append(data)
        num_pages += 1

    doTexts = []
    for i, text in enumerate(texts):
        doTexts.append((text,i,good_urls[i]))

    total_count = len(doTexts)

    tree_stuff = doLDA(doTexts,0,None)

    populateEntries(entries,tree_stuff)

    profile = build_profile(entries,likes,unlikes)

    return_data = {"profile":profile,"pages":entries,"treemap":tree_stuff}

    print json.dumps(entries,indent=2)

    return return_data


@app.route('/ss/<path:path>')
def send_js(path):
    return send_from_directory('ss', path)

# Called when twitter is scraped...
@app.route('/search/', methods=['GET'])
def handle_search():
    q = request.args.get("q")
    name = request.args.get("name")
    num_pages = int(request.args.get("n",1))
    print "GET: Search ->", name, q, num_pages

    return_data = process_search([q],name,num_pages)

    resp = Response(json.dumps(return_data,indent=2))
    nes.index(index=config["butler_index"], doc_type="results",body={"name":name,"query":q,"data":return_data},id=name)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

if __name__ == "__main__":
    app.debug = True
    print "Running..."
    app.run(host="0.0.0.0",port=config["port"])
    

