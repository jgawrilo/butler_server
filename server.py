# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup
import re
import nltk
import sys
from readability.readability import Document
from selenium import webdriver
import pyap
import json
from table import HTMLTableParser
from nltk.corpus import stopwords
from flask import Flask, request, Response, send_from_directory
from gensim import corpora
from gensim.models.hdpmodel import HdpModel
from gensim.models import TfidfModel
from gensim.summarization import summarize
import haul
import random
from datetime import datetime
from elasticsearch import Elasticsearch
import hashlib
from fuzzywuzzy import fuzz
from google import google
import search2
import os
from multiprocessing import Process, Pool
import logging
import string

reload(sys)  
sys.setdefaultencoding('utf8')

config = json.load(open("config.json"))
nes = Elasticsearch([config["es"]])

app = Flask(__name__)
app.config['SECRET_KEY'] = 'kyc_butler'
app.debug = True
LOGGING_FORMAT = '%(asctime)s - Line %(lineno)d %(levelname)s %(message)s'
LOGGING_LOCATION = 'butler_server.log'
LOGGING_LEVEL = logging.INFO
handler = logging.FileHandler(LOGGING_LOCATION)
handler.setLevel(LOGGING_LEVEL)
formatter = logging.Formatter(LOGGING_FORMAT)
handler.setFormatter(formatter)
app.logger.addHandler(handler)

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

def getPhoneNumbers(text,likes,unlikes,url):
    phones = []
    try:

        for match in re.finditer(r"\(?\b[2-9][0-9]{2}\)?[-. ]?[2-9][0-9]{2}[-. ]?[0-9]{4}\b", text):
            match = match.group()
            pid = "phone" + hashlib.md5(match).hexdigest()
            if pid not in unlikes:
                phones.append({"id":pid,"value":match})
    except Error:
        app.logger.error("Couldn't get phones -> " +url)

    return phones

def get_emails(text,likes,unlikes,url):
    ret = []
    try:
        for email in (email[0] for email in re.findall(regex, text) if not email[0].startswith('//')):
            eid = "email"+hashlib.md5(email).hexdigest()
            if eid not in unlikes:
                ret.append({"id":eid,"value":email})
    except Error:
        app.logger.error("Couldn't get emails -> " +url)
    return ret

def doLDA(docu,level,last_count,top_text):
    app.logger.info("** LDA ** " + " ".join((str(len(docu)),str(level),str(last_count))))
    # remove common words and tokenize
    slots, documents, urls, titles, summaries = [],[],[],[],[]

    full_results = []

    empty_slots, empty_urls,empty_titles, empty_summaries = [],[],[],[]

    empty_results = []

    for x in docu:
        if x[0]:
            documents.append(x[0])
            slots.append(x[1])
            urls.append(x[2])
            titles.append(x[3])
            summaries.append(x[4])
            full_results.append(x[5])
        else:
            empty_slots.append(x[1])
            empty_urls.append(x[2])
            empty_titles.append(x[3])
            empty_summaries.append(x[4])
            empty_results.append(x[5])

    stoplist = set(stopwords.words('english'))
    punctuation = set(string.punctuation)
    texts = [[word.lower() for word in document if word.lower() not in stoplist.union(punctuation) and not is_float(word)]
             for document in documents]


    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    tfidf = TfidfModel(corpus)
    hdp = HdpModel(tfidf[corpus],dictionary,random_state=7)
    lda = hdp.suggested_lda_model()

    topic_answers_dict = {"count":len(docu),"summary":None,"title":top_text,"url":None,"level":level,"children":[],"node_id":last_count,"type":"cluster"}
    app.logger.info("AGG Level object -> " + json.dumps(topic_answers_dict))
    
    if empty_slots:
        app.logger.info("%d empty pages heading to a single topic" % len(empty_slots))
        last_count += 1
        topic_answers_dict["children"].append({"count":len(empty_slots),"summary":None,"title":"No Text","url":None,"node_id":last_count,"level":1,"children":[],"type":"cluster"})
        for i,x in enumerate(empty_slots):
            topic_answers_dict["children"][-1]["children"].append({"type":empty_results[i]["type"],"count":1,"summary":"","title":empty_titles[i],"url":empty_urls[i],"node_id":empty_slots[i],"level":2,"children":[],"scores":[]})

    result_dict = {}
    app.logger.info("%d pages with text" % len(documents))
    for i in range(len(documents)):
        entries = sorted(lda[corpus[i]],key=lambda x: x[1],reverse=True)
        if entries:
            d = lda.show_topics(-1,formatted=False)[entries[0][0]]
            ret = {"result":full_results[i],"text":documents[i],"summary":summaries[i],"url":urls[i],"title":titles[i],"row":i, "slot":slots[i], "number":d[0],"scores":map(lambda x:{"value":x[0].replace("\"",""),"score":x[1]},d[1])[:5]}
            result_dict[d[0]] = result_dict.get(d[0],[])
            result_dict[d[0]].append(ret)
        else:
            app.logger.info("issue...")
    for k in result_dict:
        app.logger.info("Topic %d has %d pages" % (k,len(result_dict[k])))
        if len(documents) == len(result_dict[k]):
            app.logger.info("Every document was in a single topic.")
            for res in result_dict[k]:
                topic_answers_dict["children"].append({"type":res["result"]["type"],"count":1,"url":res["url"],"summary":res["summary"],"title":res["title"],"node_id":slots[res["row"]],"level":level+1,"children":[],"scores":res["scores"]})
        elif len(result_dict[k]) > 1:
            app.logger.info("Found a topic with more than 1 page.  Time to run again with...")
            docs = [(x["text"],x["slot"],x["url"],x["title"],x["summary"],x["result"]) for x in result_dict[k]]
            app.logger.info(" ".join(map(lambda x: x[2],docs)))
            topic_answers_dict["children"].append(doLDA(docs,level+1,last_count+1," ".join([x["value"] for x in result_dict[k][0]["scores"]])))
        else:
            app.logger.info("Topic has a single page...")
            app.logger.info(result_dict[k][0]["url"])
            app.logger.info(result_dict[k][0]["scores"])
            topic_answers_dict["children"].append({"type":result_dict[k][0]["result"]["type"],"count":1,"summary":result_dict[k][0]["summary"],"title":result_dict[k][0]["title"],"url":result_dict[k][0]["url"],"node_id":slots[result_dict[k][0]["row"]],"level":level+1,"children":[],"scores":result_dict[k][0]["scores"]})

    return topic_answers_dict

def getAddresses(text,likes,unlikes):
    try:
        addresses = pyap.parse(text, country='US')
        addresses = map(lambda x: " ".join(str(x).upper().split()),addresses)
        return filter(lambda x: x["id"] not in unlikes, map(lambda x: {"id":"address"+hashlib.md5(x).hexdigest(),"value":x},addresses))
    except Error:
        app.logger.error("Couldn't get addresses.")

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
    try:
        ua = random.choice(user_agents)  # select a random user agent
        headers = {
            "Connection" : "close",  # another way to cover tracks
            "User-Agent" : ua
        }
        response = requests.get(url,headers=headers,timeout=5)
        html = response.content.encode("utf-8","ignore")
        return html
    except:
        app.logger.error("Error getting HTML for -> " + url)
        raise

def get_images(url):
    result = haul.find_images(url)
    return map(lambda x: url + x if x.startswith("/") else x,result.image_urls)

def get_text_title(html,url):
    text, title = "",""
    try:
        soup = BeautifulSoup(html, "lxml")
        data = soup.findAll(text=True)
        result = filter(visible, data)
        text = " ".join(result).encode("utf-8","ignore")
        title = ""
        if soup.title != None:
            title = soup.title.string
        return text.strip(), title.strip()
    except:
        app.logger.error("Couldn't get text and title for -> " + url)
        return text,title


def get_readability_text(html,url):
    try:
        readable_article = Document(html).summary()
        readable_title = Document(html).short_title()
        return readable_article.strip()
    except:
        app.logger.error("Couldn't get readability text for -> " + url)
        return ""


def doNLP(text,likes,unlikes,the_url):
    try:
        app.logger.info("Starting NLP on -> " + the_url)

        url = config["nlp_service"] + '/?properties={"annotators": "tokenize,ssplit,pos,ner,depparse,openie"}'
        return_ents, best_return_rels, return_tokens, return_rels = [],[],[],[]
        resp = requests.post(url,data=text,timeout=60)
        app.logger.info("Finished NLP on -> " + the_url)

        data = json.loads(resp.text)

        entities = []
        return_tokens = []

        for sentence in data["sentences"]:
            for token in sentence["tokens"]:
                return_tokens.append(token["lemma"])
                if token["ner"] not in  ["O","NUMBER","DURATION","DATE","MONEY","ORDINAL","PERCENT","TIME"]:
                    entities.append((token["ner"],token["word"],token["index"]))

            for rel in sentence["openie"]:
                return_rels.append({"id":"other"+hashlib.md5(" ".join([rel["subject"],rel["relation"],rel["object"]])).hexdigest(),"value":" ".join([rel["subject"],rel["relation"],rel["object"]])})
        
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
        #except:
        #    app.logger.error("Error duing NLP on -> " + the_url)

        return return_ents, best_return_rels, return_tokens
    except:
        app.logger.error("NLP not working -> " + the_url)
        raise Error


def get_urls(terms,num_pages=1):
    """
        get results from google for search terms
    """
    results = []
    # First try google search API
    for term in terms:
        app.logger.info("Using API to search for -> " + term)
        search_results = google.search(term, num_pages)
        results.extend([{"q":term,"url":x.link} for x in search_results])

    # If it's not working, we might be blocked. Get results through browser
    if not results:
        app.logger.warn("Didn't get any results.  Trying browser to search ->" + term)
        results = map(lambda x: {"q":terms[0],"url":x}, search2.do_search(terms[0],num_pages))
    return results

def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return  False

def build_social_json(name, url,ptype,screenshot_path):
    pid = "page_" + hashlib.md5(url).hexdigest()
    data = {
        "name":name,
        "url":url,
        "screenshot_path":screenshot_path,
        "id":pid,
        "title":None,
        "summary":None,
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

def build_json(name,url,title,entities,addresses,ptype,rels,emails,phones,images,other,screenshot_path,summary):
    pid = "page_" + hashlib.md5(url).hexdigest()
    data = {
        "name":name,
        "url":url,
        "screenshot_path":screenshot_path,
        "id":pid,
        "title":title,
        "summary":summary,
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
                for test in e_set:
                    if test != n["value"] and fuzz.ratio(test, n["value"]) >= 64:
                        pass



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
    app.logger.info("*** Name -> " + name)
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
    if not qs:
        resp = Response(json.dumps({"success":True,"message":"Please start a search."}))
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp
    q, num_pages, language = qs[-1]
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
    return_data = new_process(queries,name,num_pages,language)
    resp = Response(json.dumps(return_data,indent=2))
    nes.index(index=config["butler_index"], doc_type="results",body={"name":name,"query":q,"data":return_data,"language":language},id=name)
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
    q, num_pages,language = qs[-1]
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
    return_data = new_process(queries,name,num_pages,language)
    resp = Response(json.dumps(return_data,indent=2))
    nes.index(index=config["butler_index"], doc_type="results",body={"name":name,"query":q,"data":return_data,"language":language},id=name)
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
    try:
        ss_id = hashlib.md5(url).hexdigest() + ".png"
        os.system(config["chrome_loc"] + ' --headless --disable-gpu --no-sandbox --screenshot ' + url + " && mv screenshot.png ss/" + ss_id)
        ss_return = "/ss/" + ss_id
        app.logger.info("Screenshot created -> " + url + " " + ss_return)
        return ss_return
    except:
        app.logger.error("Screenshot not working -> " + url + " " + ss_return)

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
        return map(lambda x:(x["_source"]["query"],x["_source"]["num_pages"],x["_source"]["language"]),results["hits"]["hits"])
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
    q, num_pages,language = qs[-1]
    print q, num_pages, language
    num_pages -= 1
    if num_pages == 0:
        num_pages = 1
    return_data = new_process([q],name,num_pages,language)
    resp = Response(json.dumps(return_data,indent=2))
    nes.index(index=config["butler_index"], doc_type="results",body={"name":name,"query":q,"data":return_data,"language":language},id=name)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

# Called when twitter is scraped...
@app.route('/next/', methods=['GET'])
def handle_next():
    name = request.args.get("name")
    print "GET: Next ->", name
    qs = getQueries(name)
    q, num_pages, language = qs[-1]
    print q, num_pages
    num_pages += 1
    return_data = new_process([q],name,num_pages,language)
    resp = Response(json.dumps(return_data,indent=2))
    nes.index(index=config["butler_index"], doc_type="results",body={"name":name,"query":q,"data":return_data,"language":language},id=name)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

def populateEntries(entries,tree_stuff):
    if "scores" in tree_stuff:
        entries[tree_stuff["node_id"]]["topic"] = { \
            "scores":tree_stuff["scores"],\
            "number":"",\
            "string":" ".join([x["value"] for x in tree_stuff["scores"]])\
        }
    else:
        for child in tree_stuff["children"]:
            populateEntries(entries,child) 


def mark_data(page,likes,unlikes):
    """
        Check every item in page and populate metadata.
        This step could get annoying but leave for now
    """

    page["metadata"] = {"unliked":False, "liked":False}
    if page["id"] in unlikes:
        page["metadata"]["unliked"] = True
    if page["id"] in likes:
        page["metadata"]["liked"] = True

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

def process_single_page(in_data):
    # Split tuple
    url_obj, page_and_text_and_token, name, likes, unlikes, bad_urls, language = in_data
    url = url_obj["url"]
    query = url_obj["q"]

    app.logger.info("** Processing -> " + url)

    # Filter Line.  Put more here.
    if url.endswith(".pdf") or any(map(url.startswith,stop)) or url in bad_urls:
        app.logger.warn("Skipping.  PDF or in STOP_LIST or in BAD_URLS.")
        return ()

    html = ""
    page = page_and_text_and_token[0]
    text = page_and_text_and_token[1]
    all_text = ""
    readable_text = ""
    title = ""
    addresses = []
    rels = []
    emails  = []
    phones = []
    images = []
    social = False
    entities = []
    tokens = page_and_text_and_token[2]
    summary = ""
    
    # If passed in page, we already processed it.
    if page:
        app.logger.info("Page already mined.")
        # If user unliked page, don't process.  Probably should bash other results as well
        if page["id"] in unlikes:
            app.logger.info("Page previously unliked.")
            return ()

        # Mark page up with like/unlike metadata
        data = mark_data(page,likes,unlikes)
        return (data, text, url, page["entities"], tokens)

    # If we get here, it means new page

    # If page is social media
    if any(map(url.startswith,map(lambda x: x["urls"][0],social_mappings))):
        app.logger.info("Page is new.  Page is social media.")

        #screenshot_path = "ss"
        screenshot_path = getScreenShot(url)
        data = build_social_json(name,url,"social",screenshot_path)
        data = mark_data(data,likes,unlikes)

    # Page is NOT social media and NOT alredy mined.
    else:
        try:
            app.logger.info("Page is new.")
            html = get_html(url)
            all_text,title = get_text_title(html,url)
            readable_text = get_readability_text(html,url)
            summary, _ = get_text_title(readable_text,url)
            summary = " ".join(summary[:500].split())
            text = all_text
            addresses = getAddresses(all_text,likes,unlikes)
            #screenshot_path = "ss"
            screenshot_path = getScreenShot(url)
            entities,other,tokens = doNLP(text,likes,unlikes,url)
            emails = get_emails(all_text,likes,unlikes,url)
            phones = getPhoneNumbers(all_text,likes,unlikes,url)

            if len(addresses) > 5 or len(emails) > 5 or len(phones) > 5:
                app.logger.warn("Too many of something found.  Adding bad URL -> " + url)
                nes.index(index=config["butler_index"], doc_type="bad_urls",body={"name":name,"query":query,"url":url})
                return ()

            data = build_json(name,url,title,entities,addresses,"page",rels,emails,phones,images,other,screenshot_path,summary)
            data = mark_data(data,likes,unlikes)
        except:
            app.logger.warn("Error occurred during processing.  Adding bad URL -> " + url)
            nes.index(index=config["butler_index"], doc_type="bad_urls",body={"name":name,"query":query,"url":url})
            return ()

    nes.index(index=config["butler_index"], doc_type="texts",body={"name":name,"query":query,"time":datetime.now().isoformat(),
        "language":language,"url":url,"text":all_text,"main_text":text,"title":title,"tokens":tokens})

    #get_tables(url,i)
    return (data, text, url, entities, tokens)

def getBadURLS(name):
    query = {
        "query": {
            "term": {
               "name": {
                  "value": name
                       }
                    }
                }
    }
    urls = nes.search(index=config["butler_index"], doc_type="bad_urls", body=query)
    return set(map(lambda x: x["_source"]["url"], urls["hits"]["hits"]))

def new_process(q,name,num_pages=1,language="english"):
    """
        Start of search process
    """

    app.logger.info("** Starting new process ** ")

    # Get likes and unlikes
    likes, unlikes = getLikesUnlikes(name)
    app.logger.info("Found %d likes and %d unlikes." % (len(likes), len(unlikes)))

    # Need to grab what the user was currently looking at
    last_results = do_reload(name)
    liked_urls = []

    if len(last_results["hits"]["hits"]) >= 1:
        data = last_results["hits"]["hits"][-1]["_source"]["data"]
        for p in data["pages"]:
            if p["id"] in likes:
                liked_urls.append({"url":p["url"],"q":data["meta"]["q"]})

    app.logger.info("%d liked urls we're going to keep" % (len(liked_urls)))

    
    app.logger.info("Getting %d google pages" % num_pages)

    # mine urls
    urls = get_urls(q,num_pages)
    app.logger.info(str(len(urls)) + " urls found.")

    urls = urls + liked_urls

    

    # Get bad urls so we can skip
    bad_urls = getBadURLS(name)
    app.logger.info("Found %d bad urls." % len(bad_urls))

    # Index each query associated with the project (should be one for now)
    for query in q:
        nes.index(index=config["butler_index"], doc_type="queries",body={"name":name,"query":query,"time":datetime.now().isoformat(), "num_pages":num_pages, "language":language})

    # For the URLS we got back, check to see if we have them already and store info if we do
    pages_and_texts_and_tokens = []
    for url in urls:
        text = ""
        tokens = []
        page = getByURL(url["url"],"pages",name)
        if page:
            data = getByURL(url["url"],"texts",name)
            text = data["text"]
            tokens = data["tokens"]
        pages_and_texts_and_tokens.append((page,text,tokens))


    # How many threads to process with        
    pool = Pool(processes=2)

    app.logger.info("Processing %d urls" % len(urls))

    # Do the thing
    results = pool.map(process_single_page, map(lambda x:(x[0],x[1],name,likes,unlikes,bad_urls,language),zip(urls,pages_and_texts_and_tokens)))

    app.logger.info("Finished %d urls" % len(results))

    pool.close()
    
    # Filter out results from urls with errors
    results = filter(lambda x: len(x) > 0,results)

    app.logger.info("%d urls after filtering" % len(results))

    # Get texts, urls, and entry objects
    texts = map(lambda x: x[1],results)
    good_urls = map(lambda x: x[2],results)
    entries = map(lambda x: x[0],results)
    tokens = map(lambda x: x[4],results)
    titles = map(lambda x: x[0]["title"],results)
    summaries = map(lambda x: x[0]["summary"],results)

    doTexts = []
    for i, token_guy in enumerate(tokens):
        doTexts.append((token_guy,i,good_urls[i],titles[i],summaries[i],results[i][0]))

    app.logger.info("Running LDA with %d pages" % len(doTexts))

    tree_stuff = doLDA(doTexts,0,len(doTexts),None)

    app.logger.info("Done running LDA")

    populateEntries(entries,tree_stuff)

    app.logger.info("Building Profile")

    profile = build_profile(entries,likes,unlikes)

    app.logger.info("Profile Built")

    meta = {"name":name,"q":q,"num_pages":len(doTexts),"language":language}

    return_data = {"profile":profile,"pages":entries,"treemap":tree_stuff,"meta":meta}

    app.logger.info("Returning Results.")

    return return_data


@app.route('/ss/<path:path>')
def send_js(path):
    return send_from_directory('ss', path)

# Called when twitter is scraped...
@app.route('/search/', methods=['GET'])
def handle_search():
    """
        Happens when user submits search
    """
    q = request.args.get("q")
    name = request.args.get("name")
    language = request.args.get("language")
    num_pages = int(request.args.get("n",1))

    app.logger.info("*** Search -> " +  " ".join((map(str,[name, q, num_pages, language]))))

    #return_data = process_search([q],name,num_pages,language)
    return_data = new_process([q],name,num_pages,language)
    nes.index(index=config["butler_index"], doc_type="results",body={"name":name,"query":q,"data":return_data,"language":language},id=name)

    resp = Response(json.dumps(return_data,indent=2))
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

if __name__ == "__main__":
    app.debug = True
    print "Running..."
    app.run(host="0.0.0.0",port=config["port"])

