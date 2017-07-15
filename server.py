# -*- coding: utf-8 -*-
import traceback
import requests
from bs4 import BeautifulSoup
from collections import defaultdict
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
from elasticsearch import Elasticsearch, NotFoundError
import hashlib
from fuzzywuzzy.process import extractBests
from fuzzywuzzy.process import dedupe as fuzzy_dedupe
from google import google
import search2
import os
from multiprocessing import Process, Pool
import logging
import string
import custom_extractors
import langdetect
import pandas as pd
from sri_service import star_search
from collections import Counter
import cdr_search

import subprocess, threading

class Command(object):
    def __init__(self, cmd):
        self.cmd = cmd
        self.process = None

    def run(self, timeout):
        def target():
            self.process = subprocess.Popen(self.cmd, shell=True)
            self.process.communicate()

        thread = threading.Thread(target=target)
        thread.start()

        thread.join(timeout)
        if thread.is_alive():
            self.process.terminate()
            thread.join()
        return self.process.returncode

reload(sys)  
sys.setdefaultencoding('utf8')

config = json.load(open("config.json"))
nes = Elasticsearch([config["es"]],verify_certs=False)

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

custom_extractors_list = [
    {"site":"Crunchbase","url_starts_with":"https://www.crunchbase.com/organization/"},
    {"site":"Intelius","url_starts_with":"https://www.intelius.com/people/"}
]

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
    except:
        app.logger.error("Couldn't get phones -> " +url)

    return phones

def get_emails(text,likes,unlikes,url):
    ret = []
    try:
        for email in (email[0] for email in re.findall(regex, text) if not email[0].startswith('//')):
            eid = "email"+hashlib.md5(email).hexdigest()
            if eid not in unlikes:
                ret.append({"id":eid,"value":email})
    except:
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

    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1
    texts = [[token for token in text if frequency[token] > 1] for text in texts]

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    #tfidf = TfidfModel(corpus)
    hdp = HdpModel(corpus,dictionary,random_state=7)
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
            last_count += 1
            topic_answers_dict["children"].append(doLDA(docs,level+1,last_count," ".join([x["value"] for x in result_dict[k][0]["scores"]])))
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
    except:
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


def get_table_rels(url,html):
    app.logger.info("Tables are being called: " + url)
    rels = []
    try:
        tables = pd.read_html(html)
        for table in tables:
            app.logger.info("{} tables found. ".format(len(table.columns)) + url)
            if len(table) > 50:
                continue
            if len(table.columns) == 2:
                for i in range(len(table)):
                    rels.append({"subject":str(table.ix[i][0]), "object":str(table.ix[i][1]), "id":"other"+hashlib.md5(" ".join([str(table.ix[i][0]),":",str(table.ix[i][1])])).hexdigest(),"value":" ".join([str(table.ix[i][1])]),"type":str(table.ix[i][0])})
            else:
                pass
                # each column is a relationship
        return rels
    except:
        app.logger.error("Error with tables: " + url)
        return rels

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
        return ""

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
        resp = requests.post(url,data=text,timeout=15)
        data = json.loads(resp.text)
        app.logger.info("Finished NLP Call on -> " + the_url)

        entities = []
        return_tokens = []

        for sentence in data["sentences"]:
            for token in sentence["tokens"]:
                return_tokens.append(token["lemma"])
                if token["ner"] not in  ["O","NUMBER","DURATION","DATE","MONEY","ORDINAL","PERCENT","TIME"]:
                    entities.append((token["ner"],token["word"],token["index"]))

            for rel in sentence["openie"]:
                return_rels.append({"subject":rel["subject"], "object":rel["object"], "id":"other"+hashlib.md5(" ".join([rel["subject"],rel["relation"],rel["object"]])).hexdigest(),"value":" ".join([rel["subject"],rel["relation"],rel["object"]]),"type":""})
        
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
            if rel["subject"].upper() in entity_set and rel["object"].upper() in entity_set:
                best_return_rels.append(rel)

        deduped = fuzzy_dedupe(map(lambda x: x[0],ent_dict))

        return_ents = [{"value":extractBests(x[0],deduped)[0][0].upper(),"type":x[1],"count":ent_dict[x],"id":"entity"+hashlib.md5(extractBests(x[0],deduped)[0][0].upper() + "->" + x[1]).hexdigest()} for x in ent_dict \
        if "entity"+hashlib.md5(extractBests(x[0],deduped)[0][0].upper() + "->" + x[1]).hexdigest() not in unlikes]

        app.logger.info("Finished NLP Return on -> " + the_url)
        return return_ents, best_return_rels, return_tokens
    except:
        app.logger.error("NLP not working -> " + the_url)
        return [],[],[] 


def get_urls(terms,name):
    """
        get results from google for search terms
    """
    res_set = set()
    results = []
    # First try google search API
    for term in terms:
        app.logger.info(name + " Using API to search for -> " + term["query"])
        search_results = google.search(term["query"], term["num_pages"])
        for x in search_results:
            if x.link not in res_set:
                results.append({"q":term["query"],"url":x.link,"language":term["language"]})
                res_set.add(x.link)

    # If it's not working, we might be blocked. Get results through browser
    if not results:
        app.logger.warn(name + " Didn't get any results.  Trying browser to search")
        results = search2.do_search(terms)

    if not results:
        app.logger.warn(name + " Didn't get any results.  Trying servers")
        for box in config["search_boxes"]:
            results = json.loads(requests.post(box,json=terms).text)
            if results:
                return results

    return results

def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return  False

def build_social_json(name, url,ptype,screenshot_path):
    html = get_html(url)
    all_text,title = get_text_title(html,url)
    pid = "page_" + hashlib.md5(url).hexdigest()
    data = {
        "name":name,
        "url":url,
        "screenshot_path":screenshot_path,
        "lang":"",
        "id":pid,
        "title":title,
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

def build_json(name,url,title,entities,addresses,ptype,rels,emails,phones,images,other,screenshot_path,summary,lang):
    pid = "page_" + hashlib.md5(url).hexdigest()
    data = {
        "name":name,
        "url":url,
        "lang":lang,
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

def getUserName(url):
    try:
        if url.startswith("https://github.com/"):
            return url.split("https://github.com/")[1].split("/")[0].strip()
        elif url.startswith("https://plus.google.com/"):
            return url.split("https://plus.google.com/")[1].split("/")[0].strip()
        elif url.startswith("https://www.linkedin.com/"):
            return url.split("https://www.linkedin.com/in/")[1].split("/")[0].strip()
        elif url.startswith("https://www.instagram.com/"):
            return url.split("https://www.instagram.com/")[1].split("/")[0].strip()
        elif url.startswith("https://www.pinterest.com/"):
            return url.split("https://www.pinterest.com/")[1].split("/")[0].strip()
        elif url.startswith("https://www.facebook.com/"):
            return url.split("https://www.facebook.com/public/")[1].split("/")[0].strip()
        elif url.startswith("https://twitter.com/"):
            return url.split("https://twitter.com/")[1].split("/")[0].split("?")[0].strip()
        elif url.startswith("https://www.youtube.com/"):
            return url.split("https://www.youtube.com/channel/")[1].split("/")[0].strip()
    except:
        return ""

def add_things(data,phone_dict,address_dict,names_dict,email_dict,social_dict,other_dict):
    philds = {
        "email_addr":("emails",email_dict,False),
        "phone_num":("phones",phone_dict,False),
        "bitcoin_addr":("bitcoins",other_dict,True),
        "pgp_key_ls":("pgps",other_dict,True),
        "pgp_key_hash": ("pgp hashes",other_dict,True),
        "org":("organizations",other_dict,True),
        "person_name":("people",names_dict,False),
        "gpe":("gpes",other_dict,True),
        "pgp_email_addr":("pgp emails",email_dict,False),
        "ssn_num":("ssns",other_dict,True),
        "onion_appearance":("onions",other_dict,True)
    }

    for dd in data:
        for field in philds.values():
            data_name = field[0]
            data_dict = field[1]
            is_other = field[2]

            for de in dd[data_name]:
                if is_other:
                    data_dict[de["id"]] = data_dict.get(de["id"],[de["value"],0,set(),data_name])
                    data_dict[de["id"]][1] += 1
                    data_dict[de["id"]][2].add(json.dumps({"id":"no_page","url":"Dark Web Persona Mapper"}))
                else:
                    data_dict[de["id"]] = data_dict.get(de["id"],[de["value"],0,set()])
                    data_dict[de["id"]][1] += 1
                    data_dict[de["id"]][2].add(json.dumps({"id":"no_page","url":"Dark Web Persona Mapper"}))

def build_profile(entries,likes,unlikes):
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

    """
    all_names = {}
    for e in entries:
        for n in e["entities"]:
            if n["type"] == "PERSON":
                all_names[n["value"]] = all_names.get(n["value"],[n["id"],0])
                all_names[n["value"]][1] += n["count"]
    deduped_names = fuzzy_dedupe(all_names.keys())
    """

    if config["star_search"]:
        pool = Pool(processes=config["page_threads"])
        #### EMAILS ####
        ez = map(lambda x: x["profile"]["emails"], entries)
        ez = filter(lambda x: x, ez)
        ones_to_check = []
        for e1 in ez:
            for one in e1:
                ones_to_check.append(one)
        app.logger.info(ones_to_check)
        emails = map(lambda x: {"Email":x["value"],"RegistrationKey": "MyDogAteMyKey", "Action": "analyze"}, ones_to_check)
        app.logger.info("STAR SEARCH: ")
        app.logger.info(emails)
        ds_results = []
        if emails:
            ds_results = pool.map(star_search,emails)
            pool.close()
        app.logger.info("STAR SEARCH RESULTS:")
        app.logger.info(ds_results)
        add_things(ds_results,phone_dict,address_dict,names_dict,email_dict,social_dict,other_dict)

        ### PHONES ####
        pool = Pool(processes=config["page_threads"])
        ez = map(lambda x: x["profile"]["phone_numbers"], entries)
        ez = filter(lambda x: x, ez)
        ones_to_check = []
        for e1 in ez:
            for one in e1:
                ones_to_check.append(one)
        app.logger.info(ones_to_check)
        pgs = map(lambda x: {"Phone":x["value"],"RegistrationKey": "MyDogAteMyKey", "Action": "analyze"}, ones_to_check)
        app.logger.info("STAR SEARCH: ")
        app.logger.info(pgs)
        ds_results = []
        if pgs:
            ds_results = pool.map(star_search,pgs)
            pool.close()
        app.logger.info("STAR SEARCH RESULTS:")
        app.logger.info(ds_results)
        add_things(ds_results,phone_dict,address_dict,names_dict,email_dict,social_dict,other_dict)

        #### PGP_EMAIL ####
        pool = Pool(processes=config["page_threads"])
        ez = map(lambda x: x["profile"]["emails"], entries)
        ez = filter(lambda x: x, ez)
        ones_to_check = []
        for e1 in ez:
            for one in e1:
                ones_to_check.append(one)
        app.logger.info(ones_to_check)
        emails = map(lambda x: {"PGP_EMAIL":x["value"],"RegistrationKey": "MyDogAteMyKey", "Action": "analyze"}, ones_to_check)
        app.logger.info("STAR SEARCH: ")
        app.logger.info(emails)
        ds_results = []
        if emails:
            ds_results = pool.map(star_search,emails)
            pool.close()
        app.logger.info("STAR SEARCH RESULTS:")
        app.logger.info(ds_results)
        add_things(ds_results,phone_dict,address_dict,names_dict,email_dict,social_dict,other_dict)

        #### PEOPLE ####
        pool = Pool(processes=config["page_threads"])
        ez = map(lambda x: x["entities"], entries)
        ez = filter(lambda x: x, ez)
        ones_to_check = []
        for e1 in ez:
            for one in e1:
                if one["type"] == "PERSON":
                    ones_to_check.append(one["value"])

        gos = []
        for ez in Counter(ones_to_check).most_common()[:3]:
            gos.append(ez[0])

        app.logger.info(gos)
        emails = map(lambda x: {"PersonName":x,"RegistrationKey": "MyDogAteMyKey", "Action": "analyze"}, gos)
        app.logger.info("STAR SEARCH: ")
        app.logger.info(emails)
        ds_results = []
        if emails:
            ds_results = pool.map(star_search,emails)
            pool.close()
        app.logger.info("STAR SEARCH RESULTS:")
        app.logger.info(ds_results)
        add_things(ds_results,phone_dict,address_dict,names_dict,email_dict,social_dict,other_dict)

    for e in entries:
        if e["type"] == "social":
            social_dict[e["id"]] = social_dict.get(e["id"],[e["url"],0,None,None])
            social_dict[e["id"]][1] += 1
            social_dict[e["id"]][3] = getUserName(e["url"])
            continue
        for n in e["profile"]["other"]:
            other_dict[n["id"]] = other_dict.get(n["id"],[n["value"],0,set(),n["type"]])
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
                names_dict[n["id"]][1] += 1 
                names_dict[n["id"]][2].add(json.dumps({"id":e["id"],"url":e["url"]}))
                #best_val = extractBests(n["value"],deduped_names)[0][0]
                #best_id = all_names[best_val][0]
                #names_dict[best_id] = names_dict.get(best_id,[best_val,0,set()])
                #names_dict[best_id][1] = all_names[best_val][1]
                #names_dict[best_id][2].add(json.dumps({"id":e["id"],"url":e["url"]}))

    main_profile["other"] = sorted([{"id":x,"type":other_dict[x][3],"value":other_dict[x][0],"count":other_dict[x][1],"from":list(map(json.loads,other_dict[x][2])), "metadata":{"liked":x in likes, "unliked":x in unlikes}} for x in other_dict],key=lambda x: len(x["from"]),reverse=True)
    main_profile["phone_numbers"] = sorted([{"id":x,"value":phone_dict[x][0],"count":phone_dict[x][1],"from":list(map(json.loads,phone_dict[x][2])), "metadata":{"liked":x in likes, "unliked":x in unlikes}} for x in phone_dict],key=lambda x: len(x["from"]),reverse=True)
    main_profile["addresses"] = sorted([{"id":x,"value":address_dict[x][0],"count":address_dict[x][1],"from":list(map(json.loads,address_dict[x][2])), "metadata":{"liked":x in likes, "unliked":x in unlikes}} for x in address_dict],key=lambda x: len(x["from"]),reverse=True)
    main_profile["names"] = sorted([{"id":x,"value":names_dict[x][0],"count":names_dict[x][1],"from":list(map(json.loads, names_dict[x][2])), "metadata":{"liked":x in likes, "unliked":x in unlikes}} for x in names_dict],key=lambda x: len(x["from"]),reverse=True)[:3]
    main_profile["emails"] = sorted([{"id":x,"value":email_dict[x][0],"count":email_dict[x][1],"from":list(map(json.loads, email_dict[x][2])), "metadata":{"liked":x in likes, "unliked":x in unlikes}} for x in email_dict],key=lambda x: len(x["from"]),reverse=True)
    main_profile["relationships"] = sorted([{"id":x,"type":"","value":names_dict[x][0],"count":names_dict[x][1],"from":list(map(json.loads, names_dict[x][2])), "metadata":{"liked":x in likes, "unliked":x in unlikes}} for x in names_dict],key=lambda x: len(x["from"]),reverse=True)[3:]
    main_profile["social_media"] = sorted([{"id":x,"url":social_dict[x][0],"count":social_dict[x][1],"profile_url":social_dict[x][2],"username":social_dict[x][3], "metadata":{"liked":x in likes, "unliked":x in unlikes}} for x in social_dict],key=lambda x: x["count"],reverse=True)
    
    return main_profile


# Called when a name is given
@app.route('/name/', methods=['GET'])
def handle_name():
    name = request.args.get("name")
    app.logger.info("*** Name -> " + str(name))
    nes.index(index=config["butler_index"], doc_type="searches",body={"name":name},id=name)
    return resp

# Called when something is unliked
@app.route('/unlike/', methods=['GET'])
def handle_unlike():
    name = request.args.get("name")
    uid = request.args.get("id")
    app.logger.info("GET: Unlike " + str(name) + " " + str(uid))
    nes.index(index=config["butler_index"], doc_type="unlikes",body={"name":name,"time":datetime.now().isoformat(),"id":uid})
    try:
        nes.delete(index=config["butler_index"], doc_type="likes",id=uid)
    except NotFoundError:
        pass
    return resp

# Called when something is liked
@app.route('/like/', methods=['GET'])
def handle_like():
    name = request.args.get("name")
    lid = request.args.get("id")
    app.logger.info("GET: Like " + str(name) + " " + str(lid))
    nes.index(index=config["butler_index"], doc_type="likes",body={"name":name,"time":datetime.now().isoformat(),"id":lid})
    try:
        nes.delete(index=config["butler_index"], doc_type="unlikes",id=lid)
    except NotFoundError:
        pass
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
                "terms" : { "field" : "name", "size":500 }
            }
    }}
    result = nes.search(index=config["butler_index"], doc_type="searches", body=query)
    resp = Response(json.dumps(result["aggregations"]["searches"]["buckets"],indent=2))
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

def do_reload(name):
    query = {
    "sort" : [
        { "time" : {"order" : "desc"}},
    ],
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
    app.logger.info("GET: Reload " + name)

    qs = getQueries(name,0)

    if not qs:
        resp = Response(json.dumps({"success":True,"message":"Please start a search."}))
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp

    likes,unlikes = getLikesUnlikes(name)
    return_data = new_process(qs,name)
    if not return_data:
        resp = Response(json.dumps({"success":False,"message":"No Results.  Please start a new search"}))
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp

    resp = Response(json.dumps(return_data,indent=2))
    nes.index(index=config["butler_index"], doc_type="results",body={"name":name,"queries":qs,"data":return_data,"time":datetime.now().isoformat()})
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

# Called when save/export is clicked
@app.route('/save_export/', methods=['GET'])
def handle_save():
    name = request.args.get("name")
    return resp

@app.route('/test_url/', methods=['GET'])
def handle_test_url():
    url = request.args.get("url")
    bad_urls = []
    name = "TEST"

    # Filter Line.  Put more here.
    if url.endswith(".pdf") or any(map(url.startswith,stop)) or url in bad_urls:
        return Response(json.dumps({}))

    html = ""
    page = None
    text = ""
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
    tokens = []
    summary = ""
    likes,unlikes = [],[]

    # If we get here, it means new page

    # If page is social media
    if any(map(url.startswith,map(lambda x: x["urls"][0],social_mappings))):
        screenshot_path = getScreenShot(url)
        data = build_social_json(name,url,"social",screenshot_path)
        return Response(json.dumps(data,indent=2))

    # Page is NOT social media and NOT alredy mined AND custom extractor for it.
    elif any(map(url.startswith,map(lambda x: x["url_starts_with"],custom_extractors_list))):
        try:
            html = get_html(url)
            all_text,title = get_text_title(html,url)
            readable_text = get_readability_text(html,url)
            summary, _ = get_text_title(readable_text,url)
            summary = " ".join(summary[:500].split())
            text = all_text
            addresses = getAddresses(all_text,likes,unlikes)
            screenshot_path = getScreenShot(url)
            entities,other,tokens = doNLP(text,likes,unlikes,url)

            if url.startswith("https://www.crunchbase.com/organization/"):
                other.extend(custom_extractors.get_crunchbase_data(url))
            if url.startswith("https://www.intelius.com/people/"):
                other.extend(custom_extractors.get_intelius_data(url))
            
            emails = get_emails(all_text,likes,unlikes,url)
            phones = getPhoneNumbers(all_text,likes,unlikes,url)

            addresses = addresses[:50]
            emails = emails[:50]
            phones = phones[:50]

            lang = ""
            data = build_json(name,url,title,entities,addresses,"page",rels,emails,phones,images,other,screenshot_path,summary,lang)
            return Response(json.dumps(data,indent=2))
        except Exception as e:
            raise e
            return Response(json.dumps({}))

    else:
        try:
            html = get_html(url)
            all_text,title = get_text_title(html,url)
            readable_text = get_readability_text(html,url)
            summary, _ = get_text_title(readable_text,url)
            summary = " ".join(summary[:500].split())
            text = all_text
            addresses = getAddresses(all_text,likes,unlikes)
            screenshot_path = getScreenShot(url)
            entities,other,tokens = doNLP(text,likes,unlikes,url)
            emails = get_emails(all_text,likes,unlikes,url)
            phones = getPhoneNumbers(all_text,likes,unlikes,url)
            lang = ""
            table_rels = get_table_rels(url,html)
            other.extend(table_rels)

            addresses = addresses[:50]
            emails = emails[:50]
            phones = phones[:50]

            data = build_json(name,url,title,entities,addresses,"page",rels,emails,phones,images,other,screenshot_path,summary,lang)
            data["text"] = all_text
            return Response(json.dumps(data,indent=2))
        except Exception as e:
            raise e
            return Response(json.dumps({}))

    return Response(json.dumps({}))

@app.route('/crunch/',methods=['GET'])
def handle_crunch():
    name = request.args.get("name")
    app.logger.info("GET: Crunch -> " + name)
    qs = getQueries(name,0)
    return_data = new_process(qs,name)
    if not return_data:
        resp = Response(json.dumps({"success":False,"message":"No Results.  Please start a new search"}))
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp

    resp = Response(json.dumps(return_data,indent=2))
    nes.index(index=config["butler_index"], doc_type="results",body={"name":name,"queries":qs,"data":return_data,"time":datetime.now().isoformat()})
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
        command = Command("(" + config["chrome_loc"] + ' --headless --disable-gpu --no-sandbox --screenshot ' + url + "; mv screenshot.png ss/" + ss_id + ") &")
        code = command.run(timeout=60)
        if str(code) == "0":
            ss_return = "/ss/" + ss_id
            app.logger.info("Screenshot created -> " + url + " " + ss_return)
            return ss_return
        else:
            ss_return = "/ss/" + ss_id
            app.logger.error("Screenshot not working -> " + url)
            return ss_return
    except:
        app.logger.error("Screenshot not working -> " + url)
        ss_return = "/ss/" + ss_id
        return ss_return

def process_dark_page(in_data):
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

    try:
        app.logger.info("Page is new.")
        all_text,title = text, url_obj["title"]
        summary, _ = get_text_title(all_text,url)
        summary = " ".join(summary[:500].split())
        text = all_text
        addresses = getAddresses(all_text,likes,unlikes)
        screenshot_path = "/ss/onion.jpeg"
        
        entities,other,tokens = doNLP(text,likes,unlikes,url)
        emails = get_emails(all_text,likes,unlikes,url)
        phones = getPhoneNumbers(all_text,likes,unlikes,url)

        table_rels = get_table_rels(url,html)
        other.extend(table_rels)

        addresses = addresses[:50]
        emails = emails[:50]
        phones = phones[:50]

        data = build_json(name,url,title,entities,addresses,"dark web",rels,emails,phones,images,other,screenshot_path,summary,language)
        data = mark_data(data,likes,unlikes)
    except:
        app.logger.warn("Error occurred during processing.  Adding bad URL -> " + url)
        nes.index(index=config["butler_index"], doc_type="bad_urls",body={"name":name,"query":query,"url":url})
        return ()

    try:
        nes.index(index=config["butler_index"], doc_type="texts",body={"name":name,"query":query,"time":datetime.now().isoformat(),
        "language":language,"url":url,"text":all_text,"main_text":text,"title":title,"tokens":tokens})
    except:
        app.logger.error("ES Indexing Issue!! " + url)
        return ()
    #get_tables(url,i)
    app.logger.info("Returning back correctly!! " + url)
    return (data, text, url, entities, tokens)

def dark_search(url,auth_user,auth_pass,text,likes,unlikes,name,num_pages,language,bad_urls):
    app.logger.info("Searching dark web for: " + text)
    text = text.strip()
    QUERY = 'text:"'+text+'"'

    dark_es = Elasticsearch(
        [url],
        http_auth=(auth_user, auth_pass),
        port=443,
        use_ssl=True,
        verify_certs=False
    )

    res = dark_es.search(index="onions", q=QUERY,size=num_pages*10)
    size = res['hits']['total']
    app.logger.info(str(size) + " dark web search results found in index.")
    results = [{"url":x["_source"].get("url","http://" + x["_source"].get("domain")),"title":x["_source"]["title"],"text":x["_source"]["text"]} for x in res['hits']['hits'] if "url" in x["_source"] or "domain" in x["_source"]]
    for url in results:
        page = getByURL(url["url"],"pages",name)
        tokens = []
        if page:
            data = getByURL(url["url"],"texts",name)
            if data:
                tokens = data.get("tokens",[])
        url["page"] = page
        url["tokens"] = tokens
    trans_results = map(lambda x:({"url":x["url"],"q":text,"title":x["title"],"text":x["text"]}, (x["page"],x["text"],x["tokens"]), name, likes, unlikes, bad_urls, language), results)

    pool = Pool(processes=config["page_threads"])
    results = pool.map(process_dark_page,trans_results)
    pool.close()
    app.logger.info(str(len(results)) + " dark web search results returned from processing.")
    return results

def getQueries(name,add_sub):
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
        return map(lambda x:{"new":False,"query":x["_source"]["query"],"num_pages":max(x["_source"]["num_pages"] + add_sub,1),"language":x["_source"]["language"]},results["hits"]["hits"])
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
    app.logger.info("GET: Previous -> " + name)
    qs = getQueries(name,-1)
    return_data = new_process(qs,name)
    if not return_data:
        resp = Response(json.dumps({"success":False,"message":"No Results.  Please start a new search"}))
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp

    if not return_data:
        resp = Response(json.dumps({"success":False,"message":"No Results.  Please start a new search"}))
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp

    resp = Response(json.dumps(return_data,indent=2))
    nes.index(index=config["butler_index"], doc_type="results",body={"name":name,"queries":qs,"data":return_data,"time":datetime.now().isoformat()})
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

# Called when twitter is scraped...
@app.route('/next/', methods=['GET'])
def handle_next():
    name = request.args.get("name")
    app.logger.info("GET: Next -> " + name)
    qs = getQueries(name,1)
    return_data = new_process(qs,name)
    if not return_data:
        resp = Response(json.dumps({"success":False,"message":"No Results.  Please start a new search"}))
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp

    resp = Response(json.dumps(return_data,indent=2))
    nes.index(index=config["butler_index"], doc_type="results",body={"name":name,"queries":qs,"data":return_data,"time":datetime.now().isoformat()})
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
    url_obj, page_and_text_and_token, name, likes, unlikes, bad_urls = in_data
    url = url_obj["url"]
    query = url_obj["q"]
    lang = url_obj["language"]

    app.logger.info("** Processing -> " + url)

    # Filter Line.  Put more here.
    if url.endswith(".pdf") or any(map(url.startswith,stop)) or url in bad_urls:
        app.logger.warn("Skipping.  PDF or in STOP_LIST or in BAD_URLS. " + url)
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
        app.logger.info("Page already mined." + url)
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
        app.logger.info("Page is new.  Page is social media. " + url)

        screenshot_path = getScreenShot(url)
        data = build_social_json(name,url,"social",screenshot_path)
        data = mark_data(data,likes,unlikes)

    # Page is NOT social media and NOT alredy mined AND custom extractor for it.
    elif any(map(url.startswith,map(lambda x: x["url_starts_with"],custom_extractors_list))):
        try:
            app.logger.info("Page is new. " + url)
            if page_and_text_and_token[3]:
                html = page_and_text_and_token[4]
            else:
                html = get_html(url)
            all_text,title = get_text_title(html,url)
            readable_text = get_readability_text(html,url)
            summary, _ = get_text_title(readable_text,url)
            summary = " ".join(summary[:500].split())
            text = all_text
            addresses = getAddresses(all_text,likes,unlikes)
            screenshot_path = getScreenShot(url)
            entities,other,tokens = doNLP(text,likes,unlikes,url)

            if url.startswith("https://www.crunchbase.com/organization/"):
                other.extend(custom_extractors.get_crunchbase_data(url))
            if url.startswith("https://www.intelius.com/people/"):
                other.extend(custom_extractors.get_intelius_data(url))
            

            emails = get_emails(all_text,likes,unlikes,url)
            phones = getPhoneNumbers(all_text,likes,unlikes,url)

            addresses = addresses[:50]
            emails = emails[:50]
            phones = phones[:50]

            data = build_json(name,url,title,entities,addresses,"page",rels,emails,phones,images,other,screenshot_path,summary,lang)
            data = mark_data(data,likes,unlikes)
        except:
            app.logger.error("Error occurred during processing.  Adding bad URL -> " + url)
            nes.index(index=config["butler_index"], doc_type="bad_urls",body={"name":name,"query":query,"url":url})
            return ()

    else:
        try:
            app.logger.info("Page is new. " + url)
            if page_and_text_and_token[3]:
                html = page_and_text_and_token[4]
            else:
                html = get_html(url)
            all_text,title = get_text_title(html,url)
            readable_text = get_readability_text(html,url)
            summary, _ = get_text_title(readable_text,url)
            summary = " ".join(summary[:500].split())
            text = all_text
            addresses = getAddresses(all_text,likes,unlikes)
            screenshot_path = getScreenShot(url)
            entities,other,tokens = doNLP(text,likes,unlikes,url)
            emails = get_emails(all_text,likes,unlikes,url)
            phones = getPhoneNumbers(all_text,likes,unlikes,url)
            table_rels = get_table_rels(url,html)

            other.extend(table_rels)

            if text.strip():
                #lang = langdetect.detect(text.strip())
                app.logger.info("language is {}".format(lang))

            addresses = addresses[:50]
            emails = emails[:50]
            phones = phones[:50]

            data = build_json(name,url,title,entities,addresses,"page",rels,emails,phones,images,other,screenshot_path,summary,lang)
            data = mark_data(data,likes,unlikes)
        except:
            app.logger.warn("Error occurred during processing.  Adding bad URL -> " + url)
            nes.index(index=config["butler_index"], doc_type="bad_urls",body={"name":name,"query":query,"url":url})
            return ()

    app.logger.info("Trying to index:" + url)
    nes.index(index=config["butler_index"], doc_type="texts",body={"name":name,"query":query,"time":datetime.now().isoformat(),
    "language":lang,"url":url,"text":all_text,"main_text":text,"title":title,"tokens":tokens})
    app.logger.info("Indexed! " + url)
    app.logger.info("Returning back correctly!!" + url)
    return (data, text, url, entities, tokens)

def get_likes_to_search(last_results,likes):
    terms = []
    data_things = [
        "names",
        "emails",
        "phone_numbers",
        "addresses",
        "relationships",
        "other",
        "social_media"
    ]

    if len(last_results["hits"]["hits"]) >= 1:
        data = last_results["hits"]["hits"][-1]["_source"]["data"]
        for page in data["pages"]:
            for entity in page["entities"]:
                if entity["id"] in likes:
                    terms.append(entity["value"])

        for thing in data_things:
            for p in data["profile"][thing]:
                if p["id"] in likes:
                    if thing != "social_media":
                        terms.append(p["value"])
                    else:
                        if p["username"]:
                            terms.append(p["username"])

    return terms


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

def new_process(q,name):
    """
        Start of main process
    """

    app.logger.info("*** new_process *** => " + name)
    app.logger.info(json.dumps(q,indent=2))
    # Get likes and unlikes
    likes, unlikes = getLikesUnlikes(name)
    app.logger.info("Found %d likes and %d unlikes => " % (len(likes), len(unlikes))  + name)

    # Need to grab what the user was currently looking at
    last_results = do_reload(name)
    liked_urls = []
    url_set = set()

    likes_to_search = get_likes_to_search(last_results,likes)
    #TODO: Need to figure out how to store like value efficiently

    urls = []

    # KEEP liked pages
    if len(last_results["hits"]["hits"]) >= 1:
        data = last_results["hits"]["hits"]
        for data_res in data:
            one_res = data_res["_source"].get("data",{})
            for p in one_res.get("pages",[]):
                if p["id"] in likes and p["url"] not in url_set:
                    urls.append({"url":p["url"],"q":one_res["meta"]["q"],"language":one_res["meta"]["language"]})
                    url_set.add(p["url"])

    app.logger.info("%d liked urls we're going to keep" % (len(liked_urls)))

    # mine urls
    new_urls = get_urls(q,name)

    for url in new_urls:
        if url["url"] not in url_set:
           urls.append(url)
           url_set.add(url["url"])

    #TODO: Likes terms section - hold off for now
    """
    if likes_to_search:
        results_per_q = max(1,int(float(num_pages*8) / len(likes_to_search)))
        app.logger.info("Querying for other things: " + str(results_per_q))

        for like in likes_to_search:
            app.logger.info("Also searching '%s' as a like." % like)
            new_urls = get_urls([like],1)[:results_per_q]
            for url in new_urls:
                if url["url"] not in url_set:
                   urls.append(url)
                   url_set.add(url["url"])
    """

    app.logger.info(str(len(urls)) + " total urls found.")
    

    # Get bad urls so we can skip
    bad_urls = getBadURLS(name)
    app.logger.info("Found %d bad urls." % len(bad_urls))

    # Index each NEW query associated with the project
    for query in q:
        nes.index(index=config["butler_index"], doc_type="queries",body={"name":name,"query":query["query"],"time":datetime.now().isoformat(), "num_pages":query["num_pages"], "language":query["language"]},id=hashlib.md5(query["query"]).hexdigest())

    # For the URLS we got back, check to see if we have them already and store info if we do
    pages_and_texts_and_tokens = []
    for url in urls:
        text = ""
        tokens = []
        page = None
        page = getByURL(url["url"],"pages",name)
        if page:
            data = getByURL(url["url"],"texts",name)
            if data:
                text = data.get("text","")
                tokens = data.get("tokens",[])
        pages_and_texts_and_tokens.append((page,text,tokens,False,""))

    # Dark Web
    dark_results = []
    for silo in config.get("silos",[]):
        if silo["name"] == "Dark Web":
            for query in q:
                app.logger.info("Doing dark web search => " + name + " " + query["query"])
                dr = dark_search(silo["es_url"],silo["auth_user"],silo["auth_pass"],query["query"],likes, unlikes, name, query["num_pages"], query["language"], bad_urls)
                dark_results.extend(dr)
    app.logger.info(str(len(dark_results)))

    # CDR
    if config["cdr_search"]:
        app.logger.info("CDR SEARCH!")
        for query in q:
            for res in cdr_search.get_cdr_results(query["query"], query["num_pages"]*10):
                if res[0] not in url_set:
                    urls.append({"q":query["q"],"url":res[0],"language":query["language"]})
                    pages_and_texts_and_tokens.append((None,"",[],True,res[1]))

    # How many threads to process with        
    pool = Pool(processes=config["page_threads"])
    app.logger.info("Data Processing %d urls for %s" % (len(urls),name))
    results = []
    results = pool.map(process_single_page, map(lambda x:(x[0],x[1],name,likes,unlikes,bad_urls),zip(urls,pages_and_texts_and_tokens)))
    pool.close()

    results = results + dark_results

    app.logger.info(name + " ***** Finished %d urls" % len(results))

    # Filter out results from urls with errors
    results = filter(lambda x: x,results)

    app.logger.info(name + " %d urls after filtering" % len(results))

    if not results:
        return None

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

    app.logger.info(name + " Running LDA with %d pages" % len(doTexts))

    tree_stuff = doLDA(doTexts,0,len(doTexts),None)

    app.logger.info(name + " Done running LDA")

    populateEntries(entries,tree_stuff)
    app.logger.info(name + " Building Profile")

    profile = build_profile(entries,likes,unlikes)

    app.logger.info(name + " Profile Built")

    meta = {"name":name,"q":[q[-1]["query"]],"num_pages":len(doTexts),"language":q[-1]["language"]}

    return_data = {"profile":profile,"pages":entries,"treemap":tree_stuff,"meta":meta}

    app.logger.info("Returning Results. %d pages." % len(entries))

    return return_data


@app.route('/ss/<path:path>')
def send_js(path):
    app.logger.info("PATH: " + path)
    if os.path.isfile(os.path.join('ss',path)):
        return send_from_directory('ss', path)
    else:
        return send_from_directory('ss', "loading.png")

# Called when twitter is scraped...
@app.route('/search/', methods=['GET'])
def handle_search():
    """
        Happens when user submits search
    """
    q = request.args.get("q")
    name = request.args.get("name")
    language = request.args.get("language")
    num_pages = int(request.args.get("n",config["start_num_pages"]))

    qs = getQueries(name,0)
    qs.append({"query":q,"num_pages":num_pages,"language":language, "new":True})

    app.logger.info("*** New Search -> " +  " ".join((map(str,[name, q, num_pages, language]))))

    return_data = new_process(qs,name)
    if not return_data:
        resp = Response(json.dumps({"success":False,"message":"No Results.  Please start a new search"}))
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp

    nes.index(index=config["butler_index"], doc_type="results",body={"name":name,"queries":qs,"data":return_data,"time":datetime.now().isoformat()})

    resp = Response(json.dumps(return_data,indent=2))
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

if __name__ == "__main__":
    app.debug = True
    app.run(host="0.0.0.0",port=config["port"])

