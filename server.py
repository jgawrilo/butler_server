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

from flask import Flask, request, Response
from gensim import corpora
from collections import defaultdict
from gensim.models.hdpmodel import HdpModel
import haul
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pickle

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

page_dict = {}
entity_dict = {}
address_dict = {}
email_dict = {}
phone_dict = {}
text_dict = {}
dislike_page_set = set()
dislike_phone_set = set() 

app = Flask(__name__)

name = "unknown"
data_state = {}

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
        entity_dict[ent_txt] = entity_dict.get(ent_txt,"e"+str(len(entity_dict)))
    return [{"value":x[0],"type":x[1],"count":ent_dict[x],"id":entity_dict[x[0]]} for x in ent_dict if x[1] not in 
    ["CARDINAL","DATE","MONEY","PERCENT","TIME","WORK_OF_ART"]]

def getPhoneNumbers(text):
    phones = []
    for match in re.finditer(r"\(?\b[2-9][0-9]{2}\)?[-. ]?[2-9][0-9]{2}[-. ]?[0-9]{4}\b", text):
        match = match.group()
        if match not in dislike_page_set:
            phone_dict[match] = phone_dict.get(match,"phone"+str(len(phone_dict)))
            phones.append({"id":phone_dict[match],"value":match})
    return phones


def get_emails(text):
    """Returns an iterator of matched emails found in string s."""
    # Removing lines that start with '//' because the regular expression
    # mistakenly matches patterns like 'http://foo@bar.com' as '//foo@bar.com'.
    ret = []
    for email in (email[0] for email in re.findall(regex, text) if not email[0].startswith('//')):
        email_dict[email] = email_dict.get(email,"em"+str(len(email_dict)))
        ret.append({"id":email_dict[email],"value":email})
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
    for address in addresses:
        address_dict[address] = address_dict.get(address,"a"+str(len(address_dict)))
    return map(lambda x: {"id":address_dict[x],"value":x},addresses)

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
    headers = headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2172.95 Safari/537.36'}
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

def build_social_json(url,ptype):
    pid = page_dict.get(url,"p"+str(len(page_dict)))
    data = {
        "url":url,
        "id":pid,
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
    page_dict[url] = data
    return data

def build_json(url,title,entities,addresses,ptype,rels,emails,phones,images):
    pid = page_dict.get(url,"p"+str(len(page_dict)))
    data = {
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
    page_dict[url] = data
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
            "usernames":[],
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
            social_dict[e["id"]] = social_dict.get(e["id"],[e["url"],0])
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
    main_profile["social_media"] = sorted([{"id":x,"url":social_dict[x][0],"count":social_dict[x][1]} for x in social_dict],key=lambda x: x["count"],reverse=True)
    
    return main_profile

# Called when a name is given
@app.route('/name/', methods=['GET'])
def handle_name():
    global data_state
    print data_state
    print "GET: Name"
    global name
    name = request.args.get("name")
    os.system("mkdir -p ./saves/"+name)
    return resp


# Called when something is unliked
@app.route('/unlike/', methods=['GET'])
def handle_unlike():
    print "GET: Unlike"
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
    
    return crunch()

# Called when something is liked
@app.route('/like/', methods=['GET'])
def handle_like():
    print "GET: Like"
    sid = request.args.get("id")
    if sid.startswith("phone"):
        for phone in phone_dict:
            if phone_dict[phone] == sid:
                return new_search(phone)

# Called when clear is clicked
@app.route('/clear/', methods=['GET'])
def handle_clear():
    global page_dict, entity_dict, address_dict, email_dict, phone_dict, text_dict, dislike_page_set
    global dislike_phone_set, name, data_state

    page_dict = {}
    entity_dict = {}
    address_dict = {}
    email_dict = {}
    phone_dict = {}
    text_dict = {}
    dislike_page_set = set()
    dislike_phone_set = set() 

    name = "unknown"
    data_state = {}
    print "GET: Clear"
    return resp


# Called when reload is clicked
@app.route('/current/', methods=['GET'])
def handle_current():
    resp = Response(json.dumps(data_state,indent=2))
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

# Called when reload is clicked
@app.route('/reload/', methods=['GET'])
def handle_reload():
    global page_dict, entity_dict, address_dict, email_dict, phone_dict, text_dict, dislike_page_set
    global dislike_phone_set, name, data_state
    print "GET: Reload"

    name = request.args.get("name")
    with codecs.open("./saves/" + name + "/butler_data.json","r",encoding="utf8") as ip:
        data_dump = json.loads(ip.read())
        page_dict = data_dump["page_dict"]
        entity_dict = data_dump["entity_dict"]
        address_dict = data_dump["address_dict"]
        email_dict = data_dump["email_dict"]
        phone_dict = data_dump["phone_dict"]
        text_dict = data_dump["text_dict"]
        data_state = data_dump["data_state"]
        with open ("./saves/" + name + "/dislike_page_set.json", 'rb') as fp:
            dislike_page_set = pickle.load(fp) 
    return resp

# Called when save/export is clicked
@app.route('/save_export/', methods=['GET'])
def handle_save():
    global page_dict, entity_dict, address_dict, email_dict, phone_dict, text_dict, dislike_page_set
    global dislike_phone_set, name, data_state
    data_dump = {"page_dict":page_dict,
        "entity_dict":entity_dict,
        "address_dict":address_dict,
        "email_dict":email_dict,
        "phone_dict":phone_dict,
        "text_dict":text_dict,
        "data_state":data_state
        }
    with open("./saves/" + name + "/dislike_page_set.json", 'wb') as fp:
        pickle.dump(dislike_page_set, fp)

    print "GET: Save"
    with codecs.open("./saves/" + name + "/butler_data.json","w",encoding="utf8") as output:
        output.write(json.dumps(data_dump,indent=2))

    return resp

def crunch():

    q = "CHANGE ME"

    global data_state

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
            data = build_social_json(url,"social")
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

        #with codecs.open("data/"+str(i)+".html","w",encoding="utf8",errors="ignore") as out:
        #    out.write(html)
        #with codecs.open("data/"+str(i)+".txt","w",encoding="utf8",errors="ignore") as out:
        #    out.write(text)
        #with codecs.open("data/"+str(i)+".rtxt","w",encoding="utf8",errors="ignore") as out:
        #    out.write(readable_text)
        
        #with codecs.open("data/"+str(i)+".sstxt","w",encoding="utf8",errors="ignore") as out:
        #    out.write(ss_text)

        if text != "":
            pass
            #rels = getRelationships(str(i)+".txt")
        else:
            rels = []
        if not social:
            data = build_json(url,title,entities,addresses,"page",rels,emails,phones,images)
        #with codecs.open("data/"+str(i)+".json","w",encoding="utf8",errors="ignore") as out:
        #    out.write(json.dumps(data,indent=2))
        entries.append(data)
        #idx_out.write(str(i) + "\t" + url + "\n")
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts) #fit the vectorizer to synopses
    answers = doLDA(texts,q)

    for i,a in enumerate(answers):
        entries[i]["topic"] = a

    #print(tfidf_matrix.shape)
    dist = 1 - cosine_similarity(tfidf_matrix)
    
    #idx_out.close()

    linkage_matrix = linkage(dist) #define the linkage_matrix using ward clustering pre-computed distances
    #print good_urls
    #print linkage_matrix

    tree = to_tree(linkage_matrix)
    #print tree
    d3Dendro = dict(children=[], name="Top",count=len(good_urls))
    add_node(tree, d3Dendro, good_urls)

    updateAllNodes(d3Dendro["children"][0])

    profile = build_profile(entries)

    return_data = {"profile":profile,"pages":entries,"treemap":d3Dendro["children"][0]}

    data_state = return_data

    #with codecs.open("data/data.json","w",encoding="utf8",errors="ignore") as out:
    #    out.write(json.dumps(return_data,indent=2))

    print json.dumps(return_data)
    resp = Response(json.dumps(return_data,indent=2))
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp




# Called when twitter is scraped...
@app.route('/search/', methods=['GET'])
def handle_search():
    global data_state
    q = request.args.get("q")
    num_pages = request.args.get("n")
    if num_pages == None:
        num_pages = 1
    print q
    print num_pages, "pages"
    
    #filelist = glob.glob("data/*")
    #for f in filelist:
    #    os.remove(f)

    #idx_out = codecs.open("data/idx.txt","w",encoding="utf8")

    urls = get_urls(q,num_pages)
    texts = []
    good_urls = []
    entries = []
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
            good_urls.append(url)
            continue

        if url in dislike_page_set or url.endswith(".pdf") or any(map(url.startswith,stop)):
            print "Skipping..."
            continue

        if any(map(url.startswith,map(lambda x: x["urls"][0],social_mappings))):
            print "Social"
            data = build_social_json(url,"social")
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

        #with codecs.open("data/"+str(i)+".html","w",encoding="utf8",errors="ignore") as out:
        #    out.write(html)
        #with codecs.open("data/"+str(i)+".txt","w",encoding="utf8",errors="ignore") as out:
        #    out.write(text)
        #with codecs.open("data/"+str(i)+".rtxt","w",encoding="utf8",errors="ignore") as out:
        #    out.write(readable_text)
        
        #with codecs.open("data/"+str(i)+".sstxt","w",encoding="utf8",errors="ignore") as out:
        #    out.write(ss_text)

        if text != "":
            pass
            #rels = getRelationships(str(i)+".txt")
        else:
            rels = []
        if not social:
            data = build_json(url,title,entities,addresses,"page",rels,emails,phones,images)
        #with codecs.open("data/"+str(i)+".json","w",encoding="utf8",errors="ignore") as out:
        #    out.write(json.dumps(data,indent=2))
        entries.append(data)
        #idx_out.write(str(i) + "\t" + url + "\n")
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts) #fit the vectorizer to synopses
    answers = doLDA(texts,q)

    for i,a in enumerate(answers):
        entries[i]["topic"] = a

    #print(tfidf_matrix.shape)
    dist = 1 - cosine_similarity(tfidf_matrix)
    
    #idx_out.close()

    linkage_matrix = linkage(dist) #define the linkage_matrix using ward clustering pre-computed distances
    #print good_urls
    #print linkage_matrix

    tree = to_tree(linkage_matrix)
    #print tree
    d3Dendro = dict(children=[], name="Top",count=len(good_urls))
    add_node(tree, d3Dendro, good_urls)

    updateAllNodes(d3Dendro["children"][0])

    profile = build_profile(entries)

    return_data = {"profile":profile,"pages":entries,"treemap":d3Dendro["children"][0]}

    data_state = return_data

    #with codecs.open("data/data.json","w",encoding="utf8",errors="ignore") as out:
    #    out.write(json.dumps(return_data,indent=2))

    print json.dumps(return_data)
    resp = Response(json.dumps(return_data,indent=2))
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

if __name__ == "__main__":
    app.debug=True
    #context = ('server.crt', 'server.key')

    print "Running..."
    app.run(threaded=True,
        host="0.0.0.0",
        port=(5000)) #ssl_context=context)
    

