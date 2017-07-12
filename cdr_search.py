from elasticsearch import Elasticsearch
import json

config = json.load(open("config.json"))

def get_cdr_results(term, num_results):
    dark_es = Elasticsearch(
        [config["cdr_url"]],
        http_auth=(config["cdr_u"], config["cdr_p"]),
        port=443,
        use_ssl=True,
        verify_certs=False
    )
    QUERY="raw_content:" + term
    res = dark_es.search(index="memex-domains", q=QUERY,size=num_results)
    
    size = res['hits']['total']
    print size
    return map(lambda x: (x["_source"]["url"],x["_source"]["raw_content"]), res["hits"]["hits"])
    # url and raw_content

if __name__ == "__main__":
    print get_cdr_results("juddy",10)[0][0]