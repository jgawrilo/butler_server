# butler_server

## Service Dependencies
This project depends on an [Elasticsearch](https://www.elastic.co/guide/en/elasticsearch/reference/current/_installation.html) instance as well as a [CoreNLP Server](https://stanfordnlp.github.io/CoreNLP/corenlp-server.html).  Please follow the links if you plan to install and use locally.

## Configuration
`cp config.json.template config.json`

You'll need to change these settings to point to your Elasticsearch instance, the port you'd like to serve the butler server on, and your CoreNLP service endpoing.

`pip install -r requirements.txt`

`python server.py`
