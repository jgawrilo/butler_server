# butler_server

## Service Dependencies
This project depends on an [Elasticsearch](https://www.elastic.co/guide/en/elasticsearch/reference/current/_installation.html) instance as well as a [CoreNLP Server](https://stanfordnlp.github.io/CoreNLP/corenlp-server.html).  Please follow the links if you plan to install and use locally.

Here is the command to run the CoreNLP Server locally.

`java -mx8g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 25000 -threads 2`

## Configuration and Install
`cp config.json.template config.json`

You'll need to change config.json settings to point to your Elasticsearch instance, the port you'd like to serve the butler server on, and your CoreNLP service endpoing.

`pip install -r requirements.txt`

`./nltk.sh`

## Setting up Elasticsearch Index
`python utils.py create` - creates buter lindex

## Running Server
`python server.py`

## Utility Scripts
`python utils.py delete` - deletes buter lindex

`python utils.py reset` - recreates buter lindex
