# Butler Server
This is the web server and analytics component of Butler.  If you're looking to install the full Butler application you may want to check out the [Dockerized Butler Installion](https://github.com/jgawrilo/butler_install) as it will probably be easier.  If you're set on installing the components on your own continue on.

## Service Dependencies
This project depends on an [Elasticsearch](https://www.elastic.co/guide/en/elasticsearch/reference/current/_installation.html) instance as well as a [CoreNLP Server](https://stanfordnlp.github.io/CoreNLP/index.html#download).  Please follow the links if you plan to install and use locally. Otherwise you'll need to ensure these endpoints are properly configured in the config.json file.

Here is the command to run the CoreNLP Server locally (once in directory).

`java -mx8g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 25000 -threads 2`

Additionally the project depends on [Chrome 59](https://developers.google.com/web/updates/2017/04/headless-chrome) for screenshots.

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
