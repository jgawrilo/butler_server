# Butler Server
This is the web server and analytics component of Butler.  If you're looking to install the full Butler application you may want to check out the [Dockerized Butler Installation](https://github.com/jgawrilo/butler_install) as it will be easier.  If you're set on installing the the bare software components please continue on.

## Service Dependencies
This project depends on an [Elasticsearch](https://www.elastic.co/guide/en/elasticsearch/reference/current/_installation.html) instance as well as a [CoreNLP Server](https://stanfordnlp.github.io/CoreNLP/index.html#download).  Please follow the links for installation if you plan to install and use locally. 

You'll need to ensure these endpoints ("es" and "nlp_service") are properly configured in the config.json file.  Please see config.json.template for details.

## CoreNLP Service
Here is the command to run the CoreNLP Server locally (once in directory).  It is reccomended you run with 8gb RAM and 2 threads as below.

`java -mx8g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 25000 -threads 2`

## Latest Chrome and ChromeDriver
Additionally the project depends on [Chrome 59](https://developers.google.com/web/updates/2017/04/headless-chrome) for screenshots and web scrapes.
[ChromeDriver](https://sites.google.com/a/chromium.org/chromedriver/)

## Configuration and Install
`cp config.json.template config.json`

You'll need to change config.json settings to point to your Elasticsearch instance, the port you'd like to serve the butler server on, and your CoreNLP service endpoing.

`pip install -r requirements.txt`

`./nltk.sh`

## Setting up Elasticsearch Index
`python utils.py create` - creates buter lindex

## gunicorn

## Running Server
`python server.py`
`gunicorn --access-logfile - -w 1 -b 0.0.0.0:5000 --timeout 600 server:app`

## Utility Scripts
### Warning: These will wipe all data.
`python utils.py delete` - deletes buter data index

`python utils.py reset` - recreates buter data index
