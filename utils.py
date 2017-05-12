# -*- coding: utf-8 -*-

from elasticsearch import Elasticsearch
import json
import sys

CONFIG = "config.json"
SCHEMA = "schema.json"
BUTLER_INDEX = "butler"

class Creator(object):

  def __init__(self):
    self.config = json.load(open(CONFIG))
    self.es = Elasticsearch([self.config["es"]])

  def delete(self):
    self.es.indices.delete(index=BUTLER_INDEX, ignore=[400, 404])

  def create(self):
    mapping = open(SCHEMA).read()
    self.es.indices.create(index=BUTLER_INDEX, body=mapping, ignore=[400, 404])

  def reset(self):
    self.delete()
    self.create()


if __name__ == "__main__":
  creator = Creator()
  if sys.argv[1] == "create":
    creator.create()
  elif sys.argv[1] == "delete":
    creator.delete()
  elif sys.argv[1] == "reset":
    creator.reset()
  else:
    print "Huh?"

  


