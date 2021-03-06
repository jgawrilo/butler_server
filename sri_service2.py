import json, requests, hashlib
from requests.auth import HTTPBasicAuth

config = json.load(open("config.json"))

philds = {
  "email_addr":"emails",
  "phone_num":"phones",
  "bitcoin_addr":"bitcoins",
  "pgp_key_ls":"pgps",
  "pgp_key_hash": "pgp hashes",
  "org": "organizations",
  "person_name":"people",
  "gpe": "gpes",
  "pgp_email_addr":"pgp emails",
  "ssn_num":"ssns",
  "onion_appearance":"onions"
}

def star_search(data):
  try:
    headers = {'content-type': 'application/json'}
    resp = requests.post(config["star_search_url"], auth=HTTPBasicAuth(config["star_search_auth_u"], config["star_search_auth_p"]),data = json.dumps(data), verify=False, headers=headers)
    results = json.loads(json.JSONDecoder().decode(resp.text))
    ret_results = {}
    for key in results.keys():
      obj = results[key]
      for field in philds:
        temp_res = obj.get(field,{}).keys()
        print field
        print temp_res
        temp_res = list(map(lambda x: str(x.replace("u'","").replace("'","")),temp_res))
        print temp_res
        data = list(map(lambda x:{"id":hashlib.md5(x).hexdigest(),"value":x},temp_res))
        if data:
          ret_results[philds[field]] = ret_results.get(philds[field],[])
          ret_results[philds[field]].extend(data)
    return ret_results
  except:
    return {}


if __name__ == '__main__':
  print star_search({"Email":"test@mit.edu","RegistrationKey": "key", "Action": "analyze-v2"})
