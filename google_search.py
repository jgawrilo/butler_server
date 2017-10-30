from google import google
import search2

import json

'''
Basically make this a service.
Add account api calls in there?
'''

def get_urls(terms,num_pages=1):
    """
        get results from google for search terms
    """
    results = []
    # First try google search API
    for term in terms:
        print "Trying API"
        search_results = google.search(term, num_pages)
        results.extend([{"q":term,"url":x.link} for x in search_results])

    print len(results)
    print json.dumps(results)

    # If it's not working, we might be blocked. Get results through browser
    if not results:
        print "Browser"
        results = map(lambda x: {"q":terms[0],"url":x}, search2.do_search(terms[0],num_pages))
    return results


if __name__ == '__main__':
    links = get_urls(["justin gawrilow"])
    print len(links)
    print json.dumps(links)
