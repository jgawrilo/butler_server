from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import json
import hashlib

config = json.load(open("config.json"))
if config["platform"] != "MAC":
    from pyvirtualdisplay import Display


def get_ok_caller(url):
    if config["platform"] != "MAC":
        display = Display(visible=0, size=(800, 600))
        display.start()

    options = webdriver.ChromeOptions()
    options.add_argument('window-size=1200x600')
    options.add_argument("--no-sandbox")

    # initialize the driver
    driver = webdriver.Chrome(chrome_options=options)
    driver.get(url)
    time.sleep(5)

    # table table-stats table-condensed table-striped table-sortable
    for i in driver.find_elements_by_css_selector('td'):
        print i.text


def get_family_tree(url):
    try:
        if config["platform"] != "MAC":
            display = Display(visible=0, size=(800, 600))
            display.start()

        options = webdriver.ChromeOptions()
        options.add_argument('window-size=1200x600')
        options.add_argument("--no-sandbox")

        # initialize the driver
        driver = webdriver.Chrome(chrome_options=options)
        driver.get(url)
        time.sleep(10)
        driver.find_elements_by_css_selector('.btn.btn-success.btn-sm.detail-link')[0].click()
        time.sleep(5)
        for i in driver.find_elements_by_css_selector('.panel.panel-default'):
            print i.text
        
    except:
        return []



def get_intelius_data(url):
    try:
        if config["platform"] != "MAC":
            display = Display(visible=0, size=(800, 600))
            display.start()

        options = webdriver.ChromeOptions()
        options.add_argument('headless')
        options.add_argument('window-size=1200x600')
        options.add_argument("--no-sandbox");

        # initialize the driver
        driver = webdriver.Chrome(chrome_options=options)
        driver.get(url)
        time.sleep(5)
        deets = driver.find_elements_by_css_selector('.little-data')
        for deet in deets:
            lines = deet.text.split("\n")[1:]
            tups = [{"id":"other"+hashlib.md5(" ".join([lines[i],lines[i+1]])).hexdigest(),"value":" ".join([lines[i+1]]),"type":lines[i].replace(":","")} for i,x in enumerate(lines) if i % 2 == 0]
            return tups
    except:
        return []

def get_crunchbase_data(url):
    tups = []
    try:
        if config["platform"] != "MAC":
            display = Display(visible=0, size=(800, 600))
            display.start()

        options = webdriver.ChromeOptions()
        options.add_argument('window-size=1200x600')
        options.add_argument("--no-sandbox");

        # initialize the driver
        driver = webdriver.Chrome(chrome_options=options)
        driver.get(url)
        time.sleep(5)
        tups = []
        deets = driver.find_elements_by_css_selector('.details.definition-list')
        for deet in deets:
            lines = deet.text.split("\n")
            tups = [{"id":"other"+hashlib.md5(" ".join([lines[i],lines[i+1]])).hexdigest(),"value":" ".join([lines[i+1]]),"type":lines[i].replace(":","")} for i,x in enumerate(lines) if i % 2 == 0]
        deets = driver.find_elements_by_css_selector('.definition-list.container')
        for deet in deets:
            lines = deet.text.split("\n")
            for i,x in enumerate(lines):
                if i % 2 == 0:
                    tups.append({"id":"other"+hashlib.md5(" ".join([lines[i],lines[i+1]])).hexdigest(),"value":" ".join([lines[i+1]]),"type":lines[i].replace(":","")})
        return tups
    except:
        return tups

if __name__ == "__main__":
    #print get_crunchbase_data("https://www.crunchbase.com/organization/qadium-solutions")
    #print get_intelius_data("https://www.intelius.com/people/Justin-Gawrilow/Washington-DC/0ca4fh5qkz1")
    #print get_family_tree("http://www.familytreenow.com/search/genealogy/results?first=Justin&last=gawrilow")
    print get_ok_caller("http://www.okcaller.com/7037277445")
