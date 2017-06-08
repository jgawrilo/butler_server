
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import json

config = json.load(open("config.json"))
if config["platform"] != "MAC":
    from pyvirtualdisplay import Display
    

def do_search(q,num_pages=1):
    if config["platform"] != "MAC":
        display = Display(visible=0, size=(800, 600))
        display.start()

    options = webdriver.ChromeOptions()
    options.add_argument('headless')
    options.add_argument('window-size=1200x600')
    options.add_argument("--no-sandbox");

    # initialize the driver
    driver = webdriver.Chrome(chrome_options=options)
    driver.get("http://www.google.com")
    input_element = driver.find_element_by_name("q")
    input_element.send_keys(q)
    input_element.submit()

    urls = []

    RESULTS_LOCATOR = "//div/h3/a"

    WebDriverWait(driver, 10).until(
        EC.visibility_of_element_located((By.XPATH, RESULTS_LOCATOR)))

    page1_results = driver.find_elements(By.XPATH, RESULTS_LOCATOR)

    for item in page1_results:
        #print(item.text, item.get_attribute("href"))
        urls.append(item.get_attribute("href"))

    done_num = 1

    for i in range(done_num,num_pages):
        clicker = driver.find_element_by_class_name("pn")
        clicker.click()

        time.sleep(5)

        page1_results = driver.find_elements(By.XPATH, RESULTS_LOCATOR)

        for item in page1_results:
            #print(item.text, item.get_attribute("href"))
            urls.append(item.get_attribute("href"))

    if config["platform"] != "MAC":
        display.stop()
        
    return urls
