from selectorlib import Extractor
import requests 
import random
from dateutil import parser as dateparser
from utility.amazon.get_proxy import getRandomProxy

# Create an Extractor by reading from the YAML file
e = Extractor.from_yaml_file('utility/amazon/selectors.yml')

def scrape(url):    

    user_agent_list = [
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.1.1 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:77.0) Gecko/20100101 Firefox/77.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:77.0) Gecko/20100101 Firefox/77.0',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
    ]
    user_agent = random.choice(user_agent_list)
    headers = {
        'authority': 'www.amazon.com',
        'pragma': 'no-cache',
        'cache-control': 'no-cache',
        'dnt': '1',
        'upgrade-insecure-requests': '1',
        'user-agent': user_agent,
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'sec-fetch-site': 'none',
        'sec-fetch-mode': 'navigate',
        'sec-fetch-dest': 'document',
        'accept-language': 'en-GB,en-US;q=0.9,en;q=0.8',
    }

    # Download the page using requests
    print("Downloading %s"%url)
    r = requests.get(url, headers=headers, proxies=getRandomProxy())
    # Simple check to check if page was blocked (Usually 503)
    if r.status_code > 500:
        if "To discuss automated access to Amazon data please contact" in r.text:
            print("Page %s was blocked by Amazon. Please try using better proxies\n"%url)
        else:
            print("Page %s must have been blocked by Amazon as the status code was %d"%(url,r.status_code))
        return None
    # Pass the HTML of the page and create 
    return e.extract(r.text)


def scrapped(url):
    reviewList = []
    data = scrape(url) 
    if data:
        for r in data['reviews']:
            r["product"] = data["product_title"]
            r['url'] = url
            if 'verified' in r:
                if 'Verified Purchase' in r['verified']:
                    r['verified'] = 'Yes'
                else:
                    r['verified'] = 'No'
            r['rating'] = r['rating'].split(' out of')[0]
            date_posted = r['date'].split('on ')[-1]
            if r['images']:
                r['images'] = "\n".join(r['images'])
            r['date'] = dateparser.parse(date_posted).strftime('%d %b %Y')
            reviewList.append(r)
    
    return reviewList