from selectorlib import Extractor
import requests 
import random
from dateutil import parser as dateparser
from utility.amazon.get_proxy import getRandomProxy

# Create an Extractor by reading from the YAML file
e = Extractor.from_yaml_file('utility/amazon/selectors.yml')

def scrape(url):    

   
    headers = {
        'authority': 'www.amazon.com',
        'pragma': 'no-cache',
        'cache-control': 'no-cache',
        'dnt': '1',
        'upgrade-insecure-requests': '1',
        'user-agent': ('Mozilla/5.0 (X11; Linux x86_64)' 'AppleWebKit/537.36 (KHTML, like Gecko)' 'Chrome/44.0.2403.157 Safari/537.36'),
        'Accept-Language': 'en-US, en;q=0.5',
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
    print(r.text)
    return e.extract(r.text)


def scrapped(url):
    reviewList = []
    data = scrape(url) 
    print(data)
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

if __name__ == "__main__":
    scrapped("https://www.amazon.in/Sparx-Mens-White-Slippers-8-SF0549G_NVWH0008/dp/B07QRJJXDT/?_encoding=UTF8&pd_rd_w=gFdfX&content-id=amzn1.sym.6aeb164c-387d-440e-8808-65edf45c4683&pf_rd_p=6aeb164c-387d-440e-8808-65edf45c4683&pf_rd_r=PAQZR60CR8TAH8H482KM&pd_rd_wg=MM9m9&pd_rd_r=a3d9c495-a75f-4997-912b-b41ed0e252de&ref_=pd_gw_ci_mcx_mr_hp_atf_m")