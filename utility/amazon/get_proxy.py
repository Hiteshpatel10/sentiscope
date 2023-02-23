import requests

def getRandomProxy():
    proxy={
        "http": "http://raespmdu-rotate:z64qjwq7ta18@p.webshare.io:80/",
        "https": "http://raespmdu-rotate:z64qjwq7ta18@p.webshare.io:80/"
    }

    response = requests.get(
        "https://ipv4.webshare.io/", proxies= proxy
    )

    res = {
    "status": "OK",
    "reason": "",
    "data": {
        "carrier": "",
        "city": "Udaipur",
        "country_code": "GR",
        "country_name": "Greece",
        "ip": response.text,
        "isp": "FORTHnet",
        "region": "Thesaly"
        
        }
    }

    return res