import random
import requests

def getRandomProxy():
    proxies = [
        "http://raespmdu-rotate:z64qjwq7ta18@p.webshare.io:80/",
        "https://raespmdu-rotate:z64qjwq7ta18@p.webshare.io:80/",
        "http://hqpvfzee-rotate:z9fh8f0dxkzi@p.webshare.io:80/",
        "https://hqpvfzee-rotate:z9fh8f0dxkzi@p.webshare.io:80/"
    ]

    proxy = random.choice(proxies)

    try:
        response = requests.get("https://ipv4.webshare.io/", proxies={"http": proxy, "https": proxy})
        response.raise_for_status()
        ip_address = response.text.strip()
        return {
            "status": "OK",
            "reason": "",
            "data": {
                "carrier": "",
                "city": "London",
                "country_code": "UR",
                "country_name": "America",
                "ip": ip_address,
                "isp": "FORTHnet",
                "region": "Thesaly"
            }
        }
    except requests.RequestException as e:
        return {
            "status": "ERROR",
            "reason": str(e),
            "data": {}
        }
