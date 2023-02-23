from amazon_review_scrapper import scrapped
from amazon_senti import senti
import csv

def main():
    productUrl = "https://www.amazon.com/dp/B01E8IZ4ZA?linkCode=ogi&tag=cosmopolitan_auto-append-20&ascsubtag=%5Bartid%7C10049.g.38666976%5Bsrc%7C%5Bch%7C%5Blt%7Csale%5Bpid%7Cf5476327-d8c1-4e0a-81c3-fd7e0c3353f8"
    for i in range(100):
        print(f"Running for page {i}")
        try: 
            reviewUrl = productUrl.replace('dp','product-reviews') + '&pageNumber=' + str(i)
            scrapped(reviewUrl)
        except Exception as e:
            print(e)
    
    senti()

    

if __name__ == "__main__":
    main()