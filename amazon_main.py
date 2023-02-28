from utility.amazon.amazon_review_scrapper import scrapped
from utility.amazon.amazon_senti import senti
import pandas as pd
import os

header=["title","content","date","variant","images","verified","author","rating","product","url"]

def amazonMain():
    urls = [
        "https://www.amazon.in/Samsung-Galaxy-Storage-5000mAh-Battery/product-reviews/B0B4F3G74S/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews",
        "https://www.amazon.in/realme-Racing-Storage-Additional-Exchange/product-reviews/B0993YD3KJ/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews",
        "https://www.amazon.in/Test-Exclusive_2020_1159-Multi-3GB-Storage/product-reviews/B089MVC43X/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews"
    ]
    for productUrl in urls:
        reviewList = []
        flag = 0
        for i in range(400):
            print(f"Running for page {i}")
            try: 
                reviewUrl = productUrl.replace('dp','product-reviews') + '&pageNumber=' + str(i)
                list = scrapped(reviewUrl)
                if list is not None:
                    print(f'{len(list)} ====  {flag}')
                    reviewList.extend(list)
                else:
                    print(flag)
                    flag = flag + 1

                if(flag >= 5):
                    print(f"No reviews found on page {i}")
                    break

            except Exception as e:
                print(e)
                flag = flag + 1
                print(flag)

                if(flag >= 5):
                    print(f"No reviews found on page {i}")
                    break

        output_df = pd.DataFrame.from_dict(reviewList)
        output_df.to_csv(f'data/am1-train.csv', index=False, header=not os.path.exists('data/am1-train.csv'), mode='a')
    senti()
    

if __name__ == "__main__":
    amazonMain()
    
