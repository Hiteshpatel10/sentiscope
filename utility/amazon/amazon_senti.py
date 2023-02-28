import pandas as pd

# For ploting graph
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
import plotly.offline as py
import plotly.io as pio
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.express as px

#nltk
import nltk
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
from wordcloud import WordCloud

# none
import numpy as np
import joblib

def predict_sentiment(model, vectorizer, text):
   
    # Preprocess the input text using the TfidfVectorizer object
    features = vectorizer.transform([text])
    # Reshape the feature vector to have a 2D shape of (1, n)
    features = features.reshape(1, -1)
    # Predict the sentiment label using the model
    sentiment = model.predict(features)[0]
    return sentiment


def senti():

    df = pd.read_csv('../../data/am1-train.csv', parse_dates=['date'])
    df.dropna(subset=['content'], inplace=True)
    filename = "../../model/amazon/amazon-logistic.joblib"
    model = joblib.load(filename)
    vectorizer = joblib.load("../../model/amazon/amazon-vector-logistic.joblib")


    df['sentiment'] = df['content'].apply(lambda x: predict_sentiment(model, vectorizer, x))

    # Splitting the data based on sentiment
    positive = df[df['sentiment'] == 1]
    negative = df[df['sentiment'] == -1]
    neutral = df[df['sentiment'] == 0]

    # Review no histogram
    fig = px.histogram(df, x="rating")
    fig.update_traces(marker_color="turquoise",marker_line_color='rgb(8,48,107)',marker_line_width=1.5)
    fig.update_layout(title_text='Product Score')
    pio.write_image(fig, 'product_score.png')

    # Create stopword list:
    stopwords = set(STOPWORDS)
    stopwords.update(["br", "href", "good", "great"])

    ## good and great removed because they were included in negative sentiment
    pos = " ".join(str(review) for review in positive.content)
    wordcloud2 = WordCloud(stopwords=stopwords).generate(pos)
    plt.imshow(wordcloud2, interpolation='bilinear')
    plt.axis("off")

    #Negative review wordcloud
    neg = " ".join(str(review) for review in negative.content)
    wordcloud3 = WordCloud(stopwords=stopwords).generate(neg)
    plt.imshow(wordcloud3, interpolation='bilinear')
    plt.axis("off")
    plt.savefig('wordcloud33.png')

    #Negative review wordcloud
    neu= " ".join(str(review) for review in neutral.content)
    wordcloud4 = WordCloud(stopwords=stopwords).generate(neu)
    plt.imshow(wordcloud4, interpolation='bilinear')
    plt.axis("off")
    plt.savefig('wordcloud44.png')

    # Group the reviews by date and count the number of reviews on each day for each sentiment
    pos_reviews_per_day = positive.groupby(positive['date'].dt.date)['rating'].count()
    neg_reviews_per_day = negative.groupby(negative['date'].dt.date)['rating'].count()
    neu_reviews_per_day = neutral.groupby(neutral['date'].dt.date)['rating'].count()

    # Group the reviews by date and count the number of reviews on each day
    reviews_per_day = df.groupby(df['date'].dt.date)['rating'].count()



    # Create the time series plot for each sentiment
    fig = plt.figure(figsize=(12,6))
    plt.plot(pos_reviews_per_day, label='Positive')
    plt.plot(neg_reviews_per_day, label='Negative')
    plt.plot(neu_reviews_per_day, label='Neutral')
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Number of Reviews', fontsize=14)
    plt.title('Amazon Reviews Time Series Plot', fontsize=16)
    plt.xticks(fontsize=12, rotation=45)
    plt.legend(fontsize=12)
    plt.savefig('time_series.png')

    # Create the time series plot for reviews
    fig = plt.figure(figsize=(12,6))
    plt.plot(reviews_per_day)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Number of Reviews', fontsize=14)
    plt.title('Amazon Reviews Time Series Plot', fontsize=16)
    plt.xticks(fontsize=12, rotation=45)
    plt.savefig('time_series.png')



if __name__ == "__main__":
    senti() 

