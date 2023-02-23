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

    df = pd.read_csv('../../data/yt-train.csv', parse_dates=['date'])
    df.dropna(subset=['comment'], inplace=True)
    filename = "../../model/youtube/yt-logistic.joblib"
    model = joblib.load(filename)
    vectorizer = joblib.load("../../model/youtube/yt-vector-logistic.joblib")


    df['sentiment'] = df['comment'].apply(lambda x: predict_sentiment(model, vectorizer, x))

    # Splitting the data based on sentiment
    positive = df[df['sentiment'] == 1]
    negative = df[df['sentiment'] == -1]
    neutral = df[df['sentiment'] == 0]

    # Histogram 
    sentiment_counts = df['sentiment'].value_counts()
    sentiment_df = pd.DataFrame({'Sentiment': sentiment_counts.index, 'Count': sentiment_counts.values})
    fig = px.histogram(sentiment_df, x='Sentiment', y='Count', color='Sentiment', color_discrete_sequence=['green', 'red', 'gray'])
    fig.update_layout(title_text='Sentiment Analysis', xaxis_title='Sentiment', yaxis_title='Count')
    pio.write_image(fig, 'sentiment_histogram.png')

    # Create stopword list:
    stopwords = set(STOPWORDS)
    stopwords.update(["br", "href", "good", "great"])

    if len(positive) > 0:
        pos = " ".join(str(review) for review in positive.comment)
        wordcloud2 = WordCloud(stopwords=stopwords).generate(pos)
        plt.imshow(wordcloud2, interpolation='bilinear')
        plt.axis("off")
        plt.savefig('wordcloud_positive.png')

    if len(negative) > 0:
        neg = " ".join(str(review) for review in negative.comment)
        wordcloud3 = WordCloud(stopwords=stopwords).generate(neg)
        plt.imshow(wordcloud3, interpolation='bilinear')
        plt.axis("off")
        plt.savefig('wordcloud_negative.png')

    if len(neutral) > 0:
        neu = " ".join(str(review) for review in neutral.comment)
        wordcloud4 = WordCloud(stopwords=stopwords).generate(neu)
        plt.imshow(wordcloud4, interpolation='bilinear')
        plt.axis("off")
        plt.savefig('wordcloud_neutral.png')

    # Group the reviews by date and count the number of reviews on each day for each sentiment
    pos_reviews_per_day = positive.groupby(positive['date'].dt.date)['comment'].count()
    neg_reviews_per_day = negative.groupby(negative['date'].dt.date)['comment'].count()
    neu_reviews_per_day = neutral.groupby(neutral['date'].dt.date)['comment'].count()

    # Group the reviews by date and count the number of reviews on each day
    reviews_per_day = df.groupby(df['date'].dt.date)['comment'].count()



    # Create the time series plot for each sentiment
    fig = plt.figure(figsize=(12,6))
    plt.plot(pos_reviews_per_day, label='Positive')
    plt.plot(neg_reviews_per_day, label='Negative')
    plt.plot(neu_reviews_per_day, label='Neutral')
    plt.xlabel('date', fontsize=14)
    plt.ylabel('Number of comments', fontsize=14)
    plt.title('Amazon Reviews Time Series Plot', fontsize=16)
    plt.xticks(fontsize=12, rotation=45)
    plt.legend(fontsize=12)
    plt.savefig('time_series_sentiment.png')

    # Create the time series plot for reviews
    fig = plt.figure(figsize=(12,6))
    plt.plot(reviews_per_day)
    plt.xlabel('date', fontsize=14)
    plt.ylabel('Number of Reviews', fontsize=14)
    plt.title('Amazon Reviews Time Series Plot', fontsize=16)
    plt.xticks(fontsize=12, rotation=45)
    plt.savefig('time_series.png')


