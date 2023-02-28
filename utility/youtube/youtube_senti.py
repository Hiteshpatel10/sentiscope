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

    df = pd.read_csv('./data/yt-train.csv', parse_dates=['date'])
    df.dropna(subset=['comment'], inplace=True)
    filename = "./model/youtube/yt-logistic.joblib"
    model = joblib.load(filename)
    vectorizer = joblib.load("./model/youtube/yt-vector-logistic.joblib")


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
    # fig.write_image('./static/youtube/sentiment_histogram.png')
    fig.write_html('./static/youtube/sentiment_histogram.html')


    # Create stopword list:
    stopwords = set(STOPWORDS)
    stopwords.update(["br", "href", "good", "great"])

    if len(positive) > 0:
        pos = " ".join(str(review) for review in positive.comment)
        wordcloud2 = WordCloud(stopwords=stopwords).generate(pos)
        plt.imshow(wordcloud2, interpolation='bilinear')
        plt.axis("off")
        plt.savefig('./static/youtube/wordcloud_positive.png', transparent=True)

    if len(negative) > 0:
        neg = " ".join(str(review) for review in negative.comment)
        wordcloud3 = WordCloud(stopwords=stopwords).generate(neg)
        plt.imshow(wordcloud3, interpolation='bilinear')
        plt.axis("off")
        plt.savefig('./static/youtube/wordcloud_negative.png', transparent=True)

    if len(neutral) > 0:
        neu = " ".join(str(review) for review in neutral.comment)
        wordcloud4 = WordCloud(stopwords=stopwords).generate(neu)
        plt.imshow(wordcloud4, interpolation='bilinear')
        plt.axis("off")
        plt.savefig('./static/youtube/wordcloud_neutral.png', transparent=True)


    # Group the reviews by date and count the number of reviews on each day for each sentiment
    pos_reviews_per_day = positive.groupby(positive['date'].dt.date)['comment'].count()
    neg_reviews_per_day = negative.groupby(negative['date'].dt.date)['comment'].count()
    neu_reviews_per_day = neutral.groupby(neutral['date'].dt.date)['comment'].count()

    # Group the reviews by date and count the number of reviews on each day
    reviews_per_day = df.groupby(df['date'].dt.date)['comment'].count()



    # Create the time series plot for each sentiment
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pos_reviews_per_day.index, y=pos_reviews_per_day.values, name='Positive'))
    fig.add_trace(go.Scatter(x=neg_reviews_per_day.index, y=neg_reviews_per_day.values, name='Negative'))
    fig.add_trace(go.Scatter(x=neu_reviews_per_day.index, y=neu_reviews_per_day.values, name='Neutral'))
    fig.update_layout(title_text='Amazon Reviews Time Series Plot (Sentiment)', xaxis_title='Date', yaxis_title='Number of Comments')
    # fig.write_image('./static/youtube/time_series_sentiment.png')
    fig.write_html('./static/youtube/time_series_sentiment.html')

    # Create the time series plot for reviews
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=reviews_per_day.index, y=reviews_per_day.values, name='Reviews'))
    fig.update_layout(title_text='Amazon Reviews Time Series Plot', xaxis_title='Date', yaxis_title='Number of Reviews')
    # fig.write_image('./static/youtube/time_series.png')
    fig.write_html('./static/youtube/time_series.html')


