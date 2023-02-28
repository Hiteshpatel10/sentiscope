import pandas as pd

# For ploting graph
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
import plotly.graph_objs as go
import plotly.express as px

#nltk
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
from wordcloud import WordCloud

# model save
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

    df = pd.read_csv('./data/am1-train.csv', parse_dates=['date'])
    df.dropna(subset=['content'], inplace=True)
    filename = "./model/amazon/amazon-logistic.joblib"
    model = joblib.load(filename)
    vectorizer = joblib.load("./model/amazon/amazon-vector-logistic.joblib")


    df['sentiment'] = df['content'].apply(lambda x: predict_sentiment(model, vectorizer, x))

    # Splitting the data based on sentiment
    positive = df[df['sentiment'] == 1]
    negative = df[df['sentiment'] == -1]
    neutral = df[df['sentiment'] == 0]

    # Review no histogram
    fig = px.histogram(df, x="rating")
    fig.update_traces(marker_color="turquoise",marker_line_color='rgb(8,48,107)',marker_line_width=1.5)
    fig.update_layout(title_text='Product Score')
    # pio.write_image(fig, 'product_sce.png')
    fig.write_html('./static/amazon/product_score_histogram.html')


    #sentimetnt histogram
    # Counting the number of comments in each category
    counts = {'Negative': len(negative), 'Neutral': len(neutral), 'Positive': len(positive),}
    # Creating a bar chart
    fig = px.bar(x=list(counts.keys()), y=list(counts.values()), color=list(counts.keys()))
    # Customizing the chart
    fig.update_traces(marker_line_width=1.5)
    fig.update_layout(title_text='Sentiment Analysis Results', xaxis_title='Sentiment Category', yaxis_title='Comment Count')
    # Saving the chart as an HTML file
    fig.write_html('./static/amazon/sentiment_histogram.html')



    # Create stopword list:
    stopwords = set(STOPWORDS)
    stopwords.update(["br", "href", "good", "great", "phone", "mobile"])

    if len(positive) > 0:
        pos = " ".join(str(review) for review in positive.content)
        wordcloud2 = WordCloud(stopwords=stopwords).generate(pos)
        plt.imshow(wordcloud2, interpolation='bilinear')
        plt.axis("off")
        plt.savefig('./static/amazon/wordcloud_positive.png', transparent=True)

    if len(negative) > 0:
        neg = " ".join(str(review) for review in negative.content)
        wordcloud3 = WordCloud(stopwords=stopwords).generate(neg)
        plt.imshow(wordcloud3, interpolation='bilinear')
        plt.axis("off")
        plt.savefig('./static/amazon/wordcloud_negative.png', transparent=True)

    if len(neutral) > 0:
        neu = " ".join(str(review) for review in neutral.content)
        wordcloud4 = WordCloud(stopwords=stopwords).generate(neu)
        plt.imshow(wordcloud4, interpolation='bilinear')
        plt.axis("off")
        plt.savefig('./static/amazon/wordcloud_neutral.png', transparent=True)


    # Group the reviews by date and count the number of reviews on each day for each sentiment
    pos_reviews_per_day = positive.groupby(positive['date'].dt.date)['content'].count()
    neg_reviews_per_day = negative.groupby(negative['date'].dt.date)['content'].count()
    neu_reviews_per_day = neutral.groupby(neutral['date'].dt.date)['content'].count()

    # Group the reviews by date and count the number of reviews on each day
    reviews_per_day = df.groupby(df['date'].dt.date)['content'].count()



    # Create the time series plot for each sentiment
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pos_reviews_per_day.index, y=pos_reviews_per_day.values, name='Positive'))
    fig.add_trace(go.Scatter(x=neg_reviews_per_day.index, y=neg_reviews_per_day.values, name='Negative'))
    fig.add_trace(go.Scatter(x=neu_reviews_per_day.index, y=neu_reviews_per_day.values, name='Neutral'))
    fig.update_layout(title_text='Amazon Reviews Time Series Plot (Sentiment)', xaxis_title='Date', yaxis_title='Number of contents')
    # fig.write_image('./static/amazon/time_series_sentiment.png')
    fig.write_html('./static/amazon/time_series_sentiment.html')

    # Create the time series plot for reviews
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=reviews_per_day.index, y=reviews_per_day.values, name='Reviews'))
    fig.update_layout(title_text='Amazon Reviews Time Series Plot', xaxis_title='Date', yaxis_title='Number of Reviews')
    # fig.write_image('./static/amazon/time_series.png')
    fig.write_html('./static/amazon/time_series.html')

