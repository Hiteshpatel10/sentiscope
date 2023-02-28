import pandas as pd
import joblib
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('data/yt-train.csv')

sentiments = SentimentIntensityAnalyzer()
df["positive"] = df['Comment'].apply(lambda review: sentiments.polarity_scores(str(review))["pos"])
df["negative"] = df['Comment'].apply(lambda review: sentiments.polarity_scores(str(review))["neg"])
df["neutral"] = df['Comment'].apply(lambda review: sentiments.polarity_scores(str(review))["neu"])
df["compound"] = df['Comment'].apply(lambda review: sentiments.polarity_scores(str(review))["compound"])

score = df["compound"].values
sentiment = []
for i in score:
    if i >= 0.05 :
        sentiment.append(1)
    elif i <= -0.05 :
        sentiment.append(-1)
    else:
        sentiment.append(0)
df["sentiment"] = sentiment

vectorizer = TfidfVectorizer(max_features=2500)
x = vectorizer.fit_transform(df['Comment'].apply(lambda x: np.str_(x)))
x_train ,x_test,y_train,y_test=train_test_split(x,df['sentiment'], test_size=.30 , random_state=42)

model=LogisticRegression()

#Model fitting 
model.fit(x_train,y_train)

#testing the model
pred=model.predict(x_test)

#model accuracy
print(accuracy_score(y_test,pred))

joblib.dump(model, "../../model/youtube/yt-logistic.joblib")  
joblib.dump(vectorizer, "../../model/youtube/yt-vector-logistic.joblib")

