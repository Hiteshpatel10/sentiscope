import pandas as pd
import joblib
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score

# Load the saved model and vectorizer
model = joblib.load('../../model/amazon/amazon-logistic.joblib')
vectorizer = joblib.load('../../model/amazon/amazon-vector-logistic.joblib')

# Load the test data
df_test = pd.read_csv('../../data/amazon-test.csv')

# Preprocess the test data
sentiments = SentimentIntensityAnalyzer()
df_test["positive"] = df_test['content'].apply(lambda review: sentiments.polarity_scores(str(review))["pos"])
df_test["negative"] = df_test['content'].apply(lambda review: sentiments.polarity_scores(str(review))["neg"])
df_test["neutral"] = df_test['content'].apply(lambda review: sentiments.polarity_scores(str(review))["neu"])
df_test["compound"] = df_test['content'].apply(lambda review: sentiments.polarity_scores(str(review))["compound"])

score = df_test["compound"].values
sentiment = []
for i in score:
    if i >= 0.05 :
        sentiment.append(1)
    elif i <= -0.05 :
        sentiment.append(-1)
    else:
        sentiment.append(0)
df_test["sentiment"] = sentiment

# Transform the test data
x_test = vectorizer.transform(df_test['content'].apply(lambda x: np.str_(x)))

# Make predictions
y_pred = model.predict(x_test)

# Calculate accuracy
accuracy = accuracy_score(df_test['sentiment'], y_pred)
print("Accuracy:", accuracy)
