import pandas as pd
import joblib
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score

# Load the saved models and vectorizers
logistic_model = joblib.load('../../model/amazon/amazon-logistic.joblib')
logistic_vectorizer = joblib.load('../../model/amazon/amazon-vector-logistic.joblib')

knn_model = joblib.load('../../model/amazon/amazon-knn.joblib')
knn_vectorizer = joblib.load('../../model/amazon/amazon-vector-knn.joblib')

svm_model = joblib.load('../../model/amazon/amazon-svm.joblib')
svm_vectorizer = joblib.load('../../model/amazon/amazon-vector-svm.joblib')

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

# Transform the test data for each model
x_test_logistic = logistic_vectorizer.transform(df_test['content'].apply(lambda x: np.str_(x)))
x_test_knn = knn_vectorizer.transform(df_test['content'].apply(lambda x: np.str_(x)))
x_test_svm = svm_vectorizer.transform(df_test['content'].apply(lambda x: np.str_(x)))

# Make predictions for each model
y_pred_logistic = logistic_model.predict(x_test_logistic)
y_pred_knn = knn_model.predict(x_test_knn)
y_pred_svm = svm_model.predict(x_test_svm)

# Calculate accuracy for each model
accuracy_logistic = accuracy_score(df_test['sentiment'], y_pred_logistic)
accuracy_knn = accuracy_score(df_test['sentiment'], y_pred_knn)
accuracy_svm = accuracy_score(df_test['sentiment'], y_pred_svm)

# Print the accuracies for each model
print("Logistic Regression Accuracy:", accuracy_logistic)
print("KNN Accuracy:", accuracy_knn)
print("SVM Accuracy:", accuracy_svm)
