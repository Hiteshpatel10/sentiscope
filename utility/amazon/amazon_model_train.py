import pandas as pd
import joblib
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def amazonModelTrain():

    df = pd.read_csv('../../data/am1-train.csv')
    print(len(df))

    sentiments = SentimentIntensityAnalyzer()
    df["positive"] = df['content'].apply(lambda review: sentiments.polarity_scores(str(review))["pos"])
    df["negative"] = df['content'].apply(lambda review: sentiments.polarity_scores(str(review))["neg"])
    df["neutral"] = df['content'].apply(lambda review: sentiments.polarity_scores(str(review))["neu"])
    df["compound"] = df['content'].apply(lambda review: sentiments.polarity_scores(str(review))["compound"])
    
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
    x = vectorizer.fit_transform(df['content'].apply(lambda x: np.str_(x)))
    x_train ,x_test,y_train,y_test=train_test_split(x,df['sentiment'], test_size=.30 , random_state=42)

    # Train Logistic Regression Model
    lr_model=LogisticRegression()
    lr_model.fit(x_train,y_train)
    lr_pred=lr_model.predict(x_test)
    print("Logistic Regression Model Accuracy:", accuracy_score(y_test, lr_pred))
    joblib.dump(lr_model, "../../model/amazon/amazon-logistic.joblib")  
    joblib.dump(vectorizer, "../../model/amazon/amazon-vector-logistic.joblib")

    # Train KNN Model
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(x_train, y_train)
    knn_pred = knn_model.predict(x_test)
    print("KNN Model Accuracy:", accuracy_score(y_test, knn_pred))
    joblib.dump(knn_model, "../../model/amazon/amazon-knn.joblib")  
    joblib.dump(vectorizer, "../../model/amazon/amazon-vector-knn.joblib")

    # Train SVM Model
    svm_model = SVC(kernel='linear')
    svm_model.fit(x_train, y_train)
    svm_pred = svm_model.predict(x_test)
    print("SVM Model Accuracy:", accuracy_score(y_test, svm_pred))
    joblib.dump(svm_model, "../../model/amazon/amazon-svm.joblib")  
    joblib.dump(vectorizer, "../../model/amazon/amazon-vector-svm.joblib")

if __name__ == "__main__":
    amazonModelTrain()
