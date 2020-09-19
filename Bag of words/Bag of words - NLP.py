# Natural Language Processing

# Importing the libraries
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def performance(cm):
    tp = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tn = cm[1][1]
    accuracy = (tp + tn)/(tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)
    print("Accuracy : ", accuracy)
    print("Precision : ", precision)
    print("Recall : ", recall)
    print("F1 Score : ", f1_score)


def feature_scaling(X_train, X_test):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test


def naive_bayes(X_train, X_test, y_test, y_train):
    # Training the Naive Bayes model on the Training set
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nNaive Bayes\n")
    print(cm)
    performance(cm)


def decision_tree(X_train, X_test, y_test, y_train):
    # Feature Scaling
    X_train, X_test = feature_scaling(X_train, X_test)

    # Training the Decision Tree Classification model on the Training set
    classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    # print(np.concatenate((y_pred.reshape(len(y_pred), 1),
    #                       y_test.reshape(len(y_test), 1)), 1))

    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nDecision Tree\n")
    print(cm)
    performance(cm)


def kernel_svm(X_train, X_test, y_test, y_train):
    # Feature Scaling
    X_train, X_test = feature_scaling(X_train, X_test)

    # Training the Kernel SVM model on the Training set
    classifier = SVC(kernel='rbf', random_state=0)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    # print(np.concatenate((y_pred.reshape(len(y_pred), 1),
    #                       y_test.reshape(len(y_test), 1)), 1))

    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nKernel SVM\n")
    print(cm)
    performance(cm)


def random_forest(X_train, X_test, y_test, y_train):
    # Feature Scaling
    X_train, X_test = feature_scaling(X_train, X_test)

    # Training the Kernel SVM model on the Training set
    classifier = RandomForestClassifier(
        n_estimators=10, criterion='entropy', random_state=0)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    # print(np.concatenate((y_pred.reshape(len(y_pred), 1),
    #                       y_test.reshape(len(y_test), 1)), 1))

    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nRandom Forest\n")
    print(cm)
    performance(cm)


def main():
    # Importing the dataset
    dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

    # Cleaning the texts
    nltk.download('stopwords')
    corpus = []
    for i in range(0, 1000):
        review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        all_stopwords = stopwords.words('english')
        all_stopwords.remove('not')
        review = [ps.stem(word)
                  for word in review if not word in set(all_stopwords)]
        review = ' '.join(review)
        corpus.append(review)

    # Creating the Bag of Words model
    cv = CountVectorizer(max_features=1500)
    X = cv.fit_transform(corpus).toarray()
    y = dataset.iloc[:, 1].values

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=0)

    random_forest(X_train, X_test, y_test, y_train)
    kernel_svm(X_train, X_test, y_test, y_train)
    naive_bayes(X_train, X_test, y_test, y_train)
    decision_tree(X_train, X_test, y_test, y_train)


if __name__ == "__main__":
    main()
