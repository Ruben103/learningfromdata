from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn import svm

class SupportVectorMachine:

    def construct_classifier(self):
        clf = svm.SVC()
        return clf