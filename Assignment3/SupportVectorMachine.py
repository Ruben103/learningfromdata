from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn import svm

class SVM:

    def construct_classifier(self, kernel):
        if kernel == "linear":
            clf = svm.SVC(kernel="linear")
            return clf
