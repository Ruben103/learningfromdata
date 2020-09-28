from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn import svm

class SVM:

    def construct_classifier(self, kernel='linear', C=1.0):
        clf = svm.SVC(kernel=kernel, C=C)
        return clf

    def construct_rbf_classifier(self, kernel='rbf', gamma=0.7, C=1.0):
        clf = svm.SVC(kernel='rbf', gamma=gamma, C=C)
        return clf