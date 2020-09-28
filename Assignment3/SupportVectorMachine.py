from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn import svm

class SVM:

    def construct_classifier(self, kernel='linear', C=1.0):
        if kernel == 'linear':
            clf = svm.SVC(kernel='linear', C=C)
            return clf
