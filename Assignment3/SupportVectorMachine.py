from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC

class SVM:

    def construct_classifier(self, kernel='linear', C=1.0):
        clf = SVC(kernel=kernel, C=C)
        return clf

    def construct_rbf_classifier(self, kernel='rbf', gamma=0.7, C=1.0):
        clf = SVC(kernel='rbf', gamma=gamma, C=C)
        return clf

    def construct_linear_classifier(self, penalty='l2', C=1.0):
        clf = LinearSVC(penalty=penalty, C=C, loss='squared-hinge', dual=True)
        return clf

    def construct_best_classifier(self, kernel='linear', gamma=0.75, C=1.5):
        clf = SVC(kernel=kernel, gamma=gamma, C=C)
        return clf