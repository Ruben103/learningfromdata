from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn import svm

class SVM:

    def construct_classifier(self, kernel):
        if kernel == "linear":
            clf = svm.SVC(kernel="linear")
            return clf

    def vectorize_input(self, x, y):
        def identity(x):
            return x

        vec = TfidfVectorizer(preprocessor=identity, tokenizer=identity)
        x_vec = vec.fit_transform(x)
        y_vec = vec.fit_transform(y)

        return x_vec.toarray(), y_vec.toarray()