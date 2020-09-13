
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

class ClassifierService:

    def construct_classifier(self):

        # a dummy function that just returns its input
        def identity(x):
            return x

        # let's use the TF-IDF vectorizer
        tfidf = True

        # we use a dummy function as tokenizer and preprocessor,
        # since the texts are already preprocessed and tokenized.
        if tfidf:
            vec = TfidfVectorizer(preprocessor=identity,
                                  tokenizer=identity)
        else:
            vec = CountVectorizer(preprocessor=identity,
                                  tokenizer=identity)

        # combine the vectorizer with a Naive Bayes classifier
        classifier = Pipeline([('vec', vec),
                               ('cls', MultinomialNB(alpha=1.0, fit_prior=True))])
        return classifier

    # a dummy function that just returns its input
    def identity(self, x):
        return x