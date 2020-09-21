from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

class ClassifierService:

    def construct_decisiontreeclassifier(self, criterion='entropy', max_depth=None, min_samples_spit=2):
        classifier = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth,
                                            min_samples_split=min_samples_spit)
        return classifier

    def construct_knearest_neighboursclassifier(self, k):

        classifier = KNeighborsClassifier(n_neighbors=k)
        return classifier

    def construct_multinomial(self, vec):
        # combine the vectorizer with a Naive Bayes classifier
        classifier = Pipeline([('vec', vec),
                               ('cls', MultinomialNB(alpha=1.0, fit_prior=True))])
        return classifier

    def vectorize_input(self, x, y):
        def identity(x):
            return x

        vec = TfidfVectorizer(preprocessor=identity, tokenizer=identity)
        x_vec = vec.fit_transform(x)
        y_vec = vec.fit_transform(y)

        return x_vec.toarray(), y_vec.toarray()
