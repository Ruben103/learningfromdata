"""
The functions below describe the experiments we have used.
They are individually callable.
"""

from ClassifierService import ClassifierService
from DataService import DataService
from OutputService import PrintScores
from sklearn.naive_bayes import MultinomialNB

class Experiments:

    def experiment_binary(self):
        """
        The code below runs the experiment for the binary classification problem
        """
        x, y = DataService().read_corpus('trainset.txt', use_sentiment=True)

        """
        OUR COMMENT:
        A splitpoint variable is used to divide the whole dataset into 75% training and 25% test sets.
        """
        x_train, y_train, x_test, y_test = DataService().test_train_split(x, y)

        classifier = ClassifierService().construct_classifier()

        """
        OUR COMMENT for classifier.fit:
        The classifier object is a Pipeline object from the scikit-learn package.
        From the documentation: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html,
        we observe that this object is used to apply a number of transforms using an estimator.

        From the object we see that the final_estimator is a MultinomialNB, which is a Naive Bayes Estimator for multinomial
        objects (the sentences have more than one words).

        The fit functions fits each pattern (combination of one sentence from the set Xtrain and label from the set Ytrain)
        using the MultinomialNB estimator.

        There is no output to the function. However, it modifies the object classifier such that its parameters are trained
        """
        classifier.fit(x_train, y_train)

        """
        OUR COMMENT for classifier.predict:
        After the Pipeline object has fitted on the training data, we call .predict to predict on the test set Xtest.
        The object will perform a forward pass of the test data without updating its network paramaters.

        The function takes as input a vector of input. In this case a list of sentences.

        The output of the network is a vector of sentiment labels. In this case, the size of the output is 1500
        """
        y_pred = classifier.predict(x_test)

        print("\nPrinting scores for binary problem")
        PrintScores().print_precision_score(y_test=y_test, y_pred=y_pred)
        PrintScores().print_recall_score(y_test=y_test, y_pred=y_pred)
        PrintScores().print_f1_score(y_test=y_test, y_pred=y_pred)

        print("\nPrinting accuracy score")
        PrintScores().print_accuracy_score(y_test=y_test, y_pred=y_pred)

        PrintScores().print_confusion_matrix(y_test=y_test, y_pred=y_pred)

    def experiment_multi_class(self):
        x, y = DataService().read_corpus('trainset.txt', use_sentiment=False)

        x_train, y_train, x_test, y_test = DataService().test_train_split(x, y)

        classifier = ClassifierService().construct_classifier()

        classifier.fit(x_train, y_train)

        y_pred = classifier.predict(x_test)

        print("\nPrinting scores for multi-class problem")
        PrintScores().print_precision_score(y_test=y_test, y_pred=y_pred)
        PrintScores().print_recall_score(y_test=y_test, y_pred=y_pred)
        PrintScores().print_f1_score(y_test=y_test, y_pred=y_pred)

        print("\nPrinting accuracy score")
        PrintScores().print_accuracy_score(y_test=y_test, y_pred=y_pred)

        PrintScores().print_confusion_matrix(y_test=y_test, y_pred=y_pred)

    def experiment_probabilities_multi(self):
        x, y = DataService().read_corpus('trainset.txt', use_sentiment=False)
        x_train, y_train, x_test, y_test = DataService().test_train_split(x, y)
        classifier = ClassifierService().construct_classifier()
        classifier.fit(x_train, y_train)
        params = classifier.get_params(deep = True)
        y_pred_prob = classifier.predict_proba(x_test)
        print("bug stop")

        print("\nPosterior probabilities multi-class:")
        print('\t', y_pred_prob)

    def experiment_probabilities_binary(self):
        x, y = DataService().read_corpus('trainset.txt', use_sentiment=True)
        x_train, y_train, x_test, y_test = DataService().test_train_split(x, y)
        classifier = ClassifierService().construct_classifier()
        classifier.fit(x_train, y_train)
        y_pred_prob = classifier.predict_proba(x_test)
        y_pred_class = classifier.predict(x_test)
        param = classifier.get_params()

        print("\nPosterior probabilities binary")
        print('\t', y_pred_prob)
