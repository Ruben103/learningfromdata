from DataService import DataService
from SupportVectorMachine import SVM
from sklearn.metrics import accuracy_score, f1_score

from numpy import arange, mean
from datetime import datetime

class Experiments:

    def experimentDefaultSetting(self, trainset, testset):
        print("Reading data")
        x, y = DataService().read_corpus(trainset)
        clf = SVM().construct_classifier("linear", 1.0)

        # Vectorize the text data and return an (n_samples, n_features) matrix.
        x_vec = DataService().vectorize_input(x)
        conversion_dict, y = DataService().labels_string_to_float(y)

        x_train, y_train, x_dev, y_dev, x_test, y_test = DataService().test_dev_train_split(x_vec, y)
        x_dev_train, y_dev_train, x_dev_test, y_dev_test = DataService().test_train_split(x_dev,  y_dev)

        start_time = datetime.utcnow()
        print('Fitting training data on', len(x_dev_train), 'Samples')
        clf.fit(x_dev_train, y_dev_train)

        training_time = (datetime.utcnow() - start_time).seconds
        print("Training took", training_time, 'seconds..')

        y_pred = clf.predict(x_dev_test)
        print("Accuracy score:", accuracy_score(y_pred=y_pred, y_true=y_dev_test))
        print("F1 score (macro):", f1_score(y_pred=y_pred, y_true=y_dev_test, average='macro'))

    def experimentCrossValidation(self, trainset, testset):
        print("Reading data")
        x, y = DataService().read_corpus(trainset)
        clf = SVM().construct_classifier("linear", 1.0)

        # Vectorize the text data and return an (n_samples, n_features) matrix.
        x_vec = DataService().vectorize_input(x)
        conversion_dict, y = DataService().labels_string_to_float(y)

        x_train, y_train, x_dev, y_dev, x_test, y_test = DataService().test_dev_train_split(x_vec, y)
        x_dev_train, y_dev_train, x_dev_test, y_dev_test = DataService().test_train_split(x_dev, y_dev)

        dev_sets = DataService().cross_validation_split(x_dev_train, y_dev_train)

        cv_results ={}
        for C in arange(0.5, 1, 0.1):
            print("\nProcessing C:", C)
            average_score = []
            for set in dev_sets:
                clf = SVM().construct_classifier(C=C)
                validation_set = set
                union_set = DataService().construct_union_set(set.copy(), dev_sets.copy())

                # fit on the rest of the data
                clf.fit(union_set[0], union_set[1])

                # validate on validation set
                y_pred = clf.predict(validation_set[0])

                score = f1_score(y_true=validation_set[1], y_pred=y_pred, average='binary')
                average_score.append(score)
            cv_results[C] = mean(average_score)
            print("Average F1 score for C:", str(C) + ".", mean(average_score))

        print(cv_results)
