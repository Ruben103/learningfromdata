from DataService import DataService
from SupportVectorMachine import SVM
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


from numpy import arange, mean, inf
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
        clf.fit(x_train, y_train)
        non_zero = []

        y_pred = clf.predict(x_dev_test)
        print("Accuracy score:", accuracy_score(y_pred=y_pred, y_true=y_dev_test))
        print("F1 score (macro):", f1_score(y_pred=y_pred, y_true=y_dev_test, average='macro'))

    def experimentCParamterCrossValidation(self, trainset, testset):
        print("Reading data")
        x, y = DataService().read_corpus(trainset)
        clf = SVM().construct_classifier("linear", 1.0)

        # Vectorize the text data and return an (n_samples, n_features) matrix.
        x_vec = DataService().vectorize_input(x)
        conversion_dict, y = DataService().labels_string_to_float(y)

        x_train, y_train, x_dev, y_dev, x_test, y_test = DataService().test_dev_train_split(x_vec, y)
        x_dev_train, y_dev_train, x_dev_test, y_dev_test = DataService().test_train_split(x_dev, y_dev)

        dev_sets = DataService().cross_validation_split(x_train, y_train)

        best_accuracy = -inf
        best_classifier = None

        cv_results ={}
        for C in arange(0.5, 1.1, 0.1):
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
            print("Average F1 score for C:", str(C) + ".", round(mean(average_score)), 2)

            score = round(mean(average_score), 3)
            # save the best model and use that to classify the testset
            if score > best_accuracy:
                best_accuracy = score
                best_classifier = clf

        y_pred = best_classifier.predict(x_test)
        print("F1 score (macro):", f1_score(y_pred=y_pred, y_true=y_test, average='macro'))

    def experimentCombinatorialCrossValidation(self, trainset, testset):
        print("Reading data")
        x, y = DataService().read_corpus(trainset)
        clf = SVM().construct_classifier("linear", 1.0)

        # Vectorize the text data and return an (n_samples, n_features) matrix.
        x_vec = DataService().vectorize_input(x)
        conversion_dict, y = DataService().labels_string_to_float(y)

        x_train, y_train, x_dev, y_dev, x_test, y_test = DataService().test_dev_train_split(x_vec, y)
        x_dev_train, y_dev_train, x_dev_test, y_dev_test = DataService().test_train_split(x_dev, y_dev)

        dev_sets = DataService().cross_validation_split(x_train, y_train)

        best_accuracy = -inf
        best_classifier = None

        cv_results ={}
        for gamma in arange(0.5, 1.4, 0.15):
            for C in arange(0.5, 2.1, 0.25):
                print("\nProcessing Gamma:", gamma, "C:", C)
                average_score = []
                for set in dev_sets:
                    clf = SVM().construct_rbf_classifier(kernel='rbf', gamma=gamma, C=C)
                    validation_set = set
                    union_set = DataService().construct_union_set(set.copy(), dev_sets.copy())

                    # fit on the rest of the data
                    clf.fit(union_set[0], union_set[1])

                    # validate on validation set
                    y_pred = clf.predict(validation_set[0])

                    score = f1_score(y_true=validation_set[1], y_pred=y_pred, average='binary')
                    average_score.append(score)
                score = round(mean(average_score), 3)
                cv_results[[C, gamma]] = score
                print("Average F1 score for C:", str(C) + ".", score)

                # save the best model and use that to classify the testset
                if score > best_accuracy:
                    best_accuracy = score
                    best_classifier = clf

        y_pred = best_classifier.predict(x_test)
        print("F1 score (macro):", f1_score(y_pred=y_pred, y_true=y_test, average='macro'))

    def experimentLinearKernel(self, trainset, testset):
        print("Reading data")
        x, y = DataService().read_corpus(trainset)

        # Vectorize the text data and return an (n_samples, n_features) matrix.
        x_vec = DataService().vectorize_input(x)
        conversion_dict, y = DataService().labels_string_to_float(y)

        x_train, y_train, x_dev, y_dev, x_test, y_test = DataService().test_dev_train_split(x_vec, y)
        x_dev_train, y_dev_train, x_dev_test, y_dev_test = DataService().test_train_split(x_dev, y_dev)

        dev_sets = DataService().cross_validation_split(x_dev_train, y_dev_train)

        best_accuracy = -inf
        best_classifier = None

        cv_results1 = {}
        cv_results2 = {}
        for C in arange(0.5, 2.25, 0.25):
            print("\nProcessing C:", C)
            average_score1 = []
            average_score2 = []
            for set in dev_sets:
                clf2 = SVM().construct_linear_classifier(penalty='l2', C=C)
                validation_set = set
                union_set = DataService().construct_union_set(set.copy(), dev_sets.copy())

                # fit on the rest of the data
                clf2.fit(union_set[0], union_set[1])

                # validate on validation set
                y_pred = clf2.predict(validation_set[0])

                score = f1_score(y_true=validation_set[1], y_pred=y_pred, average='binary')
                average_score1.append(score)
            cv_results1[C] = mean(average_score1)

            score = round(mean(average_score2), 3)
            print("Average F1 score for CLF1:", round(mean(average_score1), 3))
            print("Average F1 score for CLF2:", round(mean(average_score2), 3))

            # save the best model and use that to classify the testset
            if score > best_accuracy:
                best_accuracy = score
                best_classifier = clf2

        y_pred = best_classifier.predict(x_test)
        print("F1 score (macro):", f1_score(y_pred=y_pred, y_true=y_test, average='macro'))

    def experimentBestModel(self, trainset, testset):
        print("Reading data")
        x, y = DataService().read_corpus(trainset)
        clf = SVM().construct_best_classifier()

        # Vectorize the text data and return an (n_samples, n_features) matrix.
        x_vec = DataService().vectorize_input(x)
        conversion_dict, y = DataService().labels_string_to_float(y)

        x_train, y_train, x_dev, y_dev, x_test, y_test = DataService().test_dev_train_split(x_vec, y)

        dev_sets = DataService().cross_validation_split(x_train, y_train)

        best_accuracy = -inf
        best_classifier = None

        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)

        print("F1 score (macro):", f1_score(y_pred=y_pred, y_true=y_test, average='macro'))

    def experimentFeatures(self, trainset, testset):
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
        non_zero = []
        training_time = (datetime.utcnow() - start_time).seconds
        print("Training took", training_time, 'seconds..')

        y_pred = clf.predict(x_dev_test)
        print("Accuracy score:", accuracy_score(y_pred=y_pred, y_true=y_dev_test))
        print("F1 score (macro):", f1_score(y_pred=y_pred, y_true=y_dev_test, average='macro'))

        coef = clf.coef_
        def identity(x):
            return x
        vec = TfidfVectorizer(preprocessor=identity, tokenizer=identity)
        vec.fit_transform(x)
        names = vec.get_feature_names()
        coefs_and_features = list(zip(coef[0], names))
        list_sorted_pos = sorted(coefs_and_features, key=lambda x: x[0], reverse=True)
        list_sorted_neg = sorted(coefs_and_features, key=lambda x: x[0])
        features = []
        for i in range(100):
            features.append(list_sorted_pos[i][1])
        for i in range(100):
            features.append(list_sorted_neg[i][1])

        new_data = DataService().get_features_from_data(x, features)

        clf2 = SVM().construct_classifier("linear", 1.0)
        # Vectorize the text data and return an (n_samples, n_features) matrix.
        x_vec = DataService().vectorize_input(new_data)
        conversion_dict, y = DataService().labels_string_to_float(y)
        x_train, y_train, x_dev, y_dev, x_test, y_test = DataService().test_dev_train_split(x_vec, y)
        x_dev_train, y_dev_train, x_dev_test, y_dev_test = DataService().test_train_split(x_dev, y_dev)
        start_time = datetime.utcnow()
        print('Fitting training data on', len(x_dev_train), 'Samples')
        clf2.fit(x_dev_train, y_dev_train)
        non_zero = []

        training_time = (datetime.utcnow() - start_time).seconds
        print("Training took", training_time, 'seconds..')

        y_pred = clf2.predict(x_dev_test)
        print("Accuracy score:", accuracy_score(y_pred=y_pred, y_true=y_dev_test))
        print("F1 score (macro):", f1_score(y_pred=y_pred, y_true=y_dev_test, average='macro'))


