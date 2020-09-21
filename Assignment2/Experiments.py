from DataService import DataService
from ClassifierService import ClassifierService
from sklearn.metrics import accuracy_score, f1_score
from numpy import mean, inf



class Experiments:

    def experiment1(self, trainset, testset):

        x, y = DataService().read_corpus(trainset)
        x_train, y_train, x_dev, y_dev = DataService().test_train_split(x, y)
        x_tr_vec, y_tr_vec = ClassifierService().vectorize_input(x, y)
        x_tr_vec, y_tr_vec, x_dev_vec, y_dev_vec = DataService().test_train_split(x_tr_vec, y_tr_vec)

        x_test, y_test = DataService().read_corpus(testset)
        x_test_vec, y_test_vec = ClassifierService().vectorize_input(x_test, y_test)

        sets = DataService().cross_validation_split(x_tr_vec, y_train)

        max_accuracy = -inf
        best_model = None
        for k in [25, 30, 35, 40, 45, 50]:
            clf = ClassifierService().construct_knearest_neighboursclassifier(k=k)
            accuracies = []
            for eval_set in sets:

                for train_set in sets:
                    if eval_set[1] == train_set[1]:
                        x_s = train_set[0]; y_s = train_set[1]
                        clf.fit(x_s, y_s)
                y_pred = clf.predict(eval_set[0])
                accuracies.append(f1_score(y_true=eval_set[1], y_pred=y_pred,average='macro'))

            if mean(accuracies) > max_accuracy:
                print("\nNew best model found!")
                print("\nMean F1-score:", mean(accuracies))
                best_model = clf
        try:
            predictions = best_model.predict(x_test_vec)
            print("Accuracy:", f1_score(predictions, y_test, average='macro'))
        except NotImplementedError:
            print("model is None")

