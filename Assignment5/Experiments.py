from ClassifierService import Classifier
from Data import Data
from numpy import arange
from pandas import DataFrame
from sklearn.metrics import precision_score, recall_score, f1_score

class Experiments:

    def experimentDefault(self, args):
        # Load data
        X_train, X_dev, X_test, Y_train, Y_dev, classes = Data().load_data()
        nb_features = X_train.shape[1]
        print(nb_features, 'features')
        nb_classes = Y_train.shape[1]
        print(nb_classes, 'classes')

        model = Classifier(type='MLP', nb_features=nb_features, nb_classes=nb_classes, epochs=5, batch_size=50, classes=classes, run_number=args.run)

        model.fit(X_train=X_train, Y_train=Y_train, X_dev=X_dev, Y_dev=Y_dev)
        model.predict(X_test=X_test, X_dev=X_dev, Y_dev=Y_dev)

    def experimentDropout(self, args):
        # Load data
        X_train, X_dev, X_test, Y_train, Y_dev, classes = Data().load_data()
        nb_features = X_train.shape[1]
        print(nb_features, 'features')
        nb_classes = Y_train.shape[1]
        print(nb_classes, 'classes')

        args.epochs = 200

        model = None

        folds = Data().cross_validation_split(X_train, Y_train)
        Metrics_per_set = DataFrame()
        count = 0
        for rate in arange(0.1, 1, 0.1):
            for fold in folds:
                print("Training holding fold", str(count), "out..")
                union_set = Data().construct_union_set(fold.copy(), folds.copy())

                model = Classifier(type='Dropout', nb_features=nb_features, nb_classes=nb_classes, epochs=args.epochs,
                                   batch_size=64,
                                   classes=classes, run_number=args.run, rate=rate)

                model.fit(X_train=union_set[0], Y_train=union_set[1], X_dev=fold[0], Y_dev=fold[1])
                count += 1


            if model is not None:
                accuracies = model.predict(X_test=X_dev, X_dev=None, Y_dev=Y_dev)
                Metrics_per_set["rate" + str(rate)] = accuracies




