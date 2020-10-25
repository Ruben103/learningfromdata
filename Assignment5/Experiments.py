from ClassifierService import Classifier
from Data import Data
from numpy import arange
from pandas import DataFrame, read_json
from sklearn.metrics import precision_score, recall_score, f1_score


class Experiments:

    def expCorpus(self):

        print("Readin data")
        file = open("Merged_text.txt", 'r')
        data = file.read()
        words = data.split()

        print()
        print("STOP HERE NOW")

    def experimentDefault(self, args):
        # Load data
        X_train, X_dev, X_test, Y_train, Y_dev, classes = Data().load_data()
        nb_features = X_train.shape[1]
        print(nb_features, 'features')
        nb_classes = Y_train.shape[1]
        print(nb_classes, 'classes')

        model = Classifier(type='MLP', nb_features=nb_features, nb_classes=nb_classes, epochs=5, batch_size=50,
                           classes=classes, run_number=args.run)

        model.fit(X_train=X_train, Y_train=Y_train, X_dev=X_dev, Y_dev=Y_dev)
        model.predict(X_test=X_test, X_dev=X_dev, Y_dev=Y_dev)

    def experimentBatchsize(self, args):
        # Load data
        X_train, X_dev, X_test, Y_train, Y_dev, classes = Data().load_data()
        nb_features = X_train.shape[1]
        print(nb_features, 'features')
        nb_classes = Y_train.shape[1]
        print(nb_classes, 'classes')

        args.epochs = 100

        model = None

        folds = Data().cross_validation_split(X_train, Y_train)
        metrics_per_set = DataFrame()
        for BS in [8, 16, 24, 32]:
            print("\nEvaluating batch size ", str(BS))
            count = 0
            mean_of_folds = DataFrame()
            for fold in folds:
                print("Training holding fold", str(count), "out..")
                if count == len(folds) - 1:
                    early_stopping_fold = folds[0]
                else:
                    early_stopping_fold = folds[count + 1]

                union_set = Data().construct_union_set(fold.copy(), early_stopping_fold.copy(), folds.copy())

                model = Classifier(type='DropoutAdam', nb_features=nb_features, nb_classes=nb_classes,
                                   epochs=args.epochs,
                                   batch_size=BS,
                                   classes=classes, run_number=args.run)

                model.fit(X_train=union_set[0], Y_train=union_set[1], X_dev=early_stopping_fold[0],
                          Y_dev=early_stopping_fold[1])
                mean_of_folds["Fold " + str(count + 1)] = model.predict(X_test=fold[0], X_dev=fold[0], Y_dev=fold[1])
                count += 1

            if model is not None:
                metrics_per_set["BS: " + str(BS)] = mean_of_folds.mean(axis=1)
        # metrics_per_set['cols'] = ['acc', 'prec', 'rec', 'f1']
        print(metrics_per_set)
        print(metrics_per_set.idxmax(axis=1))

    def experimentDropoutRate(self, args):
        # Load data
        X_train, X_dev, X_test, Y_train, Y_dev, classes = Data().load_data()
        nb_features = X_train.shape[1]
        print(nb_features, 'features')
        nb_classes = Y_train.shape[1]
        print(nb_classes, 'classes')

        args.epochs = 100

        model = None

        folds = Data().cross_validation_split(X_train, Y_train)
        metrics_per_set = DataFrame()
        for rate in arange(0.4, 0.9, 0.1):
            print("\nEvaluating dropout rate ", str(rate))
            count = 0
            mean_of_folds = DataFrame()
            for fold in folds:
                print("Training holding fold", str(count), "out..")
                if count == len(folds) - 1:
                    early_stopping_fold = folds[0]
                else:
                    early_stopping_fold = folds[count + 1]

                union_set = Data().construct_union_set(fold.copy(), early_stopping_fold.copy(), folds.copy())

                model = Classifier(type='DropoutAdam', nb_features=nb_features, nb_classes=nb_classes,
                                   epochs=args.epochs,
                                   batch_size=32,
                                   classes=classes, run_number=args.run, rate=rate)

                model.fit(X_train=union_set[0], Y_train=union_set[1], X_dev=early_stopping_fold[0],
                          Y_dev=early_stopping_fold[1])
                mean_of_folds["Fold " + str(count + 1)] = model.predict(X_test=fold[0], X_dev=fold[0], Y_dev=fold[1])
                count += 1

            if model is not None:
                metrics_per_set["Rate: " + str(rate)] = mean_of_folds.mean(axis=1)
        # metrics_per_set['cols'] = ['acc', 'prec', 'rec', 'f1']
        print(metrics_per_set)
        print(metrics_per_set.idxmax(axis=1))

    def experimentOptimisers(self, args):
        # Load data
        X_train, X_dev, X_test, Y_train, Y_dev, classes = Data().load_data()
        nb_features = X_train.shape[1]
        print(nb_features, 'features')
        nb_classes = Y_train.shape[1]
        print(nb_classes, 'classes')

        args.epochs = 100

        model = None
        metrics_per_set = DataFrame()

        for i in [1, 2, 3]:
            if i == 1:
                model = Classifier(type='DropoutAdamax', nb_features=nb_features, nb_classes=nb_classes,
                                   epochs=args.epochs,
                                   batch_size=64,
                                   classes=classes, run_number=args.run)
            elif i == 2:
                model = Classifier(type='DropoutAdam', nb_features=nb_features, nb_classes=nb_classes,
                                   epochs=args.epochs,
                                   batch_size=64,
                                   classes=classes, run_number=args.run)
            elif i == 3:
                model = Classifier(type='DropoutSGD', nb_features=nb_features, nb_classes=nb_classes,
                                   epochs=args.epochs,
                                   batch_size=64,
                                   classes=classes, run_number=args.run)
            model.fit(X_train=X_train, Y_train=Y_train, X_dev=X_dev, Y_dev=Y_dev)

            accuracies = model.predict(X_test=X_dev, X_dev=X_dev, Y_dev=Y_dev)
            metrics_per_set[str(i)] = accuracies
        print(metrics_per_set)

    def experimentBestModel(self, args):
        # Load data
        X_train, X_dev, X_test, Y_train, Y_dev, classes = Data().load_data()
        nb_features = X_train.shape[1]
        print(nb_features, 'features')
        nb_classes = Y_train.shape[1]
        print(nb_classes, 'classes')

        args.epochs = 100

        model = None
        metrics_per_set = DataFrame()

        model = Classifier(type='DropoutAdam', nb_features=nb_features, nb_classes=nb_classes,
                           epochs=args.epochs,
                           batch_size=8,
                           classes=classes, run_number=args.run, rate=0.4)
        model.fit(X_train=X_train, Y_train=Y_train, X_dev=X_dev, Y_dev=Y_dev)

        scores = model.predict(X_test=X_dev, X_dev=X_dev, Y_dev=Y_dev)
        metrics_per_set['scores'] = scores
        metrics_per_set['cols'] = ['Accuracy-score', 'Precision-score', 'Recall-score', 'F1-score']
        metrics_per_set.set_index('cols', drop=1)
        print(metrics_per_set)
