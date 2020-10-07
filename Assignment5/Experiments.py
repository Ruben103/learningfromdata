from ClassifierService import Classifier
from Data import Data

class Experiments:

    def experimentOne(self, args):
        # Load data
        X_train, X_dev, X_test, Y_train, Y_dev, classes = Data().load_data()
        nb_features = X_train.shape[1]
        print(nb_features, 'features')
        nb_classes = Y_train.shape[1]
        print(nb_classes, 'classes')

        model = Classifier(type='MLP', nb_features=nb_features, nb_classes=nb_classes, epochs=500, batch_size=50, classes=classes, run_number=args.run)



