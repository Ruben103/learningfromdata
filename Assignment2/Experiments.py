from sklearn.tree import DecisionTreeClassifier
from DataService import DataService
from ClassifierService import ClassifierService

class Experiments:

    def experiment1(self, trainset, testset):
        x, y = DataService().read_corpus(trainset)
        x_train, y_train, x_test, y_test = DataService().test_train_split(x,y)

        classifier = ClassifierService().construct_decisiontreeclassifier(criter)
        classifier.fit(x_train, y_train)

        predictions = classifier.predict(x_test)

        print("STOP")