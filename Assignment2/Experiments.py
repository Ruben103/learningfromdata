from sklearn.tree import DecisionTreeClassifier
from DataService import DataService
from ClassifierService import ClassifierService
from numpy import array

class Experiments:

    def experiment1(self, trainset, testset):
        x, y = DataService().read_corpus(trainset)
        x_train, y_train, x_test, y_test = DataService().test_train_split(x,y)

        classifier = ClassifierService().construct_decisiontreeclassifier()
        classifier.fit(x_train, y_train)

        predictions = classifier.predict(x_test)

        print("STOP")

    def experiment_edible_fruit(self):

        table = DataService().read_edible_fruit_table()