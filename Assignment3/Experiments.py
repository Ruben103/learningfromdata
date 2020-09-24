from DataService import DataService
import SupportVectorMachine

class Experiments:

    def experiment1(self, trainset, testset):
        x, y = DataService().read_corpus(trainset)
        clf = SupportVectorMachine().construct_classifier()
        x_train, y_train, x_test, y_test = DataService().test_train_split(x, y)
        clf.fit(x_train, y_train)



