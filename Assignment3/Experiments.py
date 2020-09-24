from DataService import DataService
from SupportVectorMachine import SVM

class Experiments:

    def experimentDefaultSetting(self, trainset, testset):
        x, y = DataService().read_corpus(trainset)
        clf = SVM().construct_classifier("linear")
        x_train, y_train, x_test, y_test = DataService().test_train_split(x, y)
        clf.fit(x_train, y_train)
        score = clf.score(x_test, y_test)
        print(score)

