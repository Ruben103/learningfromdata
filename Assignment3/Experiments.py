from DataService import DataService

class Experiments:

    def experiment1(self, trainset, testset):
        x, y = DataService().read_corpus(trainset)


