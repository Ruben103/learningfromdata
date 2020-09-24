import sys

from DataService import DataService
from Experiments import Experiments


def main():
    trainset = sys.argv[1]
    testset = sys.argv[2]

    x, y = DataService().read_corpus(trainset)


if __name__ == '__main__':
    main()
