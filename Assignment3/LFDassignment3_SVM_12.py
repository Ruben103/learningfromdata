import sys
import Experiments
from pandas import read_csv

from DataService import DataService
from Experiments import Experiments


def main():
    trainset = sys.argv[1]
    testset = sys.argv[2]

    if sys.argv[3] == '1':
        Experiments().experimentDefaultSetting(trainset, testset)
    elif sys.argv[3] == '2':
        Experiments().experimentCParamterCrossValidation(trainset, testset)
    elif sys.argv[3] == '3':
        Experiments().experimentCombinatorialCrossValidation(trainset, testset)


if __name__ == '__main__':
    main()
