import sys
import Experiments
from pandas import read_csv

from DataService import DataService
from Experiments import Experiments


def main():
    #trainset = sys.argv[1]
    #testset = sys.argv[2]

    #x, y = DataService().read_corpus(trainset)
    Experiments().experimentDefaultSetting('trainset.txt', None)



if __name__ == '__main__':
    main()
