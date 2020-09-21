import sys
from Experiments import Experiments

if __name__ == '__main__':
    trainset = sys.argv[1]
    testset = sys.argv[2]
    Experiments().experiment1(trainset, testset)

