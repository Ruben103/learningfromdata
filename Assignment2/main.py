import sys
from Experiments import Experiments

if __name__ == '__main__':
    if len(sys.argv) == 2:
        if sys.argv[1] == '2':
            Experiments().experiment_edible_fruit()
    else:
        trainset = sys.argv[1]
        testset = sys.argv[2]
        Experiments().experiment1(trainset, testset)

