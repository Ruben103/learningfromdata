from Experiments import Experiments
import sys
"""
OUR COMMENT:
This part of the code reads in the trainset.txt and converts it to two lists {X,Y}
X contains sentences
Y contains the corresponding sentiment labels.
"""


if __name__ == '__main__':

    if sys.argv[1] == '1':
        Experiments.experiment_binary(Experiments())

    elif sys.argv[1] == '2':
        Experiments.experiment_multi_class(Experiments())

    elif sys.argv[1] == '3':
        Experiments.experiment_probabilities_binary(Experiments())

    elif sys.argv[1] == '4':
        Experiments.experiment_probabilities_multi(Experiments())


