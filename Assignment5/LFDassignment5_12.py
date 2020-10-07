#!/usr/bin/env python3
import argparse
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.utils import np_utils, generic_utils
np.random.seed(2018)  # for reproducibility and comparability, don't change!
from Data import Data
from Experiments import Experiments


	
if __name__ == '__main__':
	# Read arguments
	parser = argparse.ArgumentParser(description='NN parameters')
	parser.add_argument('-r', '--run', type=int, default=0, help='run number')
	parser.add_argument('-e', '--epochs', metavar='N', type=int, default=5, help='epochs')
	parser.add_argument('-bs', '--batch-size', metavar='N', type=int, default=50, help='batch size')
	parser.add_argument('-cm', '--confusion-matrix', action='store_true', help = 'Show confusion matrix. Requires matplotlib')
	args = parser.parse_args()

	Experiments().experimentOne(args=args)


