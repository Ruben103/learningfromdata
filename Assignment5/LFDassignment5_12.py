#!/usr/bin/env python3
import argparse

import numpy as np

np.random.seed(2018)  # for reproducibility and comparability, don't change!
from Experiments import Experiments
from Tokenizer import *

import sys

if __name__ == '__main__':

	Tokenizer().sent_tokenizer()
	# Read arguments
	#parser = argparse.ArgumentParser(description='NN parameters')
	#parser.add_argument('-r', '--run', type=int, default=0, help='run number')
	#parser.add_argument('-e', '--epochs', metavar='N', type=int, default=5, help='epochs')
	#parser.add_argument('-bs', '--batch-size', metavar='N', type=int, default=50, help='batch size')
	#parser.add_argument('-cm', '--confusion-matrix', action='store_true', help = 'Show confusion matrix. Requires matplotlib')
	#args = parser.parse_args()
	#Experiments().experimentDropout(args=args)


