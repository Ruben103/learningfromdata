import nltk
from nltk import *

import os
import pandas as pd


class Tokenizer:
    def sent_tokenizer(self):
        nltk.download('punkt')
        x = os.getcwd()
        txt = open('Albert Einstein___Relativity: The Special and General Theory.txt', 'rb')
        txt = txt.read().decode('utf-8', errors='ignore')
        txt = word_tokenize(txt)
        print("bug")


