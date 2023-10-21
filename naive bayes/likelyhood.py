import pdb
import nltk
from nltk.corpus import twitter_samples
import numpy as np
import pandas as pd
import string
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
from freq import freqs, train_x, train_y


def lookup(freqs, word, label):
    n = 0
    pair = (word, label)

    if pair in freqs:
        n = freqs[pair]
    return n


def train_naive_bayes(freqs, train_x, train_y):
    loglikelihood = {}
    logprior = 0

    # Calculate V,the number of unique words in the vocab
    vocab = set([pair[0] for pair in freqs.keys()])
    V = len(vocab)

    # Calculate N_pos,N_neg,V_pos,V_neg
    N_pos = N_neg = 0
    #     V_pos=V_neg=0

    for pair in freqs.keys():
        if pair[1] > 0:
            #             V_pos+=1
            N_pos += freqs[pair]
        else:
            #             V_neg+=1
            N_neg += freqs[pair]

    #     D=len(train_y)

    # Calculate D_pos,the number of positive documents
    D_pos = (len(list(filter(lambda x: x > 0, train_y))))

    # Calculate D_neg,the number of negative documents
    D_neg = (len(list(filter(lambda x: x <= 0, train_y))))

    # Calculate logprior
    logprior = np.log(D_pos) - np.log(D_neg)

    for word in vocab:
        # Calculate the frequency of positive/negative word
        freq_pos = lookup(freqs, word, 1)
        freq_neg = lookup(freqs, word, 0)

        # Calculate the probability that each word is positice/negative
        p_w_pos = (freq_pos + 1) / (N_pos + V)
        p_w_neg = (freq_neg + 1) / (N_neg + V)

        # Calculate the log likelihood of the word
        loglikelihood[word] = np.log(p_w_pos / p_w_neg)

    return logprior, loglikelihood

logprior, loglikelihood = train_naive_bayes(freqs, train_x, train_y)
print(logprior)
print(len(loglikelihood))

