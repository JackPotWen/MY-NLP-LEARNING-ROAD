import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
import re
import string
from dataprepare import train_x, train_y

def process_tweet(tweet):
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')

    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)

    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)

    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    # remove stopwords
    # remove punctuation
    for word in tweet_tokens:
        if word not in stopwords_english and \
                word not in string.punctuation:
            stem_word = stemmer.stem(word)
            tweets_clean.append(stem_word)
    return tweets_clean


def build_freqs(tweets,ys):
    yslist = np.squeeze(ys).tolist()
    freqs = {}
    for y,tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            pair=(word,y)
            if pair in freqs:
                freqs[pair]+=1
            else:
                freqs[pair]=1
    return freqs

freqs = build_freqs(train_x, train_y)
print("Create frequency dictionary")
print("type(freqs) = " + str(type(freqs)))
print("len(freqs) = " + str(len(freqs.keys())))

"""
# test the function below
print('This is an example of a positive tweet: \n', train_x[1])
print('\nThis is an example of the processed version of the tweet: \n', process_tweet(train_x[1]))
"""