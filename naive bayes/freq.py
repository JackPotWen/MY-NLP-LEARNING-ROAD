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
"""--------------------------------------------------data prepare-------------------------------------------------------"""

all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

# Split the data
train_pos = all_positive_tweets[:4000]
test_pos = all_positive_tweets[4000:]
train_neg = all_negative_tweets[:4000]
test_neg = all_negative_tweets[4000:]

train_x = train_pos+train_neg
test_x = test_pos+test_neg

train_y = np.append(np.ones(len(train_pos)),np.zeros(len(train_neg)))
test_y = np.append(np.ones(len(test_pos)),np.zeros(len(test_neg)))

"""--------------------------------------------------data process-------------------------------------------------------"""


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

    for word in tweet_tokens:
        if word not in stopwords_english \
                and word not in string.punctuation:
            stem_word = stemmer.stem(word)
            tweets_clean.append(stem_word)

    return tweets_clean


def count_tweets(result, tweets, ys):
    for y, tweet in zip(ys, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)

            if pair in result:
                result[pair] += 1
            else:
                result[pair] = 1

    return result

freqs = count_tweets({}, train_x, train_y)