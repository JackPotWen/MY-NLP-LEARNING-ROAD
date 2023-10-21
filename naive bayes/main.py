import numpy as np
from freq import freqs, train_x, train_y, process_tweet, test_x, test_y
from likelyhood import logprior, loglikelihood, lookup


def naive_bayes_predict(tweet, logprior, loglikelihood):
    word_1 = process_tweet(tweet)

    p = 0

    p += logprior

    for word in word_1:
        if word in loglikelihood:
            p += loglikelihood[word]

    return p


def test_naive_bayes(test_x, test_y, logprior, loglikelihood):
    accuracy = 0
    y_hats = []

    for tweet in test_x:
        if naive_bayes_predict(tweet, logprior, loglikelihood) > 0:
            y_hat_i = 1
        else:
            y_hat_i = 0
        y_hats.append(y_hat_i)

    error = np.mean(np.absolute(y_hats - test_y))

    accuracy = 1 - error

    return accuracy


def get_ratio(freqs, word):
    pos_neg_ratio = {'positive': 0, 'negative': 0, 'ratio': 0.0}

    pos_neg_ratio['positive'] = lookup(freqs, word, 1)

    pos_neg_ratio['negative'] = lookup(freqs, word, 0)

    # calculate the ratio of positive to negative counts for the word
    pos_neg_ratio['ratio'] = (pos_neg_ratio['positive'] + 1) / (pos_neg_ratio['negative'] + 1)

    return pos_neg_ratio


def get_words_by_threshold(freqs, label, threshold):
    word_list = {}

    for key in freqs.keys():
        word, _ = key

        pos_neg_ratio = get_ratio(freqs, word)

        if label == 1 and pos_neg_ratio['ratio'] >= threshold:

            word_list[word] = pos_neg_ratio

        elif label == 0 and pos_neg_ratio['ratio'] <= threshold:

            word_list[word] = pos_neg_ratio

    return word_list

print("Naive Bayes accuracy = %0.4f" %
      (test_naive_bayes(test_x, test_y, logprior, loglikelihood)))

# error demo
print('Truth Predicted Tweet')
for x, y in zip(test_x, test_y):
    y_hat = naive_bayes_predict(x, logprior, loglikelihood)
    if y != (np.sign(y_hat) > 0):
        print('%d\t%0.2f\t%s' % (y, np.sign(y_hat) > 0, ' '.join(
            process_tweet(x)).encode('ascii', 'ignore')))

"""
# Test
my_tweet = 'She smiled.'
p = naive_bayes_predict(my_tweet, logprior, loglikelihood)
print('The expected output is', p)

# Test

for tweet in ['I am happy', 'I am bad', 'this movie should have been great.', 'great', 'great great', 'great great great', 'great great great great']:
    p = naive_bayes_predict(tweet, logprior, loglikelihood)
    print(f'{tweet} -> {p:.2f}')

# Test
my_tweet = 'you are bad :('
naive_bayes_predict(my_tweet, logprior, loglikelihood)

# Test
print(get_ratio(freqs, 'happi'))
print(get_words_by_threshold(freqs, label=0, threshold=0.05))
"""

