import numpy as np
from dictionary import build_freqs, process_tweet, freqs
from dataprepare import train_x, train_y


def extract_features(tweet, freqs):
    word_1 = process_tweet(tweet)
    x = np.zeros((1, 3))

    x[0, 0] = 1

    for word in word_1:
        x[0, 1] += freqs.get((word, 1.), 0)
        x[0, 2] += freqs.get((word, 0.), 0)

    assert (x.shape == (1, 3))

    return x
"""
# Check your function
tmp1 = extract_features(train_x[0], freqs)
print(tmp1)

# Check for when the words are not in the freqs dictionary
tmp2 = extract_features('hate', freqs)
print(tmp2)
"""