import numpy as np
from LR import sigmoid
from dataprepare import test_x, test_y
from dictionary import freqs, process_tweet
from Extractingthefeatures import extract_features
from trainingmodel import theta


def predict_tweet(tweet, freqs, theta):
    x = extract_features(tweet, freqs)
    y_pred = sigmoid(np.dot(x, theta))

    return y_pred


def sentiment(x):
    return str('Good') if x>0.5 else str('Bad')


def test_lr(test_x, test_y, freqs, theta):
    y_hat = []

    for tweet in test_x:
        y_pred = predict_tweet(tweet, freqs, theta)
        y_hat.append(1) if y_pred > 0.5 else y_hat.append(0)

    accuracy = (y_hat == np.squeeze(test_y)).sum() / len(test_x)

    return accuracy


tmp_accuracy = test_lr(test_x, test_y, freqs, theta)
print(f"Logistic regression model's accuracy = {tmp_accuracy:.4f}")

#ERROR ANALYSIS
# Some error analysis done for you
print('Label Predicted Tweet')
for x,y in zip(test_x,test_y):
    y_hat = predict_tweet(x, freqs, theta)

    if np.abs(y - (y_hat > 0.5)) > 0:
        print('THE TWEET IS:', x)
        print('THE PROCESSED TWEET IS:', process_tweet(x))
        print('%d\t%0.8f\t%s' % (y, y_hat, ' '.join(process_tweet(x)).encode('ascii', 'ignore')))
"""
#test
for tweet in ['I love Xu Ruiyu', 'I am bad', 'this movie should have been great.',\
              'great', 'great great', 'great great great', 'great great great great']:
    print( '{} -> {} -> {}' .format(tweet,predict_tweet(tweet,freqs,theta)[0][0] ,sentiment(predict_tweet(tweet, freqs, theta)[0][0])))

my_tweet = 'I am fucking :)'
print(predict_tweet(my_tweet, freqs, theta), sentiment(predict_tweet(my_tweet, freqs, theta)))
"""