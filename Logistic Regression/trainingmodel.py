import numpy as np
from LR import gradient
from dictionary import build_freqs, process_tweet, freqs
from dataprepare import train_x, train_y
from Extractingthefeatures import extract_features

X = np.zeros((len(train_x), 3))
for i in range(len(train_x)):
    X[i, :] = extract_features(train_x[i], freqs)

Y = train_y

J, theta = gradient(X, Y, np.zeros((3, 1)), 1e-9, 1500)
print(f"The cost after training is {J:.8f}.")
print(f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(theta)]}")