import numpy as np
import scipy.special
import sklearn.metrics
import matplotlib.pyplot as plt

ranks = np.array([3.0, 386.0, 858.0])
probabilities = np.array([83397.0 / 2049990.0, 486.0 / 2049990.0, 237.0 / 2049990.0])

alphas = np.arange(1.11, 1.51, 0.01)
n = alphas.shape[0]

r2s = np.empty(n)

for i in range(n):
    alpha = alphas[i]
    zeta = scipy.special.zeta(alpha)
    predictions = np.empty(ranks.shape[0])
    for j in range(ranks.shape[0]):
        predictions[j] = ranks[j]**(-alpha) / zeta
    r2s[i] = sklearn.metrics.r2_score(predictions, probabilities)

plt.plot(alphas, r2s)
plt.show()
