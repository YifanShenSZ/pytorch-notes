import numpy as np
import scipy
import sklearn.linear_model
import sklearn.metrics
import torch
import torchtext

train_iter = torchtext.datasets.WikiText2(split='train')
tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
vocab = torchtext.vocab.build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])

# from these we found that "." = 3, "!" = 385, "?" = 857
# print(vocab.vocab.get_stoi()["."])
# print(vocab.vocab.get_stoi()["!"])
# print(vocab.vocab.get_stoi()["?"])

word_list = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in train_iter]
words = torch.cat(tuple(filter(lambda t: t.numel() > 0, word_list)))
nwords_unique = vocab.vocab.get_itos().__len__()
frequencies = np.zeros(nwords_unique, dtype=int)
for i in range(words.numel()):
    frequencies[words[i]] += 1
sorted_indices = np.argsort(-frequencies)
sorted_frequencies = np.empty(nwords_unique, dtype=int)
for i in range(nwords_unique):
    sorted_frequencies[i] = frequencies[sorted_indices[i]]

# there are 2049990 words in total
# print(np.sum(frequencies))

# from these we found that vocab.vocab.get_itos() is not sorted by frequency
# iota = np.array([*range(nwords_unique)]) + 1
# print(np.linalg.norm(sorted_indices - iota))

# from these we found the ranks and the frequencies
# (note that rank is index in sorted_indices + 1)
# | word | rank | frequency |
# |------|------|-----------|
# | "."  |  3   |   83397   |
# | "!"  | 386  |    486    |
# | "?"  | 858  |    237    |
# index_3 = np.where(sorted_indices == 3)
# index_385 = np.where(sorted_indices == 385)
# index_857 = np.where(sorted_indices == 857)
# print(index_3, sorted_frequencies[index_3])
# print(index_385, sorted_frequencies[index_385])
# print(index_857, sorted_frequencies[index_857])

with open("sequence_lengthes.txt", 'w') as f:
    count = 1
    for i in range(words.numel()):
        word = words[i]
        if word == 3 or word == 385 or word == 857:
            print(count, file=f)
            count = 1
        else:
            count += 1

# create a unigram probability then fit it to Zipf's law:
# p(k) = k^(-a) / zeta(a) -> log(p(k)) = -a * log(k) - log(zeta(a))
# where k is the sorted index + 1 in frequency
# log_ks = np.log(np.array([*range(nwords_unique)]) + 1).reshape(-1, 1)
log_ks = np.empty((nwords_unique, 2))
log_ks[:, 0] = np.log(np.array([*range(nwords_unique)]) + 1)
log_ks[:, 1] = 1.0
log_probabilities = np.log(sorted_frequencies / np.sum(sorted_frequencies))
model = sklearn.linear_model.LinearRegression()
model.fit(log_ks, log_probabilities)

# fit returns a = 1.28900161, -log(zeta(a)) = 0.0, r^2 = 0.9786561358543696
# however, -log(zeta(1.28900161)) = -1.4006990186845665, so the fit does not comply well
# but we can have a sense that `a` approximately falls in range (1.0, 1.5]
# print(model.coef_)
# print(sklearn.metrics.r2_score(log_probabilities, model.predict(log_ks)))
