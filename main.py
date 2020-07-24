import re
import numpy as np


def get_X_Y(tokenized, word2idx, window_size):
    X, Y = [], []
    for sentence in tokenized:
        for i in range(len(sentence)):
            center = sentence[i]
            for j in range(i-window_size, i+window_size):
                if i == j or j < 0 or j >= len(sentence):
                    continue
                X.append(word2idx[center])
                Y.append(word2idx[sentence[j]])
    X, Y = np.array([X]), np.array([Y])
    return X, Y


def load_conllu_data(filename):
    sentences = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        sentence = []
        for line in lines:
            line = line.strip('\n')
            if line.startswith('#'):
                continue
            if line == '':
                sentence = ' '.join(sentence)
                pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
                sentences.append(pattern.findall(sentence.lower()))
                sentence = []
            else:
                line = line.split('\t')
                sentence.append(line[1])
    return sentences


def tokenize_corpus(corpus):
    return [x.split(' ') for x in corpus]


def pair_to_idxs(pair, word2idx):
    return (word2idx[pair[0]], [word2idx[w] for w in pair[1]])


def one_hot(word, word2idx):
    v = np.zeros(len(word2idx))
    v[word2idx[word]] = 1.0
    return v


def softmax(X):
    """
    Numerically stable implementation of softmax
    """
    exps = np.exp(X - np.max(X))
    return exps / np.sum(exps, axis=0)


def get_word_vecs(X, EMB):
    """
    X: batch of word indicies. shape (1, batch_size)
    EMB: embedding matrix. shape (vocab_size, emb_dims)
    """
    return W1[X_batch.flatten()].T


def cross_entropy_loss(probs, Y):
    m = probs.shape[1]
    log_probs = np.log(probs)
    cost = -(1 / m) * np.sum(np.sum(Y * log_probs, axis=0, keepdims=True), axis=1)
    return cost



corpus = [
    'he is a king',
    'she is a queen',
    'he is a man',
    'she is a woman',
    'warsaw is poland capital',
    'berlin is germany capital',
    'paris is france capital',
    'get me another drink',
    'a queen is nice',
    'poland is a country',
    'poland has no queen',
]

tokenized = tokenize_corpus(corpus)
vocabulary = set([token for sentence in tokenized for token in sentence])
word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}

X, Y = get_X_Y(tokenized, word2idx, 2)
m = X.shape[1]  # training instances

lr = 0.05   # learning rate
N = 50       # embedding dimensions
V = len(vocabulary)
W1 = np.random.rand(V, N) * 0.01
W2 = np.random.rand(V, N) * 0.01


batch_size = 1
costs = []
for epoch in range(3):
    epoch_loss = 0
    batch_idxs = list(range(0, m, batch_size))
    np.random.shuffle(batch_idxs)
    for i in batch_idxs:
        X_batch = X[:,i:i+batch_size]
        Y_batch = Y[:,i:i+batch_size]

        # forward pass
        X_vecs = get_word_vecs(X_batch, W1)
        hidden = np.dot(W2, X_vecs)
        probs = softmax(hidden)

        # loss
        bs = Y_batch.shape[1]
        Y_true = np.zeros((V, bs))
        Y_true[Y_batch.flatten(), np.arange(bs)] = 1
        loss = cross_entropy_loss(probs, Y_true)
        epoch_loss += np.squeeze(loss)

        # calculate gradients
        dl_dZ = probs - Y_true
        dl_dW2 = (1 / batch_size) * np.matmul(dl_dZ, X_vecs.T)
        dl_dW1 = np.matmul(W2.T, dl_dZ)

        # backward pass
        W1[X_batch.flatten(),:] -= dl_dW1.T * lr
        W2 -= lr * dl_dW2

    costs.append(epoch_loss)
    print(f'loss after epoch {epoch}: {epoch_loss}')
    print()

print()
