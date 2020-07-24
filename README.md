# Simple implementation of word2vec with Numpy

This is a bare-bones implementation of the word2vec algorithm proposed by
Mikolov et. al. without any optimization techniques (such as Hierarchial
Softmax or Negative Sampling). It is implemented simply using Numpy with both
the forward and the backward steps done manually. The original paper suggested
both a Continuous-Bag-of-Words (CBOW) and a Skip-Gram model for embedding word
vectors. This is an implementation of the latter.
