import lib.prepare_vocab as prepare_vocab
import numpy as np
import os

global vocab, encoder_inp, decoder_inp, decoder_final_output

vocab = prepare_vocab.vocab
encoder_inp = prepare_vocab.encoder_inp
decoder_inp = prepare_vocab.decoder_inp
decoder_final_output = prepare_vocab.decoder_final_output

VOCAB_SIZE = len(vocab)

embeddings_index = {}
dir = os.path.dirname(os.path.realpath(__file__))
with open('{}/../setup/corpus/glove.6B.50d.txt'.format(dir), encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

print("Glove Loded!")

embedding_dimention = 50
def embedding_matrix_creater(embedding_dimention, word_index):
    embedding_matrix = np.zeros((len(word_index)+1, embedding_dimention))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
embedding_matrix = embedding_matrix_creater(50, word_index=vocab)    

print(embedding_matrix.shape)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding

embed = Embedding(VOCAB_SIZE+1, 50, input_length=13, trainable=True)

embed.build((None,))
embed.set_weights([embedding_matrix])