import lib.glove_embedding as embedding
from lib.AttentionLayer import AttentionLayer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input, Bidirectional, Concatenate, Dropout, Attention
import tensorflow as tf
tf.get_logger().setLevel('INFO')

vocab = embedding.vocab
embed = embedding.embed
encoder_inp = embedding.encoder_inp
decoder_inp = embedding.decoder_inp
decoder_final_output = embedding.decoder_final_output

enc_inp = Input(shape=(13, ))
dec_inp = Input(shape=(13, ))

enc_embed = embed(enc_inp)
enc_lstm = Bidirectional(LSTM(400, return_state=True, dropout=0.05, return_sequences = True))
encoder_outputs, forward_h, forward_c, backward_h, backward_c = enc_lstm(enc_embed)
state_h = Concatenate()([forward_h, backward_h])
state_c = Concatenate()([forward_c, backward_c])
enc_states = [state_h, state_c]

dec_embed = embed(dec_inp)
dec_lstm = LSTM(400*2, return_state=True, return_sequences=True, dropout=0.05)
output, _, _ = dec_lstm(dec_embed, initial_state=enc_states)

attn_layer = AttentionLayer()
attn_op, attn_state = attn_layer([encoder_outputs, output])
decoder_concat_input = Concatenate(axis=-1)([output, attn_op])

VOCAB_SIZE = len(vocab)
dec_dense = Dense(VOCAB_SIZE, activation='softmax')
final_output = dec_dense(decoder_concat_input)

model = Model([enc_inp, dec_inp], final_output)
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.fit([encoder_inp, decoder_inp], decoder_final_output, epochs=40, batch_size=24, validation_split=0.15)

model.save('model/chatbot.h5')
print("Saved model to disk")