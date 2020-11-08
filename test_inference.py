import os
import re
import os
import yaml
import numpy as np
import mysql.connector as sqlConnector
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Concatenate
from lib.AttentionLayer import AttentionLayer
import tensorflow as tf
import os
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import backend as K
from mysql.connector import Error
from mysql.connector import errorcode

dir = os.path.dirname(os.path.realpath(__file__))
config = yaml.load(open('{}/application.yml'.format(dir)), Loader=yaml.FullLoader)
tf.get_logger().setLevel('INFO')

def load_data():
    try:
        global vocab
        vocab = {}
        questions = []
        answers = []
        cnx = sqlConnector.connect(
          host=config['db']['ip'],
          user=config['db']['user'],
          password=config['db']['pass'],
          database=config['db']['schema']
        )
        c = cnx.cursor()

        c.execute("SELECT * FROM vocab")
        for row in c.fetchall():
            vocab[row[1]] = row[0] - 1

        return vocab

    except sqlConnector.Error as error:
        print(error)

    finally:
        if (cnx.is_connected()):
            cnx.close()      
            print('Selected data from database')

def clean_text(txt):
    txt = txt.lower()
    txt = re.sub(r"i'm", "i am", txt)
    txt = re.sub(r"he's", "he is", txt)
    txt = re.sub(r"she's", "she is", txt)
    txt = re.sub(r"that's", "that is", txt)
    txt = re.sub(r"what's", "what is", txt)
    txt = re.sub(r"where's", "where is", txt)
    txt = re.sub(r"\'ll", " will", txt)
    txt = re.sub(r"\'ve", " have", txt)
    txt = re.sub(r"\'re", " are", txt)
    txt = re.sub(r"\'d", " would", txt)
    txt = re.sub(r"won't", "will not", txt)
    txt = re.sub(r"can't", "can not", txt)
    txt = re.sub(r"[^\w\s]", "", txt)
    return txt

attn_layer = AttentionLayer()

model = load_model('model/chatbot.h5', custom_objects={'AttentionLayer' : attn_layer})

encoder_inputs = model.layers[0].input
embed = model.layers[2]
enc_embed = embed(encoder_inputs)
enocoder_layer = model.layers[3]

encoder_outputs, fstate_h, fstate_c, bstate_h, bstate_c = enocoder_layer(enc_embed)

h = Concatenate()([fstate_h, bstate_h])
c = Concatenate()([fstate_c, bstate_c])
encoder_states = [h, c]

enc_model = Model(encoder_inputs, [encoder_outputs, encoder_states])

latent_dim = 800

decoder_inputs = model.layers[1].input
decoder_lstm = model.layers[6]
decoder_dense = model.layers[9]
decoder_state_input_h = Input(shape=(latent_dim,), name='input_3')
decoder_state_input_c = Input(shape=(latent_dim,), name='input_4')

decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

dec_embed = embed(decoder_inputs)

decoder_outputs, state_h, state_c = decoder_lstm(dec_embed, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]

dec_model = Model([decoder_inputs, decoder_states_inputs], [decoder_outputs] + decoder_states)

dec_dense = model.layers[-1]
attn_layer = model.layers[7]

from tensorflow.keras.preprocessing.sequence import pad_sequences
vocab = load_data()
inv_vocab = {w:v for v, w in vocab.items()}

prepro1 = ""
while prepro1 != 'q':
    prepro1  = input("you : ")
    try:
        prepro1 = clean_text(prepro1)
        prepro = [prepro1]

        txt = []
        for x in prepro:
            lst = []
            for y in x.split():
                try:
                    lst.append(vocab[y])
                except:
                    lst.append(vocab['<OUT>'])
            txt.append(lst)
        txt = pad_sequences(txt, 13, padding='post')

        enc_op, stat = enc_model.predict(txt)

        empty_target_seq = np.zeros((1, 1))
        empty_target_seq[0, 0] = vocab['<SOS>']
        stop_condition = False
        decoded_translation = ''

        while not stop_condition :

            dec_outputs, h, c = dec_model.predict([empty_target_seq] + stat)

            attn_op, attn_state = attn_layer([enc_op, dec_outputs])
            decoder_concat_input = Concatenate(axis=-1)([dec_outputs, attn_op])
            decoder_concat_input = dec_dense(decoder_concat_input)

            sampled_word_index = np.argmax(decoder_concat_input[0, -1, :] )
            sampled_word = inv_vocab[sampled_word_index] + ' '

            if sampled_word != '<EOS> ':
                decoded_translation += sampled_word  

            if sampled_word == '<EOS> ' or len(decoded_translation.split()) > 13:
                stop_condition = True 

            empty_target_seq = np.zeros((1, 1))  
            empty_target_seq[0, 0] = sampled_word_index
            stat = [h, c]  

        print("chatbot: ", decoded_translation)  
    except:
        print("sorry didn't got you , please type again :( ")
