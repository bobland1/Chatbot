import os
import yaml
import mysql.connector as sqlConnector
from mysql.connector import Error
from mysql.connector import errorcode

dir = os.path.dirname(os.path.realpath(__file__))
config = yaml.load(open('{}/../application.yml'.format(dir)), Loader=yaml.FullLoader)

def load_data():
    try:
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

        c.execute("SELECT value FROM questions")
        for row in c.fetchall():
            questions.append(row[0])

        c.execute("SELECT value FROM answers")
        for row in c.fetchall():
            answers.append(row[0])

        c.execute("SELECT * FROM vocab")
        for row in c.fetchall():
            vocab[row[1]] = row[0] - 1

        return questions, answers, vocab

    except sqlConnector.Error as error:
        print(error)

    finally:
        if (cnx.is_connected()):
            cnx.close()      
            print('Selected data from database')

global vocab, encoder_inp, decoder_inp, decoder_final_output

clean_questions, clean_answers, vocab = load_data()

encoder_inp = []
for line in clean_questions:
    lst = []
    for word in line.split():
        if word not in vocab:
            lst.append(vocab['<OUT>'])
        else:
            lst.append(vocab[word])
        
    encoder_inp.append(lst)

decoder_inp = []
for line in clean_answers:
    lst = []
    for word in line.split():
        if word not in vocab:
            lst.append(vocab['<OUT>'])
        else:
            lst.append(vocab[word])        
    decoder_inp.append(lst)

from tensorflow.keras.preprocessing.sequence import pad_sequences
encoder_inp = pad_sequences(encoder_inp, 13, padding='post', truncating='post')
decoder_inp = pad_sequences(decoder_inp, 13, padding='post', truncating='post')

decoder_final_output = []
for i in decoder_inp:
    decoder_final_output.append(i[1:]) 

decoder_final_output = pad_sequences(decoder_final_output, 13, padding='post', truncating='post')

print(decoder_final_output.shape, decoder_inp.shape, encoder_inp.shape, len(vocab), list(vocab.keys())[0])

from tensorflow.keras.utils import to_categorical
decoder_final_output = to_categorical(decoder_final_output, len(vocab))
print(decoder_final_output.shape)