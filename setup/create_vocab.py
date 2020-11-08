import os
import yaml
import mysql.connector as sqlConnector
from mysql.connector import Error
from mysql.connector import errorcode

dir = os.path.dirname(os.path.realpath(__file__))
config = yaml.load(open('{}/../application.yml'.format(dir)), Loader=yaml.FullLoader)

def load_data():
    try:
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

        return questions, answers

    except sqlConnector.Error as error:
        print(error)

    finally:
        if (cnx.is_connected()):
            cnx.close()      
            print('Selected data from database')

clean_questions, clean_answers = load_data()

word2count = {}
for line in clean_questions:
    for word in line.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1
for line in clean_answers:
    for word in line.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

threshold = 5

vocab = {}
word_num = 0
for word, count in word2count.items():
    if count >= threshold:
        vocab[word] = word_num
        word_num += 1  

for i in range(len(clean_answers)):
    clean_answers[i] = '<SOS> ' + clean_answers[i] + ' <EOS>'

tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
x = len(vocab)
for token in tokens:
    vocab[token] = x
    x += 1

vocab['cameron'] = vocab['<PAD>']
vocab['<PAD>'] = 0

inv_vocab = {w:v for v, w in vocab.items()}

try:
    cnx = sqlConnector.connect(
      host=config['db']['ip'],
      user=config['db']['user'],
      password=config['db']['pass'],
      database=config['db']['schema']
    )
    c = cnx.cursor()

    for i in range(len(inv_vocab)):
        c.execute("INSERT INTO vocab (value) VALUES (%s)", (inv_vocab[i],))
        cnx.commit()

except sqlConnector.Error as error:
    print(error)

finally:
    if (cnx.is_connected()):
        cnx.close()      
        print('Inserted data into database')