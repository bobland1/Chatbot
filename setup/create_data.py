import os
import yaml
import re
import mysql.connector as sqlConnector
from mysql.connector import Error
from mysql.connector import errorcode

dir = os.path.dirname(os.path.realpath(__file__))
config = yaml.load(open('{}/../application.yml'.format(dir)), Loader=yaml.FullLoader)

lines = open('{}/corpus/movie_lines.txt'.format(dir), encoding='utf-8',
             errors='ignore').read().split('\n')

conversations = open('{}/corpus/movie_conversations.txt'.format(dir), encoding='utf-8',
             errors='ignore').read().split('\n')
print('Loaded corpus')

exchange = []
for conversation in conversations:
    exchange.append(conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(",", "").split())

dialog = {}
for line in lines:
    dialog[line.split(' +++$+++ ')[0]] = line.split(' +++$+++ ')[-1]

questions = []
answers = []
for conversation in exchange:
    for i in range(len(conversation) - 1):
        questions.append(dialog[conversation[i]])
        answers.append(dialog[conversation[i+1]])

sorted_questions = []
sorted_answers = []
for i in range(len(questions)):
    if len(questions[i]) < 13:
        sorted_questions.append(questions[i])
        sorted_answers.append(answers[i])

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

clean_questions = []
clean_answers = []
for line in sorted_questions:
    clean_questions.append(clean_text(line))
        
for line in sorted_answers:
    clean_answers.append(clean_text(line))

for i in range(len(clean_answers)):
    clean_answers[i] = ' '.join(clean_answers[i].split()[:11])

clean_questions=clean_questions[:30000]
clean_answers=clean_answers[:30000]
print('Cleaned up questions and answers')

try:
    cnx = sqlConnector.connect(
      host=config['db']['ip'],
      user=config['db']['user'],
      password=config['db']['pass'],
      database=config['db']['schema']
    )
    c = cnx.cursor()

    for i in range(len(clean_answers)):
        c.execute("INSERT INTO answers (value) VALUES (%s)", (clean_answers[i],))
        c.execute("INSERT INTO questions (value, answerId) VALUES (%s, %s)", (clean_questions[i], c.lastrowid))
        cnx.commit()

except sqlConnector.Error as error:
    print(error)

finally:
    if (cnx.is_connected()):
        cnx.close()     
        print('Inserted data into database')