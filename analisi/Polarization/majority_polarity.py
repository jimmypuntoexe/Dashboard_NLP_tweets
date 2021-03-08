from w2v_polarity import w2vPolarity
from glove_polarity import glovePolarity
from bert_polarity import bertPolarity
from gensim.models import Word2Vec
from glove import Glove
import numpy as np
import csv

# Load W2V models
trump_w2v = Word2Vec.load("models/trump_w2v.model")
clinton_w2v = Word2Vec.load("./models/clinton_w2v.model")

# Load GloVe models
trump_glove = Glove.load("models/trump_glove.model")
clinton_glove = Glove.load("./models/clinton_glove.model")

# Load BERT models ( NUMPY NO TENSOR !!!!!!!!!!!!!!!)
with open('./models/tokens_trump_bert.csv', newline='') as f:
    reader = csv.reader(f)
    trump_bert_tokens = list(reader)
trump_bert_emb = np.load('./models/embedding_trump_bert.npy', allow_pickle=True)   

with open('./models/tokens_clinton_bert.csv', newline='') as f:
    reader = csv.reader(f)
    clinton_bert_tokens = list(reader)
clinton_bert_emb = np.load('models/embedding_clinton_bert.npy', allow_pickle=True)   






topic = ['hillaryclinton', 'hillary', 'democratic', 'dem']

positive = ['good', 'great', 'nice', 'positive', 'love' ]
negative = ['bad', 'badly', 'negative', 'false', 'wrong']



positive_count = 0
w2vec_pol = w2vPolarity(trump_w2v, topic, positive, negative)
print(w2vec_pol)
if  w2vec_pol > 0:
    positive_count = positive_count + 1

glove_pol = glovePolarity(trump_glove, topic, positive, negative)
print(glove_pol)
if glove_pol > 0:
    positive_count = positive_count + 1



#a = np.loadtxt('./models/embeddings_clinton_bert.csv', delimiter=',')  
bert_pol = bertPolarity(trump_bert_tokens, trump_bert_emb, topic, positive, negative)
print(bert_pol)
if bert_pol > 0 :
    positive_count = positive_count + 1


if positive_count > 1:
    print(1)
else:
    print(-1)