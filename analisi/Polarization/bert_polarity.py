from scipy.spatial.distance import cosine
import numpy as np
import torch
from transformers import BertTokenizer, BertModel

def word_vec(tokens, embeddings, word):  # return the embedding of a word
  word_vecs = []
  i_sentence = 0
  for sentence in tokens:
    i_token = 0
    for t in sentence:
      if t == word:
        word_vecs.append(embeddings[i_sentence][i_token])

      i_token = i_token + 1
    i_sentence = i_sentence + 1

  return word_vecs

def sum_vec(word_vecs):  
  vec_sum = np.zeros(len(word_vecs[0]))
  for v in word_vecs:
    vec_sum = vec_sum + v#.numpy()
  
  return vec_sum



def bertPolarity(tokens, embeddings, T, positive, negative):
  # Load pre-trained model tokenizer (vocabulary)
  tokenizer = BertTokenizer.from_pretrained('./Polarization/models/bert/tokenizer/')
  # Load pre-trained model (weights)
  # model = BertModel.from_pretrained('/models/bert/model/')
  topic = []
  for t in T:
    marked_text = "[CLS] " + t + " [SEP]"
    # Tokenize our sentence with the BERT tokenizer.
    tokenized_text = tokenizer.tokenize(marked_text)
    topic.append(tokenized_text[1:-1])

  A = []
  for p in positive:
    marked_text = "[CLS] " + p + " [SEP]"
    # Tokenize our sentence with the BERT tokenizer.
    tokenized_text = tokenizer.tokenize(marked_text)
    A.append(tokenized_text[1:-1])

  B = []
  for n in negative:
    marked_text = "[CLS] " + n + " [SEP]"
    # Tokenize our sentence with the BERT tokenizer.
    tokenized_text = tokenizer.tokenize(marked_text)
    B.append(tokenized_text[1:-1])

  #flattening
  topic = [item for sublist in topic for item in sublist]
  A = [item for sublist in A for item in sublist]
  B = [item for sublist in B for item in sublist]

  positive_score = 0
  negative_score = 0
  
  #Get list of vector
  t_list = []
  for t in topic:
    t_vec =  sum_vec(word_vec(tokens, embeddings, t))
    t_list.append(t_vec)

  a_list = []
  for a in A:
    a_vec =  sum_vec(word_vec(tokens, embeddings, a))
    a_list.append(a_vec)
      
  b_list = []
  for b in B:
    b_vec =  sum_vec(word_vec(tokens, embeddings, b))
    b_list.append(b_vec)

 # Polarity:
  for t in t_list:
    positive_temp = 0
    for a in a_list:
      positive_temp = positive_temp + ( 1 - cosine(t, a))
    positive_temp = positive_temp/len(a_list)
    

    negative_temp = 0
    for b in b_list:
      negative_temp = negative_temp + ( 1 - cosine(t, b))
    negative_temp = negative_temp/len(b_list)


    positive_score = positive_score + positive_temp
    negative_score = negative_score + negative_temp

  polarity = positive_score - negative_score

  return polarity

