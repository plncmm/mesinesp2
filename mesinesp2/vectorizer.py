import sys
sys.path.append("..")
import mesinesp2.tokenizer
import nltk
import numpy as np
import pandas as pd
import re
import gensim # module for computing word embeddings
import numpy as np # linear algebra module
import sklearn.feature_extraction # package to perform tf-idf vertorization
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from gensim.models.keyedvectors import KeyedVectors
from flair.embeddings import DocumentPoolEmbeddings, StackedEmbeddings, FlairEmbeddings, TransformerDocumentEmbeddings
from flair.data import Sentence
import fasttext.util
import time
stop_words = stopwords.words('spanish')
nltk.download('stopwords')


def get_idf(decs):
    tokenized_decs = {}
    for i in decs.keys():
        tokenized_decs[i] = tokenized_decs[i] = mesinesp2.tokenizer.tokenizer(decs[i], split_sentences=False, is_df = False, normalize = True)
        tokenized_decs[i] = [word for word in tokenized_decs[i] if word not in stop_words]
    sentences = []
    for sentence in tokenized_decs.values():
        sentences.append(' '.join(sentence))
    tfidfvectorizer = sklearn.feature_extraction.text.TfidfVectorizer()           # instance of the tf-idf vectorizer
    tfidfvectorizer.fit(sentences)                                                # fitting the vectorizer and transforming the properties
    idf = {key:val for key, val in zip(tfidfvectorizer.get_feature_names(), tfidfvectorizer.idf_)}
    #with open(filepath, 'w+', encoding='utf-8') as json_file:
    #    json.dump(idf, json_file, indent=2, ensure_ascii=False)
    return idf, tokenized_decs, sentences

def to_vector(text, model, idf, is_tokenized=False):
    """ Receives a sentence string along with a word embedding model and 
    returns the vector representation of the sentence"""
    if not is_tokenized: text= text.split() # splits the text by space and returns a list of words
    vec = np.zeros(300) # creates an empty vector of 300 dimensions
    for word in text: # iterates over the sentence
        if (word in model) & (word in idf): # checks if the word is both in the word embedding and the tf-idf model
            vec += model[word]*idf[word] # adds every word embedding to the vector
    if np.linalg.norm(vec) > 0:
        return vec / np.linalg.norm(vec) # divides the vector by their normal
    else:
        return vec



def get_pretrained_embeddings(sentences, model, idf, tokenized_decs):
    vectorized_sent = [to_vector(text, model, idf) for text in sentences]
    embedding_dict = {list(tokenized_decs.keys())[i]: vectorized_sent[i].tolist() for i in range(len(list(tokenized_decs.keys())))}   
    return embedding_dict

def get_embeddings_from_text(sentences, model, idf):
    vectorized_sent = [to_vector(text, model, idf, is_tokenized=True) for text in sentences]
    return vectorized_sent


def get_contextualized_embeddings(decs, embeddings):
    contextualized_embeddings = {}
    cnt = 0
    for k, v in decs.items():
        cnt += 1
        if cnt%100==0: print(f'{cnt} codes transformed..')
        s = Sentence(v)
        embeddings.embed(s)
        contextualized_embeddings[k] = s.embedding.detach().numpy() 
    return contextualized_embeddings

def create_decs_embeddings():
    descriptions = mesinesp2.tokenizer.get_descriptions("../data/raw/DeCS2020.obo", tokenize_definition = False, tokenize_name = False)
    decs = {}
    for key,val in descriptions.items():
        if key.startswith("D"):
            code = key
            document = val["name"]
            if "def" in val:
                document = document + " " + val["def"]
            decs[code] = document

    print(f'Cantidad de códigos: {len(decs)}.')


    idf, tokenized_decs, sentences = get_idf(decs)
    with open('../embeddings/idf.json', 'w+', encoding='utf-8') as json_file:
        json.dump(idf, json_file, indent=2, ensure_ascii=False)

    # SBW
    start = time.time()
    sbw = KeyedVectors.load_word2vec_format('../embeddings/SBW-vectors-300-min5.txt', limit = 100000)
    sbw_embeddings = get_pretrained_embeddings(sentences, sbw, idf, tokenized_decs)
    with open('../embeddings/decs_sbw.json', 'w') as fp:
        json.dump(sbw_embeddings, fp) 
    print(f'{time.time()-start} seconds to get sbw embeddings.')


    # Mix
    start = time.time()
    mix = fasttext.load_model('../embeddings/mix_fasttext.bin')
    mix_embeddings = get_pretrained_embeddings(sentences, mix, idf, tokenized_decs)
    with open('../embeddings/decs_mix.json', 'w') as fp:
        json.dump(mix_embeddings, fp) 
    print(f'{time.time()-start} seconds to get mix embeddings.')


    # Bert embeddings
    #start = time.time()
    #bert = TransformerDocumentEmbeddings("dccuchile/bert-base-spanish-wwm-uncased") # Reference: https://github.com/UKPLab/sentence-transformers
    #bert_embeddings = get_contextualized_embeddings(decs, bert) # 768 Dimensiones
    #with open('../embeddings/bert.json', 'w') as fp:
    #    json.dump(bert_embeddings, fp)
    #print(f'{time.time()-start} seconds to get bert embeddings.')

    # Flair emebedings
    #start = time.time()
    #stacked_embeddings = StackedEmbeddings(embeddings = [FlairEmbeddings('es-forward'), FlairEmbeddings('es-backward')])
    #flair = DocumentPoolEmbeddings([stacked_embeddings])
    #flair_embeddings = get_contextualized_embeddings(decs, flair) # 4096 Dimensiones
    #with open('../embeddings/flair.json', 'w') as fp:
    #    json.dump(flair_embeddings, fp)
    #print(f'{time.time()-start} seconds to get flair embeddings.')
    
    return idf
