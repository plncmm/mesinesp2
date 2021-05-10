
from tokenizer import get_descriptions, transform
from vectorizer import get_idf, get_pretrained_embeddings, get_embeddings_from_text, create_decs_embeddings
import sys
import json
import time
import fasttext
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
import mesinesp2
from sklearn.metrics.pairwise import cosine_similarity
sys.path.append("..")


def similarity(data, model, idf):
    if model == 'sbw':
        sbw = KeyedVectors.load_word2vec_format('../embeddings/SBW-vectors-300-min5.txt', limit = 100000)
        model_embeddings = get_embeddings_from_text(data, sbw, idf)
        with open('../embeddings/decs_sbw.json') as f:
            decs_embs = json.load(f)

    elif model == 'mix':
        mix = fasttext.load_model('../embeddings/mix_fasttext.bin')
        model_embeddings = get_embeddings_from_text(data, mix, idf)
        with open('../embeddings/decs_mix.json') as f:
           decs_embs = json.load(f)
        
    elif model == 'flair': 
        # Como aún no sé si funcionan no está implementada esta comparación. (FLAIR)
        model_embeddings = []
        decs_embs = {}
    
    else: 
        # Como aún no sé si funcionan no está implementada esta comparación. (BERT)
        model_embeddings = []
        decs_embs = {}
    
    abstract_matrix = np.array(model_embeddings)
    decs_matrix = np.array([v for v in decs_embs.values()])
    return cosine_similarity(abstract_matrix, decs_matrix)

def top_k_values(array):
    indexes = array.argsort()[-10:][::-1]
    A = set(indexes)
    B = set(list(range(array.shape[0])))
    array[list(B.difference(A))]=0
    return array

    

if __name__ == '__main__':
    descriptions = get_descriptions('../data/raw/DeCS2020.obo')
    train, dev, test = mesinesp2.data.load_dataset('../data/raw', 1)
    #x_train, y_train = transform(train, descriptions, split_sentences = False, transform_labels = False)
    x_dev, y_dev = transform(dev, descriptions, split_sentences = False, transform_labels = False)

    # We first create each embedding file. It contains the embedding representation for each DecS code.
    idf = create_decs_embeddings()

    # Cosine similarity with sbw
    #train_sbw_similarity = similarity(x_train, 'sbw', idf)
    #print(train_sbw_similarity.shape)
    dev_sbw_similarity = similarity(x_dev, 'sbw', idf)
    print(dev_sbw_similarity.shape)
    print(dev_sbw_similarity[0, :])
    #print(np.max(dev_sbw_similarity, axis=1))

    result = np.apply_along_axis(top_k_values, 1, dev_sbw_similarity)
    #print(result.shape)
    #print(result)

    for i, abs in enumerate(x_dev):
        print('-------------------------')
        # Abstract
        print(abs)
        top = np.where(result[i,:] != 0)[0]
        ar = []
        for val in top:
            ar.append(list(descriptions.keys())[val])
        # Predicted codes with the comparison of embeddings
        print('---------------------- Predicted -----------------')
        print(ar)
        print('---------------------- Real ----------------------')
        print(y_dev[i])
        print('-------------------------')
    # Cosine similarity with mix
    #train_mix_similarity = similarity(x_train, 'mix', idf)
    #print(train_mix_similarity.shape)
    #dev_mix_similarity = similarity(x_dev, 'mix', idf)
    #print(dev_mix_similarity.shape)
    
    
    


