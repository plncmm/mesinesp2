
from tokenizer import get_descriptions, transform
from vectorizer import get_idf, get_pretrained_embeddings, get_embeddings_from_text, create_decs_embeddings
import sys
import json
import time
import fasttext
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
import mesinesp2
from metrics import f1_score
from utils import create_json
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

def top_k_values(array, k):
    indexes = array.argsort()[-k:][::-1]
    return indexes

    

if __name__ == '__main__':
    descriptions = get_descriptions('../data/raw/DeCS2020.obo')
    _, dev, _ = mesinesp2.data.load_dataset('../data/raw', 1)
    x_dev, y_dev = transform(dev, descriptions, split_sentences = False, transform_labels = False)
    model = 'sbw'
    
    #idf = create_decs_embeddings() # Run just in case you don't have decs_mix decs_sbw and idf json files
    with open('../embeddings/idf.json') as f:
        idf = json.load(f)

    dev_sbw_similarity = similarity(x_dev, model, idf)

    result = np.apply_along_axis(top_k_values, 1, dev_sbw_similarity, 100)
    create_json(dev['id'], result, descriptions, model)

    with open(f'../embeddings/{model}_predictions.json') as json_file:
        data = json.load(json_file)

    pred = []
    for doc in data['documents']:
        pred.append(doc['labels'])
        
    real = dev["decsCodes"]
    assert(len(real)==len(pred))
    tp, fn, fp, p, r, f1 = f1_score(real, pred)
    print(F'TP: {tp}')
    print(F'FN: {fn}')
    print(F'FP: {fp}')
    print(f'Precision: {p}')
    print(f'Recall: {r}')
    print(f'F1-Score: {f1}')


    
    
    
    


