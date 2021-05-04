
import re
import time
import codecs
import pathlib
import os
import sys
sys.path.append('..')
import mesinesp2.data

def get_tokens(text):
    text = re.sub("\s+"," ", text)
    text = re.split("(\W)", text)
    tokens = list(filter(lambda a: a != "", text))
    tokens = list(filter(lambda a: a != " ", tokens))
    return tokens

def tokenizer(row, split_sentences, is_df = True):
    if is_df: row = row['title'] + '. ' + row['abstractText']
    if split_sentences:
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', row) # Reference: https://www.semicolonworld.com/question/58276/python-regex-for-splitting-text-into-sentences-sentence-tokenizing
        tokens = [get_tokens(sent) for sent in sentences]
    else:
        tokens = get_tokens(row)
    return tokens

def transform(df, descriptions, split_sentences = False):
    print(f'Transforming {df.shape[0]} articles..')
    start = time.time()
    x = df.apply(tokenizer, axis = 1, args = [split_sentences, True])
    y = get_labels(df['decsCodes'], descriptions)
    print(f'{df.shape[0]} articles transformed in {(time.time()-start)/60} minutes.')
    return x, y

def get_labels(df, descriptions):
    y = []
    for codes in df:
        y_temp = []
        for code in codes:
            if code in descriptions:
                y_temp.append(descriptions[code])
            else:
                print(f'{code}: Description not founded.')
        y.append(y_temp)
    return y


def get_descriptions(filepath, tokenize_definition = True, tokenize_name = True):
    f = codecs.open(filepath, 'r', 'utf-8').read()
    descriptions = {}
    terms = f.split('[Term]')[1:]
    for term in terms:
        for line in term.splitlines():
            if line == '':
                continue
            line_info = line.split()
            if line.startswith('id:'):
                id = line_info[1][1:-1]
            if line.startswith('name:'):
                if tokenize_name:
                    descriptions[id] = {"name" : tokenizer(line[7:-1], split_sentences = False, is_df = False)}
                else:
                    descriptions[id] = {"name" : line[7:-1]}
            if line.startswith('def:'):
                if tokenize_definition:
                    descriptions[id]["def"] = tokenizer(line[7:-2], split_sentences = False, is_df = False)
                else:
                    descriptions[id]["def"] = line[7:-2]
    return descriptions
                
    


        


if __name__=='__main__':
    descriptions = get_descriptions('../data/raw/DeCS2020.obo')
    train, dev, test = mesinesp2.data.load_dataset('../data/raw', 1)
    
    x_train, y_train = transform(train, descriptions, split_sentences = True)
    x_dev, y_dev = transform(dev, descriptions, split_sentences = True)