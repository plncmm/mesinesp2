
import re
import time
import mesinesp2.data

def get_tokens(text):
    text = re.sub("\s+"," ", text)
    text = re.split("(\W)", text)
    tokens = list(filter(lambda a: a != "", text))
    tokens = list(filter(lambda a: a != " ", tokens))
    return tokens

def tokenizer(row, split_sentences):
    text = row['title'] + '. ' + row['abstractText']
    if split_sentences:
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text) # Reference: https://www.semicolonworld.com/question/58276/python-regex-for-splitting-text-into-sentences-sentence-tokenizing
        tokens = [get_tokens(sent) for sent in sentences]
    else:
        tokens = get_tokens(text)
    return tokens

def transform(df, split_sentences = False):
    print(f'Transforming {df.shape[0]} articles..')
    start = time.time()
    df = df.apply(tokenizer, axis = 1, args = [split_sentences])
    print(f'{df.shape[0]} articles tokenized in {(time.time()-start)/60} minutes.')

if __name__=='__main__':
    train, dev, test = mesinesp2.data.load_dataset('data/raw', 1)
    train = transform(train, split_sentences = True)
    dev = transform(dev, split_sentences = True)
    test = transform(test, split_sentences = True)