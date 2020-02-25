import pandas as pd
from nltk.tokenize import word_tokenize
import os
import _pickle as pickle
from tqdm import tqdm
from lemmatization import lemmatize


def read_corpus():
    sentence_1 = pd.read_table('data/multinli_train.txt', quoting=3)['sentence1'].get_values()
    sentence_2 = pd.read_table('data/multinli_train.txt', quoting=3)['sentence2'].get_values()
    labels = pd.read_table('data/multinli_train.txt', quoting=3)['genre'].get_values()
    return sentence_1, sentence_2, labels


def tokenize_sentence(sentence):
    tokens = word_tokenize(sentence.lower())
    return tokens


def extract_tokens(sentences, file_name):
    if os.path.exists(f'data/{file_name}_tokens.pkl'):
        with open(f'data/{file_name}_tokens.pkl', 'rb') as doc:
            tokens = pickle.load(doc)
    else:
        tokens = list()
        for _, sentence in zip(tqdm(list(range(len(sentences)))), sentences):
            sentence = '' if str(sentence) == 'nan' else sentence
            tokens.append(tokenize_sentence(sentence))
        with open(f'data/{file_name}_tokens.pkl', 'wb') as doc:
            pickle.dump(tokens, doc)
    return tokens


def extract_lemmas(sentences, file_name):
    if os.path.exists(f'data/{file_name}_lemmas.pkl'):
        with open(f'data/{file_name}_lemmas.pkl', 'rb') as doc:
            lemmas = pickle.load(doc)
    else:
        lemmas = list()
        for _, sentence in zip(tqdm(list(range(len(sentences)))), sentences):
            lemmas.append(lemmatize(sentence))
        with open(f'data/{file_name}_lemmas.pkl', 'wb') as doc:
            pickle.dump(lemmas, doc)
    return lemmas
