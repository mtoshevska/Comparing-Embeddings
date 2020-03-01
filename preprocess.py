import pandas as pd
from nltk.tokenize import word_tokenize
import os
import _pickle as pickle
from tqdm import tqdm
from lemmatization import lemmatize


def read_multinli_corpus():
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


def extract_vocabulary(sentences, file_name):
    if os.path.exists(f'data/{file_name}_vocab.pkl'):
        with open(f'data/{file_name}_vocab.pkl', 'rb') as doc:
            vocab = pickle.load(doc)
    else:
        vocab = list()
        for _, sentence in zip(tqdm(list(range(len(sentences)))), sentences):
            vocab.extend(sentence)
        vocab = list(set(vocab))
        with open(f'data/{file_name}_vocab.pkl', 'wb') as doc:
            pickle.dump(vocab, doc)
    return vocab


def read_wordsim():
    return pd.read_table('data/WordSim353.csv', sep=',').rename(columns={'Human (mean)': 'gt_sim'})


def read_simlex():
    return pd.read_table('data/SimLex999.txt', usecols=['word1', 'word2', 'POS', 'SimLex999']).rename(
        columns={'SimLex999': 'gt_sim'})


def extract_wordsim_vocabulary():
    if os.path.exists('data/WordSim353_vocab.pkl'):
        with open('data/WordSim353_vocab.pkl', 'rb') as doc:
            wordsim_vocab = pickle.load(doc)
    else:
        wordsim_pairs = read_wordsim()
        wordsim_vocab = list()
        wordsim_vocab.extend(wordsim_pairs['word1'].get_values())
        wordsim_vocab.extend(wordsim_pairs['word2'].get_values())
        wordsim_vocab = list(set([w.lower() for w in wordsim_vocab]))
        with open('data/WordSim353_vocab.pkl', 'wb') as doc:
            pickle.dump(wordsim_vocab, doc)
    return wordsim_vocab


def extract_simlex_vocabulary():
    if os.path.exists('data/SimLex999_vocab.pkl'):
        with open('data/SimLex999_vocab.pkl', 'rb') as doc:
            simlex_vocab = pickle.load(doc)
    else:
        simlex_pairs = read_simlex()
        simlex_vocab = list()
        simlex_vocab.extend(simlex_pairs['word1'].get_values())
        simlex_vocab.extend(simlex_pairs['word2'].get_values())
        simlex_vocab = list(set([w.lower() for w in simlex_vocab]))
        with open('data/SimLex999_vocab.pkl', 'wb') as doc:
            pickle.dump(simlex_vocab, doc)
    return simlex_vocab
