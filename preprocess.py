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


def read_simverb():
    return pd.read_table('data/SimVerb3500.txt', header=None, usecols=[0, 1, 3]).rename(
        columns={0: 'word1', 1: 'word2', 3: 'gt_sim'})


def read_rg():
    return pd.read_table('data/RG65.csv', sep=';', header=None).rename(
        columns={0: 'word1', 1: 'word2', 2: 'gt_sim'}).apply(lambda x: x * 2.5 if x.name == 'gt_sim' else x)


def read_rw():
    return pd.read_table('data/RW2034.csv', sep=';', header=None).rename(columns={0: 'word1', 1: 'word2', 2: 'gt_sim'})


def read_verb():
    return pd.read_table('data/Verb143.csv', sep=';', header=None).rename(
        columns={0: 'word1', 1: 'word2', 2: 'gt_sim'}).apply(lambda x: x * 2.5 if x.name == 'gt_sim' else x)


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


def extract_simverb_vocabulary():
    if os.path.exists('data/SimVerb3500_vocab.pkl'):
        with open('data/SimVerb3500_vocab.pkl', 'rb') as doc:
            simverb_vocab = pickle.load(doc)
    else:
        simverb_pairs = read_simverb()
        simverb_vocab = list()
        simverb_vocab.extend(simverb_pairs['word1'].get_values())
        simverb_vocab.extend(simverb_pairs['word2'].get_values())
        simverb_vocab = list(set([w.lower() for w in simverb_vocab]))
        with open('data/SimVerb3500_vocab.pkl', 'wb') as doc:
            pickle.dump(simverb_vocab, doc)
    return simverb_vocab


def extract_rg_vocabulary():
    if os.path.exists('data/RG65_vocab.pkl'):
        with open('data/RG65_vocab.pkl', 'rb') as doc:
            rg_vocab = pickle.load(doc)
    else:
        rg_pairs = read_rg()
        rg_vocab = list()
        rg_vocab.extend(rg_pairs['word1'].get_values())
        rg_vocab.extend(rg_pairs['word2'].get_values())
        rg_vocab = list(set([w.lower() for w in rg_vocab]))
        with open('data/RG65_vocab.pkl', 'wb') as doc:
            pickle.dump(rg_vocab, doc)
    return rg_vocab


def extract_rw_vocabulary():
    if os.path.exists('data/RW2034_vocab.pkl'):
        with open('data/RW2034_vocab.pkl', 'rb') as doc:
            rw_vocab = pickle.load(doc)
    else:
        rw_pairs = read_rw()
        rw_vocab = list()
        rw_vocab.extend(rw_pairs['word1'].get_values())
        rw_vocab.extend(rw_pairs['word2'].get_values())
        rw_vocab = list(set([w.lower() for w in rw_vocab]))
        with open('data/RW2034_vocab.pkl', 'wb') as doc:
            pickle.dump(rw_vocab, doc)
    return rw_vocab


def extract_verb_vocabulary():
    if os.path.exists('data/Verb143_vocab.pkl'):
        with open('data/Verb143_vocab.pkl', 'rb') as doc:
            verb_vocab = pickle.load(doc)
    else:
        verb_pairs = read_verb()
        verb_vocab = list()
        verb_vocab.extend(verb_pairs['word1'].get_values())
        verb_vocab.extend(verb_pairs['word2'].get_values())
        verb_vocab = list(set([w.lower() for w in verb_vocab]))
        with open('data/Verb143_vocab.pkl', 'wb') as doc:
            pickle.dump(verb_vocab, doc)
    return verb_vocab


def read_dataset(dataset_name):
    if dataset_name == 'WordSim353':
        return read_wordsim()
    elif dataset_name == 'SimLex999':
        return read_simlex()
    elif dataset_name == 'SimVerb3500':
        return read_simverb()
    elif dataset_name == 'RG65':
        return read_rg()
    elif dataset_name == 'RW2034':
        return read_rw()
    elif dataset_name == 'Verb143':
        return read_verb()
    else:
        raise Exception(f'No such dataset exists: {dataset_name}')


def read_embeddings(dataset_name, emb_name, emb_type, emb_size):
    with open(f'data/embeddings/{dataset_name}_{emb_name}_{emb_type}_{emb_size}.pkl', 'rb') as doc:
        embeddings = pickle.load(doc)
    return embeddings
