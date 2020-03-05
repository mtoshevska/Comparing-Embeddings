import os
import pickle
import numpy as np
import gensim.models.keyedvectors as word2vec
from preprocess import extract_wordsim_vocabulary, extract_simlex_vocabulary, extract_simverb_vocabulary


def load_glove_embeddings(embeddings_type, embeddings_size, words, dataset_name):
    if os.path.exists(f'data/{dataset_name}_glove_{embeddings_type}_{embeddings_size}.pkl'):
        with open(f'data/{dataset_name}_glove_{embeddings_type}_{embeddings_size}.pkl', 'rb') as doc:
            embeddings = pickle.load(doc)
    else:
        embeddings = dict()
        emb_type = '6B' if embeddings_type == 'wikipedia' else 'twitter.27B'
        file_name = f'data/glove.{emb_type}.{embeddings_size}d.txt'
        with open(file_name, 'r', encoding='utf-8') as doc:
            line = doc.readline()
            while line != '':
                line = line.rstrip('\n').lower()
                parts = line.split(' ')
                vals = np.array(parts[1:], dtype=np.float)
                if parts[0] in words:
                    embeddings[parts[0]] = vals
                line = doc.readline()
        with open(f'data/{dataset_name}_glove_{embeddings_type}_{embeddings_size}.pkl', 'wb') as doc:
            pickle.dump(embeddings, doc)
    return embeddings


def load_word2vec_embeddings(model_file, embeddings_size, words, dataset_name):
    if os.path.exists(f'data/{dataset_name}_word2vec_{embeddings_size}.pkl'):
        with open(f'data/{dataset_name}_word2vec_{embeddings_size}.pkl', 'rb') as doc:
            embeddings = pickle.load(doc)
    else:
        embeddings = dict()
        model = word2vec.KeyedVectors.load_word2vec_format(model_file, binary=True)
        for word in words:
            if word in model.vocab:
                embeddings[word] = model[word]
        with open(f'data/{dataset_name}_word2vec_{model_file.split("-")[0]}_{embeddings_size}.pkl', 'wb') as doc:
            pickle.dump(embeddings, doc, pickle.HIGHEST_PROTOCOL)
    return embeddings


def load_conceptnet_embeddings(model_file, embeddings_size, words, dataset_name):
    if os.path.exists(f'data/{dataset_name}_numberbatch_{embeddings_size}.pkl'):
        with open(f'data/{dataset_name}_numberbatch_{embeddings_size}.pkl', 'rb') as doc:
            embeddings = pickle.load(doc)
    else:
        embeddings = dict()
        with open(model_file, 'r') as f:
            _ = f.readline()
            while True:
                line = f.readline().strip()
                if line == '':
                    break
                parts = line.split(' ')
                word = parts[0]
                vec = np.array(list(map(float, parts[1:])))
                if word in words:
                    embeddings[word] = vec

        with open(f'data/{dataset_name}_numberbatch_{embeddings_size}.pkl', 'wb') as doc:
            pickle.dump(embeddings, doc, pickle.HIGHEST_PROTOCOL)
    return embeddings


if __name__ == '__main__':
    wordsim_vocab = extract_wordsim_vocabulary()
    simlex_vocab = extract_simlex_vocabulary()
    simverb_vocab = extract_simverb_vocabulary()

    wordsim_wikipedia_50 = load_glove_embeddings('wikipedia', 50, wordsim_vocab, 'WordSim353')
    wordsim_wikipedia_300 = load_glove_embeddings('wikipedia', 300, wordsim_vocab, 'WordSim353')
    wordsim_twitter_50 = load_glove_embeddings('twitter', 50, wordsim_vocab, 'WordSim353')
    wordsim_twitter_200 = load_glove_embeddings('twitter', 200, wordsim_vocab, 'WordSim353')
    wordsim_word2vec = load_word2vec_embeddings('GoogleNews-vectors-negative300.bin.gz',
                                                300, wordsim_vocab, 'WordSim353')
    wordsim_word2vec_freebase = load_word2vec_embeddings('freebase-vectors-skipgram1000-en.bin.gz',
                                                         1000, wordsim_vocab, 'WordSim353')
    wordsim_conceptnet = load_conceptnet_embeddings('numberbatch-en-19.08.txt',
                                                    300, wordsim_vocab, 'WordSim353')

    simlex_wikipedia_50 = load_glove_embeddings('wikipedia', 50, simlex_vocab, 'SimLex999')
    simlex_wikipedia_300 = load_glove_embeddings('wikipedia', 300, simlex_vocab, 'SimLex999')
    simlex_twitter_50 = load_glove_embeddings('twitter', 50, simlex_vocab, 'SimLex999')
    simlex_twitter_200 = load_glove_embeddings('twitter', 200, simlex_vocab, 'SimLex999')
    simlex_word2vec = load_word2vec_embeddings('GoogleNews-vectors-negative300.bin.gz', 300, simlex_vocab, 'SimLex999')
    simlex_word2vec_freebase = load_word2vec_embeddings('freebase-vectors-skipgram1000-en.bin.gz',
                                                        1000, simlex_vocab, 'SimLex999')
    simlex_conceptnet = load_word2vec_embeddings('numberbatch-en-19.08.txt', 300, simlex_vocab, 'SimLex999')

    simverb_wikipedia_50 = load_glove_embeddings('wikipedia', 50, simverb_vocab, 'SimVerb3500')
    simverb_wikipedia_300 = load_glove_embeddings('wikipedia', 300, simverb_vocab, 'SimVerb3500')
    simverb_twitter_50 = load_glove_embeddings('twitter', 50, simverb_vocab, 'SimVerb3500')
    simverb_twitter_200 = load_glove_embeddings('twitter', 200, simverb_vocab, 'SimVerb3500')
    simverb_word2vec = load_word2vec_embeddings('GoogleNews-vectors-negative300.bin.gz',
                                                300, simverb_vocab, 'SimVerb3500')
    simverb_word2vec_freebase = load_word2vec_embeddings('freebase-vectors-skipgram1000-en.bin.gz',
                                                         1000, simverb_vocab, 'SimVerb3500')
    simverb_conceptnet = load_word2vec_embeddings('numberbatch-en-19.08.txt',
                                                  300, simverb_vocab, 'SimVerb3500')
